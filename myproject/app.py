import os
import asyncio
import logging
import functools
from fastapi import FastAPI, HTTPException, Query
from marqo import Client
from tenacity import retry, stop_after_attempt, wait_fixed
from pydantic import BaseModel
from typing import List, Dict, Optional
import httpx

# -------------------------
# Setup
# -------------------------
app = FastAPI(title="Community Assistance API", version="1.0.0")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MARQO_URL = os.getenv("MARQO_URL", "http://localhost:8882")
INDEX_NAME = os.getenv("INDEX_NAME", "community_assistance_index")

# Retry config for Tenacity
MARQO_RETRY_CONFIG = {
    "stop": stop_after_attempt(3),
    "wait": wait_fixed(2),
    "reraise": True,
}

# Global client
marqo_client = None

# Topic data for fallback responses
topic_data = {
    "education": {
        "short": "Education help includes FAFSA guidance, GED programs, and tutoring.",
        "long": "Education assistance covers a wide range of services including FAFSA guidance for financial aid, GED programs for completing high school equivalency, tutoring for academic support, and access to free online learning platforms.",
        "references": ["https://studentaid.gov", "https://www.khanacademy.org", "https://www.unicef.org/education"],
        "related": ["How to apply for FAFSA and what documents are needed?", "Are there free GED programs near me?"]
    },
    "food": {
        "short": "Food assistance includes food banks and meal programs.",
        "long": "Food assistance includes access to community food banks, meal delivery programs for seniors and vulnerable groups, nutrition education, and emergency grocery support for families in need.",
        "references": ["https://www.feedingamerica.org", "https://www.fns.usda.gov/snap", "https://www.wfp.org/food-assistance"],
        "related": ["Where is the nearest food bank?", "Am I eligible for SNAP benefits?"]
    },
    "transport": {
        "short": "Transport help includes bus passes and ride assistance.",
        "long": "Transport assistance services provide free or subsidized bus passes, special transport for disabled individuals, ride assistance for medical visits, and community shuttle programs.",
        "references": ["https://www.transit.dot.gov", "https://www.uber.com/us/en/ride/", "https://www.lyft.com"],
        "related": ["How do I apply for reduced bus fare?", "Is there transport help for medical appointments?"]
    },
    "healthcare": {
        "short": "Healthcare support includes clinics, mental health, and wellness programs.",
        "long": "Healthcare support covers free and low-cost clinics, preventive screenings, prescription assistance, mental health counseling, addiction recovery programs, and wellness resources for families.",
        "references": ["https://www.who.int", "https://www.cdc.gov", "https://www.healthcare.gov"],
        "related": ["Where can I find a free clinic nearby?", "What mental health services are available?"]
    }
}

class QAResponse(BaseModel):
    question: str
    answer: str
    references: List[str]
    related_questions: List[str]

# -------------------------
# Retry Wrapper
# -------------------------
def with_marqo_retry(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        retry_decorator = retry(**MARQO_RETRY_CONFIG)
        if asyncio.iscoroutinefunction(func):
            wrapped = retry_decorator(func)
            return await wrapped(*args, **kwargs)
        else:
            wrapped = retry_decorator(func)
            return wrapped(*args, **kwargs)

    return wrapper


# -------------------------
# Initialize Client
# -------------------------
@app.on_event("startup")
async def startup_event():
    global marqo_client
    try:
        marqo_client = Client(url=MARQO_URL)
        logger.info("✅ Marqo client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to init Marqo client: {e}")
        marqo_client = None


# -------------------------
# Health check
# -------------------------
@app.get("/health")
async def health():
    if not marqo_client:
        raise HTTPException(status_code=503, detail="Marqo client not available")
    return {"status": "ok"}


# -------------------------
# Safe Document Management Functions
# -------------------------
async def get_existing_document_ids():
    """Get IDs of all existing documents in the index"""
    if not marqo_client:
        return set()
    
    try:
        # Search for all documents to get their IDs
        results = marqo_client.index(INDEX_NAME).search("", limit=1000)
        existing_ids = {hit["_id"] for hit in results.get("hits", [])}
        return existing_ids
    except Exception as e:
        logger.warning(f"Could not fetch existing document IDs: {e}")
        return set()

async def safe_add_documents(documents):
    """Add documents only if they don't already exist"""
    if not marqo_client:
        return {"added": 0, "skipped": len(documents)}
    
    existing_ids = await get_existing_document_ids()
    documents_to_add = [doc for doc in documents if doc["_id"] not in existing_ids]
    
    if not documents_to_add:
        logger.info("✅ All documents already exist in index")
        return {"added": 0, "skipped": len(documents)}
    
    try:
        results = marqo_client.index(INDEX_NAME).add_documents(
            documents_to_add, 
            tensor_fields=["content", "title"],
            client_batch_size=2
        )
        logger.info(f"✅ Added {len(documents_to_add)} new documents, skipped {len(documents) - len(documents_to_add)} existing ones")
        return {"added": len(documents_to_add), "skipped": len(documents) - len(documents_to_add)}
    except Exception as e:
        logger.error(f"❌ Error adding documents: {e}")
        raise


# -------------------------
# Init Marqo Index with Documents - FIXED VERSION
# -------------------------
@app.get("/init-marqo-client")
@with_marqo_retry
async def init_marqo_with_client():
    if not marqo_client:
        raise HTTPException(status_code=503, detail="Marqo client not available")

    try:
        # Check if index exists
        indexes = marqo_client.get_indexes()
        index_exists = any(idx['indexName'] == INDEX_NAME for idx in indexes.get('results', []))

        action_taken = "none"
        
        if not index_exists:
            marqo_client.create_index(INDEX_NAME)
            logger.info(f"✅ Created new index: {INDEX_NAME}")
            await asyncio.sleep(2)  # wait for readiness
            action_taken = "created"
        else:
            logger.info(f"ℹ️ Index already exists: {INDEX_NAME}")
            action_taken = "exists"

        # Prepare default documents
        documents = []
        for topic, data in topic_data.items():
            documents.append({
                "_id": f"doc_{topic}",
                "content": data["long"],
                "topic": topic,
                "references": data["references"],
                "title": f"{topic.capitalize()} Assistance"
            })

        # SAFELY add documents (only if they don't exist)
        add_result = await safe_add_documents(documents)

        # Refresh the index to make documents searchable
        async with httpx.AsyncClient() as client:
            refresh_response = await client.post(f"{MARQO_URL}/indexes/{INDEX_NAME}/refresh")

        return {
            "status": "success", 
            "index": INDEX_NAME,
            "action_taken": action_taken,
            "documents_added": add_result["added"],
            "documents_skipped": add_result["skipped"],
            "refresh_status": refresh_response.status_code,
            "message": "Index initialized safely without data loss"
        }

    except Exception as e:
        logger.error(f"❌ Error initializing index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Add Docs (Safe version)
# -------------------------
class DocsInput(BaseModel):
    docs: List[Dict]


@app.post("/add-docs")
@with_marqo_retry
async def add_docs(input_data: DocsInput):
    if not marqo_client:
        raise HTTPException(status_code=503, detail="Marqo client not available")

    try:
        # Ensure each document has an _id field
        for doc in input_data.docs:
            if "_id" not in doc:
                doc["_id"] = f"custom_doc_{hash(str(doc))}"
        
        add_result = await safe_add_documents(input_data.docs)
        return {
            "status": "success", 
            "added": add_result["added"],
            "skipped": add_result["skipped"]
        }
    except Exception as e:
        logger.error(f"❌ Error adding documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Ask Endpoint (RAG Query)
# -------------------------
@app.get("/ask", response_model=QAResponse)
async def ask_question(
    topic: str = Query(..., description="Choose a topic: education, food, transport, healthcare"),
    detail: str = Query("short", description="Choose 'short' or 'long' answer"),
    query: Optional[str] = Query(None, description="Optional specific question for RAG")
):
    # If a specific query is provided, use RAG search
    if query:
        if not marqo_client:
            # Fallback to topic data if Marqo client is not available
            if topic.lower() in topic_data:
                data = topic_data[topic.lower()]
                answer = data["long"] if detail == "long" else data["short"]
                return {
                    "question": query,
                    "answer": f"Search unavailable. {answer}",
                    "references": data["references"],
                    "related_questions": data["related"]
                }
            else:
                return {
                    "question": query,
                    "answer": "Search unavailable and no topic data found.",
                    "references": [],
                    "related_questions": []
                }
        
        try:
            # Use Marqo client for RAG search
            results = marqo_client.index(INDEX_NAME).search(
                q=query,
                filter_string=f"topic:{topic}",
                limit=3
            )
            
            if results.get('hits'):
                rag_content = " ".join([hit["content"] for hit in results['hits']])
                references = list({ref for hit in results['hits'] for ref in hit.get("metadata", {}).get("references", [])})
                return {
                    "question": query,
                    "answer": rag_content,
                    "references": references,
                    "related_questions": [f"More about {topic} programs", f"How to apply for {topic} assistance"]
                }
            else:
                # Fallback to topic data if no search results
                if topic.lower() in topic_data:
                    data = topic_data[topic.lower()]
                    answer = data["long"] if detail == "long" else data["short"]
                    return {
                        "question": query,
                        "answer": f"No specific results found. {answer}",
                        "references": data["references"],
                        "related_questions": data["related"]
                    }
                else:
                    return {
                        "question": query,
                        "answer": f"No information found about '{query}' in {topic}.",
                        "references": [],
                        "related_questions": []
                    }
        except Exception as e:
            logger.error(f"❌ Error running query: {e}")
            # Fallback to topic data on error
            if topic.lower() in topic_data:
                data = topic_data[topic.lower()]
                answer = data["long"] if detail == "long" else data["short"]
                return {
                    "question": query,
                    "answer": f"Search error. {answer}",
                    "references": data["references"],
                    "related_questions": data["related"]
                }
            else:
                return {
                    "question": query,
                    "answer": f"Error searching and no topic data: {str(e)}",
                    "references": [],
                    "related_questions": []
                }

    # If no query provided, return topic information
    if topic.lower() not in topic_data:
        return {
            "question": topic,
            "answer": "Sorry, no data available for this topic.",
            "references": [],
            "related_questions": []
        }

    data = topic_data[topic.lower()]
    answer = data["long"] if detail == "long" else data["short"]
    return {
        "question": topic,
        "answer": answer,
        "references": data["references"],
        "related_questions": data["related"]
    }


# -------------------------
# Additional endpoints for testing
# -------------------------
@app.get("/check-index")
async def check_index():
    if not marqo_client:
        raise HTTPException(status_code=503, detail="Marqo client not available")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{MARQO_URL}/indexes/{INDEX_NAME}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Index not found: {response.text}"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/list-indexes")
async def list_indexes():
    if not marqo_client:
        raise HTTPException(status_code=503, detail="Marqo client not available")
    
    try:
        indexes = marqo_client.get_indexes()
        return indexes
    except Exception as e:
        return {"error": str(e)}


@app.get("/list-documents")
async def list_documents(limit: int = 10):
    if not marqo_client:
        raise HTTPException(status_code=503, detail="Marqo client not available")
    
    try:
        results = marqo_client.index(INDEX_NAME).search("", limit=limit)
        documents = [{"id": hit["_id"], "topic": hit.get("topic", "unknown"), "title": hit.get("title", "No title")} 
                    for hit in results.get("hits", [])]
        return {
            "total_found": results.get("limit", 0),
            "documents": documents
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)