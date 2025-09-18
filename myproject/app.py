# app.py
import os
import asyncio
import logging
import functools
import time
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse
from marqo import Client
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception, RetryError
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import httpx
import re

# -------------------------
# Setup
# -------------------------
app = FastAPI(title="Community Assistance API", version="1.0.0")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MARQO_URL = os.getenv("MARQO_URL", "http://localhost:8882")
INDEX_NAME = os.getenv("INDEX_NAME", "community_assistance_index")

# Configuration
MAX_RESPONSE_LENGTH = 2000  # Character limit for responses
MAX_REFERENCES = 5          # Maximum number of references to return
MAX_RELATED_QUESTIONS = 3   # Maximum number of related questions

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

# -------------------------
# Request / Response Models
# -------------------------
class DocsInput(BaseModel):
    """Model for /add-docs endpoint body"""
    docs: List[Dict[str, Any]] = Field(..., description="List of documents to add to the index")

class QAResponse(BaseModel):
    question: str = Field(..., description="The question that was asked")
    answer: str = Field(..., description="The answer to the question")
    references: List[str] = Field(..., description="List of reference URLs")
    related_questions: List[str] = Field(..., description="List of related questions")
    truncated: bool = Field(False, description="Indicates if response was truncated")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    status_code: int = Field(..., description="HTTP status code")

# -------------------------
# Response Size Control Utilities
# -------------------------
def truncate_text(text: str, max_length: int) -> tuple[str, bool]:
    """Truncate text to maximum length and indicate if truncation occurred"""
    if len(text) <= max_length:
        return text, False
    return text[:max_length].rsplit(' ', 1)[0] + "...", True

def clean_and_deduplicate_references(references: List[str]) -> List[str]:
    """Clean, validate, and deduplicate reference URLs"""
    unique_refs = set()
    cleaned_refs = []
    
    for ref in references:
        # Clean the URL
        cleaned_ref = ref.strip()
        if not cleaned_ref:
            continue
            
        # Basic URL validation
        if not cleaned_ref.startswith(('http://', 'https://')):
            cleaned_ref = 'https://' + cleaned_ref
            
        # Deduplicate
        if cleaned_ref not in unique_refs:
            unique_refs.add(cleaned_ref)
            cleaned_refs.append(cleaned_ref)
            
    return cleaned_refs[:MAX_REFERENCES]

def smart_truncate_response(answer: str, references: List[str], related_questions: List[str]) -> tuple[str, List[str], List[str], bool]:
    """
    Smart truncation that preserves important content
    Returns: (truncated_answer, truncated_references, truncated_related_questions, was_truncated)
    """
    was_truncated = False
    
    # Truncate answer if needed
    if len(answer) > MAX_RESPONSE_LENGTH:
        answer, was_truncated = truncate_text(answer, MAX_RESPONSE_LENGTH)
    
    # Limit references
    if len(references) > MAX_REFERENCES:
        references = references[:MAX_REFERENCES]
        was_truncated = True
    
    # Limit related questions
    if len(related_questions) > MAX_RELATED_QUESTIONS:
        related_questions = related_questions[:MAX_RELATED_QUESTIONS]
        was_truncated = True
    
    return answer, references, related_questions, was_truncated

# -------------------------
# RAG Concatenation Utilities
# -------------------------
def concatenate_rag_content(hits: List[Dict]) -> str:
    """Smart concatenation of RAG content with proper formatting"""
    if not hits:
        return ""
    
    content_parts = []
    seen_content = set()
    
    for hit in hits:
        content = hit.get("content", "").strip()
        if content and content not in seen_content:
            seen_content.add(content)
            content_parts.append(content)
    
    # Join with proper spacing and avoid wall of text
    concatenated = " ".join(content_parts)
    
    # Clean up excessive whitespace
    concatenated = re.sub(r'\s+', ' ', concatenated).strip()
    
    return concatenated

def extract_references_from_hits(hits: List[Dict]) -> List[str]:
    """Extract and validate references from search hits"""
    references = set()
    
    for hit in hits:
        # Extract from metadata
        metadata = hit.get("metadata", {})
        hit_references = metadata.get("references", [])
        
        if isinstance(hit_references, list):
            for ref in hit_references:
                if isinstance(ref, str) and ref.strip():
                    references.add(ref.strip())
        
        # Also check if references exist in the content itself
        content = hit.get("content", "")
        if content and isinstance(content, str):
            # Simple URL extraction from content
            url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
            found_urls = re.findall(url_pattern, content)
            for url in found_urls:
                references.add(url)
    
    return clean_and_deduplicate_references(list(references))

# -------------------------
# Custom Exceptions
# -------------------------
class MarqoConnectionError(HTTPException):
    def __init__(self, detail: str = "Marqo service unavailable"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail
        )

class IndexNotReadyError(HTTPException):
    def __init__(self, detail: str = "Index is not ready"):
        super().__init__(
            status_code=status.HTTP_425_TOO_EARLY,
            detail=detail
        )

class DocumentOperationError(HTTPException):
    def __init__(self, detail: str = "Document operation failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )

class ResponseTooLargeError(HTTPException):
    def __init__(self, detail: str = "Response too large"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )

# -------------------------
# Polling Functions
# -------------------------
async def wait_for_index_ready(timeout: int = 30, check_interval: int = 1) -> bool:
    """Poll Marqo index until it's ready for operations"""
    start_time = time.time()
    attempt = 0
    
    while time.time() - start_time < timeout:
        attempt += 1
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{MARQO_URL}/indexes/{INDEX_NAME}/stats",
                    timeout=5
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ Index ready after {attempt} attempts")
                    return True
                    
        except (httpx.HTTPError, ConnectionError) as e:
            logger.debug(f"Index not ready yet (attempt {attempt}): {e}")
        
        await asyncio.sleep(min(check_interval * (1.5 ** attempt), 5))
    
    logger.error(f"❌ Index not ready after {timeout} seconds")
    raise IndexNotReadyError(f"Index not ready after {timeout} seconds")

async def wait_for_marqo_health(timeout: int = 30) -> bool:
    """Poll Marqo server until it's healthy"""
    start_time = time.time()
    attempt = 0
    
    while time.time() - start_time < timeout:
        attempt += 1
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{MARQO_URL}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Marqo server healthy")
                    return True
                    
        except (httpx.HTTPError, ConnectionError) as e:
            logger.debug(f"Marqo not healthy yet (attempt {attempt}): {e}")
        
        await asyncio.sleep(min(1 * (1.5 ** attempt), 3))
    
    logger.error("❌ Marqo server not healthy")
    raise MarqoConnectionError("Marqo server not healthy after timeout")

# -------------------------
# Simple Circuit Breaker Simulation
# -------------------------
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.is_open = False

    def __call__(self, func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_time = time.time()
            
            if self.is_open:
                if current_time - self.last_failure_time > self.recovery_timeout:
                    logger.info("⚠️ Circuit breaker half-open - testing recovery")
                    self.is_open = False
                else:
                    raise MarqoConnectionError("Service temporarily unavailable due to repeated failures")
            
            try:
                result = await func(*args, **kwargs)
                self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = current_time
                
                if self.failure_count >= self.failure_threshold:
                    self.is_open = True
                    logger.error(f"🔌 Circuit breaker opened after {self.failure_count} failures")
                
                raise
        return wrapper

# Create circuit breaker instance
marqo_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

# -------------------------
# Retry Wrapper
# -------------------------
def with_marqo_retry(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            retry_decorator = retry(**MARQO_RETRY_CONFIG)
            if asyncio.iscoroutinefunction(func):
                wrapped = retry_decorator(func)
                return await wrapped(*args, **kwargs)
            else:
                wrapped = retry_decorator(func)
                return wrapped(*args, **kwargs)
        except RetryError as e:
            logger.error(f"All retry attempts failed: {e}")
            raise MarqoConnectionError("Service unavailable after multiple retry attempts")
        except Exception as e:
            logger.error(f"Unexpected error in retry wrapper: {e}")
            raise
    return wrapper

# -------------------------
# Initialize Client
# -------------------------
@app.on_event("startup")
async def startup_event():
    global marqo_client
    try:
        if not await wait_for_marqo_health():
            logger.error("Marqo server not available during startup")
            marqo_client = None
            return
            
        marqo_client = Client(url=MARQO_URL)
        
        try:
            marqo_client.get_indexes()
            logger.info("✅ Marqo client initialized and connected")
        except Exception as e:
            logger.error(f"❌ Marqo client connected but API failed: {e}")
            marqo_client = None
            
    except Exception as e:
        logger.error(f"❌ Failed to init Marqo client: {e}")
        marqo_client = None

# -------------------------
# Health check
# -------------------------
@app.get("/health", response_model=Dict[str, Any])
async def health():
    if not marqo_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Marqo client not available"
        )
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{MARQO_URL}/health", timeout=5)
            marqo_health = response.status_code == 200
    except Exception as e:
        logger.error(f"Marqo health check failed: {e}")
        marqo_health = False
    
    return {
        "status": "healthy" if marqo_health else "degraded",
        "marqo_connected": marqo_health,
        "client_initialized": marqo_client is not None,
        "timestamp": time.time()
    }

# -------------------------
# Document Management
# -------------------------
async def get_existing_document_ids():
    if not marqo_client:
        return set()
    
    try:
        results = marqo_client.index(INDEX_NAME).search("", limit=1000) or {}
        existing_ids = {hit["_id"] for hit in results.get("hits", [])}
        return existing_ids
    except Exception as e:
        logger.warning(f"Could not fetch existing document IDs: {e}")
        raise DocumentOperationError(f"Failed to fetch document IDs: {str(e)}")

async def safe_add_documents(documents):
    if not marqo_client:
        return {"added": 0, "skipped": len(documents)}
    
    try:
        existing_ids = await get_existing_document_ids()
        documents_to_add = [doc for doc in documents if doc["_id"] not in existing_ids]
        
        if not documents_to_add:
            logger.info("✅ All documents already exist in index")
            return {"added": 0, "skipped": len(documents)}
        
        results = marqo_client.index(INDEX_NAME).add_documents(
            documents_to_add, 
            tensor_fields=["content", "title"],
            client_batch_size=2
        )
        logger.info(f"✅ Added {len(documents_to_add)} new documents, skipped {len(documents) - len(documents_to_add)} existing ones")
        return {"added": len(documents_to_add), "skipped": len(documents) - len(documents_to_add)}
    except Exception as e:
        logger.error(f"❌ Error adding documents: {e}")
        raise DocumentOperationError(f"Failed to add documents: {str(e)}")

# -------------------------
# Main Endpoints with Size Control
# -------------------------
@app.get("/init-marqo-client", response_model=Dict[str, Any])
@marqo_circuit_breaker
@with_marqo_retry
async def init_marqo_with_client():
    if not marqo_client:
        raise MarqoConnectionError("Marqo client not available")

    try:
        indexes = marqo_client.get_indexes()
        index_exists = any(idx.get('indexName') == INDEX_NAME for idx in indexes.get('results', []))

        action_taken = "none"
        
        if not index_exists:
            marqo_client.create_index(INDEX_NAME)
            logger.info(f"✅ Created new index: {INDEX_NAME}")
            
            if not await wait_for_index_ready():
                raise IndexNotReadyError("Index creation timeout - index not ready")
                
            action_taken = "created"
        else:
            logger.info(f"ℹ️ Index already exists: {INDEX_NAME}")
            action_taken = "exists"

        documents = []
        for topic, data in topic_data.items():
            documents.append({
                "_id": f"doc_{topic}",
                "content": data["long"],
                "topic": topic,
                "references": data["references"],
                "title": f"{topic.capitalize()} Assistance"
            })

        add_result = await safe_add_documents(documents)

        async with httpx.AsyncClient() as client:
            refresh_response = await client.post(f"{MARQO_URL}/indexes/{INDEX_NAME}/refresh")

        return {
            "status": "success", 
            "index": INDEX_NAME,
            "action_taken": action_taken,
            "documents_added": add_result["added"],
            "documents_skipped": add_result["skipped"],
            "refresh_status": refresh_response.status_code,
            "message": "Index initialized safely without data loss",
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"❌ Error initializing index: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize index: {str(e)}"
        )

@app.post("/add-docs", response_model=Dict[str, Any])
@marqo_circuit_breaker
@with_marqo_retry
async def add_docs(input_data: DocsInput):
    if not marqo_client:
        raise MarqoConnectionError("Marqo client not available")

    try:
        for doc in input_data.docs:
            if "_id" not in doc:
                # stable-ish fallback id
                doc["_id"] = f"custom_doc_{abs(hash(str(doc))) % (10**12)}"
        
        add_result = await safe_add_documents(input_data.docs)
        return {
            "status": "success", 
            "added": add_result["added"],
            "skipped": add_result["skipped"],
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"❌ Error adding documents: {e}")
        if isinstance(e, HTTPException):
            raise
        raise DocumentOperationError(f"Failed to add documents: {str(e)}")

@app.get("/ask", response_model=QAResponse, responses={
    404: {"model": ErrorResponse, "description": "Topic not found"},
    503: {"model": ErrorResponse, "description": "Service unavailable"},
    500: {"model": ErrorResponse, "description": "Internal server error"}
})
@marqo_circuit_breaker
async def ask_question(
    topic: str = Query(..., description="Choose a topic: education, food, transport, healthcare"),
    detail: str = Query("short", description="Choose 'short' or 'long' answer"),
    query: Optional[str] = Query(None, description="Optional specific question for RAG")
):
    # If a specific query is provided, use RAG search
    if query:
        if not marqo_client:
            if topic.lower() in topic_data:
                data = topic_data[topic.lower()]
                answer = data["long"] if detail == "long" else data["short"]
                answer, refs, related, truncated = smart_truncate_response(
                    answer, data["references"], data["related"]
                )
                return {
                    "question": query,
                    "answer": f"Search unavailable. {answer}",
                    "references": refs,
                    "related_questions": related,
                    "truncated": truncated
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Topic '{topic}' not found and search unavailable"
                )
        
        try:
            results = marqo_client.index(INDEX_NAME).search(
                q=query,
                filter_string=f"topic:{topic}",
                limit=3
            ) or {}
            
            hits = results.get('hits', [])
            if hits:
                # Proper RAG concatenation with size control
                rag_content = concatenate_rag_content(hits)
                references = extract_references_from_hits(hits)
                related_questions = [f"More about {topic} programs", f"How to apply for {topic} assistance"]
                
                # Apply size control
                answer, references, related_questions, truncated = smart_truncate_response(
                    rag_content, references, related_questions
                )
                
                return {
                    "question": query,
                    "answer": answer,
                    "references": references,
                    "related_questions": related_questions,
                    "truncated": truncated
                }
            else:
                if topic.lower() in topic_data:
                    data = topic_data[topic.lower()]
                    answer = data["long"] if detail == "long" else data["short"]
                    answer, refs, related, truncated = smart_truncate_response(
                        f"No specific results found. {answer}", 
                        data["references"], 
                        data["related"]
                    )
                    return {
                        "question": query,
                        "answer": answer,
                        "references": refs,
                        "related_questions": related,
                        "truncated": truncated
                    }
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"No information found about '{query}' in topic '{topic}'"
                    )
        except Exception as e:
            logger.error(f"❌ Error running query: {e}")
            if topic.lower() in topic_data:
                data = topic_data[topic.lower()]
                answer = data["long"] if detail == "long" else data["short"]
                answer, refs, related, truncated = smart_truncate_response(
                    f"Search error. {answer}", 
                    data["references"], 
                    data["related"]
                )
                return {
                    "question": query,
                    "answer": answer,
                    "references": refs,
                    "related_questions": related,
                    "truncated": truncated
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Search failed and no fallback data available: {str(e)}"
                )

    # If no query provided, return topic information
    if topic.lower() not in topic_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Topic '{topic}' not found"
        )

    data = topic_data[topic.lower()]
    answer = data["long"] if detail == "long" else data["short"]
    answer, references, related_questions, truncated = smart_truncate_response(
        answer, data["references"], data["related"]
    )
    
    return {
        "question": topic,
        "answer": answer,
        "references": references,
        "related_questions": related_questions,
        "truncated": truncated
    }

# -------------------------
# Additional endpoints
# -------------------------
@app.get("/check-index", response_model=Dict[str, Any])
@marqo_circuit_breaker
async def check_index():
    if not marqo_client:
        raise MarqoConnectionError("Marqo client not available")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{MARQO_URL}/indexes/{INDEX_NAME}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Index not found: {response.text}"
            )
    except Exception as e:
        logger.error(f"Error checking index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check index: {str(e)}"
        )

@app.get("/list-indexes", response_model=Dict[str, Any])
@marqo_circuit_breaker
async def list_indexes():
    if not marqo_client:
        raise MarqoConnectionError("Marqo client not available")
    
    try:
        indexes = marqo_client.get_indexes()
        return indexes
    except Exception as e:
        logger.error(f"Error listing indexes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list indexes: {str(e)}"
        )

@app.get("/list-documents", response_model=Dict[str, Any])
@marqo_circuit_breaker
async def list_documents(limit: int = Query(10, ge=1, le=100)):
    if not marqo_client:
        raise MarqoConnectionError("Marqo client not available")
    
    try:
        results = marqo_client.index(INDEX_NAME).search("", limit=limit) or {}
        hits = results.get("hits", [])
        documents = [{"id": hit["_id"], "topic": hit.get("topic", "unknown"), "title": hit.get("title", "No title")} 
                    for hit in hits]
        return {
            "total_found": len(hits),
            "documents": documents,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )

# -------------------------
# Global Exception Handler
# -------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    error_response = ErrorResponse(
        detail=exc.detail,
        error_type=exc.__class__.__name__,
        status_code=exc.status_code
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    error_response = ErrorResponse(
        detail="Internal server error",
        error_type="InternalServerError",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
