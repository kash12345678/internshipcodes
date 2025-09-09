from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import time
from marqo import Client

app = FastAPI(title="Community Assistance API", version="1.0.0")
MARQO_URL = "http://localhost:8883"
INDEX_NAME = "community_assistance_index"

# Initialize Marqo client
marqo_client = Client(url=MARQO_URL)

class QAResponse(BaseModel):
    question: str
    answer: str
    references: List[str]
    related_questions: List[str]

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

@app.get("/")
def read_root():
    return {"message": "Server is running"}

@app.get("/ask", response_model=QAResponse)
def ask_question(
    topic: str = Query(..., description="Choose a topic: education, food, transport, healthcare"),
    detail: str = Query("short", description="Choose 'short' or 'long' answer"),
    query: Optional[str] = Query(None, description="Optional specific question for RAG")
):
    if query:
        try:
            response = requests.post(
                f"{MARQO_URL}/indexes/{INDEX_NAME}/search",
                json={"q": query, "filter": f"topic:{topic}", "limit": 3},
                timeout=10
            )
            if response.status_code == 200:
                results = response.json()
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
                    return {
                        "question": query,
                        "answer": f"No specific information found about '{query}' in {topic}. Please try a different query or topic.",
                        "references": [],
                        "related_questions": []
                    }
            else:
                return {"question": query, "answer": "Marqo search failed. Using fallback data.", "references": [], "related_questions": []}
        except Exception as e:
            return {"question": query, "answer": f"Error searching for information: {str(e)}", "references": [], "related_questions": []}

    if topic.lower() not in topic_data:
        return {"question": topic, "answer": "Sorry, no data available for this topic.", "references": [], "related_questions": []}

    data = topic_data[topic.lower()]
    answer = data["long"] if detail == "long" else data["short"]
    return {"question": topic, "answer": answer, "references": data["references"], "related_questions": data["related"]}

@app.get("/init-marqo-client")
def init_marqo_with_client():
    try:
        # Delete index if exists
        try:
            marqo_client.index(INDEX_NAME).delete()
            time.sleep(1)
            print("Deleted existing index")
        except Exception as e:
            print(f"Index may not exist or delete failed: {e}")
        
        # Create index
        marqo_client.create_index(INDEX_NAME)
        time.sleep(2)
        print("Created new index")
        
        # Add documents
        documents = []
        for topic, data in topic_data.items():
            documents.append({
                "_id": f"doc_{topic}",
                "content": data["long"],
                "topic": topic,
                "references": data["references"],
                "title": f"{topic.capitalize()} Assistance"
            })
        
        # Add documents - REMOVED refresh parameter
        results = marqo_client.index(INDEX_NAME).add_documents(
            documents, 
            tensor_fields=["content", "title"],
            client_batch_size=2
        )
        
        # Manually refresh to make documents searchable
        time.sleep(1)
        refresh_response = requests.post(f"{MARQO_URL}/indexes/{INDEX_NAME}/refresh")
        
        return {
            "message": "Documents added successfully using Marqo client!",
            "documents_added": len(documents),
            "refresh_status": refresh_response.status_code
        }
        
    except Exception as e:
        return {"error": f"Marqo client failed: {str(e)}"}

@app.get("/check-marqo")
def check_marqo():
    try:
        response = requests.get(f"{MARQO_URL}/health", timeout=5)
        return {
            "status": "connected" if response.status_code == 200 else "disconnected",
            "response": response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/check-index")
def check_index():
    try:
        response = requests.get(f"{MARQO_URL}/indexes/{INDEX_NAME}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Index not found: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/list-indexes")
def list_indexes():
    try:
        response = requests.get(f"{MARQO_URL}/indexes", timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/search-test")
def search_test():
    try:
        # Test search with Marqo client
        results = marqo_client.index(INDEX_NAME).search("education assistance")
        return results
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)