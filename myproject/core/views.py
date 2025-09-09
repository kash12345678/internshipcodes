from django.http import JsonResponse
import requests
import json

def rag_view(request):
    question = request.GET.get("question", "")
    
    try:
        response = requests.post(
            "http://localhost:8882/indexes/rag_index/search",
            headers={"Content-Type": "application/json"},
            json={"q": question, "limit": 1},
            timeout=5
        )
        
        if response.status_code == 200:
            results = response.json()
            if results['hits']:
                best_match = results['hits'][0]
                return JsonResponse({
                    "question": question,
                    "answer": best_match.get('Content', best_match.get('content', 'No content found')),
                    "references": [],
                    "suggestions": []
                })
    except Exception as e:
        pass
    
    return JsonResponse({
        "question": question,
        "answer": "Marqo search unavailable. Please try again later.",
        "references": [],
        "suggestions": []
    })

def init_marqo(request):
    try:
        response = requests.post(
            "http://localhost:8882/indexes/rag_index/documents",
            headers={"Content-Type": "application/json"},
            json={
                "documents": [
                    {
                        "_id": "food_assistance_1",
                        "Title": "Food Assistance",
                        "Content": "Food assistance programs help individuals and families access nutritious meals in times of need through food banks, government subsidies like SNAP, school meal programs, and community kitchens."
                    },
                    {
                        "_id": "healthcare_services_1",
                        "Title": "Healthcare Services", 
                        "Content": "Healthcare services cover treatment, preventive care, prescription assistance, and mental health counseling through hospitals, clinics, NGOs, and government initiatives."
                    }
                ],
                "tensor_fields": ["Content"]
            },
            timeout=10
        )
        return JsonResponse({"status": "Marqo index initialized successfully", "response": response.json()})
    except Exception as e:
        return JsonResponse({"status": f"Error: {str(e)}"})
