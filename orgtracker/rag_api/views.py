from django.http import JsonResponse
from django.views import View
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

class QueryView(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            query = data.get('query')
            embedding = model.encode([query])[0]
            # Add your FAISS search logic here
            return JsonResponse({'answer': 'Your RAG response'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
