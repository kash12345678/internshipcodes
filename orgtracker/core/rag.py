from sentence_transformers import SentenceTransformer
from django.apps import apps
import numpy as np
import faiss
from transformers import pipeline

model = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

def build_vector_index():
    Organization = apps.get_model('core', 'Organization')
    orgs = Organization.objects.all()
    
    texts = []
    for org in orgs:
        text = f"{org.name}: {org.description}"
        texts.append(text)
    
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, texts

vector_index, org_texts = build_vector_index()

def retrieve_docs(question, k=5):
    query_embedding = model.encode([question])
    distances, indices = vector_index.search(query_embedding, k)
    results = []
    for i in indices[0]:
        if i < len(org_texts):
            name, desc = org_texts[i].split(":", 1)
            results.append({
                "title": name.strip(),
                "content": desc.strip()
            })
    return results

def generate_answer(question, retrieved_docs):
    context = "\n\n".join([f"{doc['title']}: {doc['content']}" for doc in retrieved_docs])
    input_text = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    summary = generator(input_text, max_length=100, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

def qa_chain(question):
    docs = retrieve_docs(question)
    answer = generate_answer(question, docs)
    return {"answer": answer, "sources": docs}
