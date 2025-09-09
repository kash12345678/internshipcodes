from django.http import JsonResponse

SOURCES = [
    {"title": "Healthcare Services", "content": "Provides medical care, health screenings, and prescription assistance"},
    {"title": "Food Assistance", "content": "Offers food banks, meal programs, and nutrition education"},
]

def query_org(request):
    question = request.GET.get("question", "").lower()
    relevant_sources = [
        src for src in SOURCES if any(word in question for word in src["title"].lower().split())
    ]
    if relevant_sources:
        answer = " ".join([f'{src["title"]}: {src["content"]}' for src in relevant_sources])
    else:
        answer = "Sorry, I don’t have information on that."
    return JsonResponse({
        "question": question,
        "answer": answer
    })

