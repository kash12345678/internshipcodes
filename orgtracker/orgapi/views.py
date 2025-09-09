from django.http import JsonResponse

SOURCES = [
    {"title": "Healthcare Services", "content": "Provides medical care, health screenings, prescription assistance, mental health counseling, and wellness programs", "reference": "https://www.healthcare.gov"},
    {"title": "Food Assistance", "content": "Offers food banks, meal programs, nutrition education, and emergency grocery support", "reference": "https://www.feedingamerica.org"},
    {"title": "Housing Assistance", "content": "Helps with emergency shelters, rent assistance, eviction prevention, and transitional housing", "reference": "https://www.hud.gov"},
    {"title": "Employment Services", "content": "Provides job training, resume workshops, career counseling, and unemployment benefits support", "reference": "https://www.careeronestop.org"},
    {"title": "Mental Health Services", "content": "Includes counseling, crisis hotlines, addiction recovery, and peer support groups", "reference": "https://www.mentalhealth.gov"},
    {"title": "Utility Assistance", "content": "Offers help with electricity, water, gas bills, and energy efficiency programs", "reference": "https://www.liheapch.acf.hhs.gov"},
    {"title": "Education and Childcare", "content": "Covers after-school programs, tutoring, childcare subsidies, and literacy support", "reference": "https://www.ed.gov"},
    {"title": "Transportation Services", "content": "Includes medical appointment rides, public transit vouchers, and senior transportation services", "reference": "https://www.transit.dot.gov"}
]

RELATED_QUESTIONS = {
    "healthcare": ["How can I find free health clinics?", "Where to get prescription assistance?"],
    "food": ["Where can I find food banks?", "How to apply for food stamps?"],
    "housing": ["How do I get emergency shelter?", "What programs prevent eviction?"],
    "employment": ["Where to get job training?", "How do I apply for unemployment benefits?"],
    "mental": ["Where can I find addiction recovery support?", "How to access crisis hotlines?"],
    "utility": ["Who helps with electricity bills?", "How to get water bill assistance?"],
    "education": ["Are there free tutoring programs?", "How to apply for childcare subsidies?"],
    "transportation": ["How do I get medical ride assistance?", "Where can I get senior transport services?"]
}

def query_org(request):
    question = request.GET.get("question", "").lower()
    relevant_sources = [
        src for src in SOURCES if any(word in question for word in src["title"].lower().split())
    ]
    if relevant_sources:
        answer = " ".join([f'{src["title"]}: {src["content"]}' for src in relevant_sources])
        references = [src["reference"] for src in relevant_sources]
        suggestions = []
        for key, qs in RELATED_QUESTIONS.items():
            if key in question:
                suggestions.extend(qs)
    else:
        answer = "Sorry, I don’t have information on that."
        references = []
        suggestions = []
    return JsonResponse({
        "question": question,
        "answer": answer,
        "references": references,
        "suggestions": suggestions
    })
