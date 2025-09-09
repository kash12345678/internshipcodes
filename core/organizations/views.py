from django.shortcuts import render
from myapp.models import ResourceLink

def link_list(request):
    links = ResourceLink.objects.all()
    return render(request, "links.html", {"links": links})
