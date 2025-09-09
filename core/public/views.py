

# Create your views here.
from django.shortcuts import render
from organizations.models import Program

def program_list(request):
    programs = Program.objects.all()
    return render(request, 'public/programs.html', {'programs': programs})