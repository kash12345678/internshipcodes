from django.urls import path
from . import views

urlpatterns = [
    path('', views.program_list, name='home'),
]