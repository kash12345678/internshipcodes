from django.urls import path
from .views import query_org

urlpatterns = [
    path('', query_org),
]
