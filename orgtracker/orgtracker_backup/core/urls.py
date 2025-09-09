from django.contrib import admin
from django.urls import path
from core import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.link_list, name='link_list'),
]
from django.urls import include; urlpatterns += [path('api/', include('rag_api.urls'))]
