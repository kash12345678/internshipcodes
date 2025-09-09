from django.contrib import admin
from .models import Organization, Source

@admin.register(Organization)
class OrganizationAdmin(admin.ModelAdmin):
    list_display = ('name', 'website', 'source', 'description')
    search_fields = ('name', 'website', 'description')

@admin.register(Source)
class SourceAdmin(admin.ModelAdmin):
    list_display = ('name', 'url')
