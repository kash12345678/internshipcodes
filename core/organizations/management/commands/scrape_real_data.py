from django.core.management.base import BaseCommand
from organizations.models import Organization, Source

class Command(BaseCommand):
    def handle(self, *args, **options):
        source = Source.objects.create(name="211.org", url="https://www.211.org")
        services = [
            ("Basic Needs", "https://www.211.org/services/basic-needs"),
            ("Housing Help", "https://www.211.org/services/housing"),
            ("Mental Health", "https://www.211.org/services/mental-health"),
            ("Employment", "https://www.211.org/services/jobs"),
            ("Healthcare", "https://www.211.org/services/health")
        ]
        for name, url in services:
            Organization.objects.create(name=name, website=url, source=source)
        print("Added 5 services")
