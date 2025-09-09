from django.core.management.base import BaseCommand
from organizations.models import Organization, Source

class Command(BaseCommand):
    def handle(self, *args, **options):
        Organization.objects.all().delete()
        Source.objects.all().delete()
        
        source = Source.objects.create(
            name="211.org Verified", 
            url="https://www.211.org"
        )

        services = [
            ("Basic Needs", "https://www.211.org/get-help/i-need-help-paying-my-bills"),
            ("Housing Help", "https://www.211.org/get-help/housing-expenses"),
            ("Food Programs", "https://www.211.org/get-help/food-programs-food-benefits"),
            ("Healthcare", "https://www.211.org/get-help/healthcare-expenses"),
            ("Mental Health", "https://www.211.org/get-help/mental-health")
        ]

        for name, url in services:
            Organization.objects.create(
                name=name,
                website=url,  # Changed to match model
                source=source,
                description=f"Official 211.org service: {name}"
            )
            print(f"Added: {name}")
