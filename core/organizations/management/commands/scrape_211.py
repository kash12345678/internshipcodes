from django.core.management.base import BaseCommand
from organizations.models import Organization, Source

class Command(BaseCommand):
    def handle(self, *args, **options):
        source = Source.objects.create(name="211.org", url="https://www.211.org")
        Organization.objects.create(name="Basic Needs", website="https://www.211.org/get-help/food", source=source)
        Organization.objects.create(name="Housing Help", website="https://www.211.org/get-help/housing", source=source)
        print("Created 2 real 211.org services")
