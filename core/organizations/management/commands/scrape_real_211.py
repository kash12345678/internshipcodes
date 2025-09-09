from django.core.management.base import BaseCommand
from organizations.models import Organization, Source
import requests
from bs4 import BeautifulSoup

class Command(BaseCommand):
    def handle(self, *args, **options):
        # Clear old data
        Organization.objects.all().delete()
        
        # Create source
        source = Source.objects.create(name="211.org", url="https://www.211.org")
        
        # Scrape actual 211.org services
        url = "https://www.211.org"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get all main navigation links
        for link in soup.select('nav a[href^="https://www.211.org"]'):
            Organization.objects.create(
                name=link.text.strip(),
                website=link['href'],
                source=source,
                description=f"211.org Service: {link.text.strip()}"
            )
            print(f"Added: {link.text.strip()}")
