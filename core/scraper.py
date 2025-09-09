from django.core.management.base import BaseCommand
from requests_html import HTMLSession
from organizations.models import Source, Organization
import json

class Command(BaseCommand):
    help = 'Scrape 211.org provider form data'

    def handle(self, *args, **options):
        session = HTMLSession()
        url = "https://www.211.org/service-provider-form"
        
        try:
            response = session.get(url)
            response.html.render(timeout=20)
            
            # ACTUAL SELECTORS (Find these in Chrome DevTools)
            form_section = response.html.find('#provider-form', first=True)
            
            if form_section:
                data = {
                    'form_action': form_section.find('form', first=True).attrs.get('action'),
                    'instructions': form_section.find('.form-instructions', first=True).text
                }
                
                # Save to database
                source, _ = Source.objects.get_or_create(
                    name="211.org Provider Form",
                    url=url
                )
                
                Organization.objects.update_or_create(
                    name="211 Service Provider Portal",
                    defaults={
                        'source': source,
                        'description': data['instructions'],
                        'website': data['form_action']
                    }
                )
                
                self.stdout.write(self.style.SUCCESS("Scraped provider form data"))
            else:
                self.stdout.write(self.style.WARNING("Form section not found"))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed: {str(e)}"))