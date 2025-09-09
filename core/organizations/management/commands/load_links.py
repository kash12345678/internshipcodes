from django.core.management.base import BaseCommand
from myapp.models import ResourceLink

class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        links = [
            ("Help Paying Bills", "https://www.211.org/get-help/i-need-help-paying-my-bills"),
            ("Caregiver Resources", "https://www.211.org/get-help/caregiver-resources"),
            ("Your Local 211", "https://www.211.org/about-us/your-local-211"),
            ("Disaster Recovery", "https://www.211.org/get-help/disaster-recovery"),
            ("Housing Expenses", "https://www.211.org/get-help/housing-expenses"),
            ("Utilities Expenses", "https://www.211.org/get-help/utilities-expenses"),
            ("Food Programs", "https://www.211.org/get-help/food-programs-food-benefits"),
            ("Healthcare Expenses", "https://www.211.org/get-help/healthcare-expenses"),
            ("Mental Health", "https://www.211.org/get-help/mental-health"),
            ("Substance Use", "https://www.211.org/get-help/substance-use"),
            ("Our Partners", "https://www.211.org/partner-us/our-partners"),
            ("Pet Help Finder", "https://www.211.org/partner-us/pet-help-finder"),
            ("Astar", "https://www.211.org/astar"),
            ("About Us", "https://www.211.org/about-us"),
            ("Meet Our People", "https://www.211.org/about-us/meet-our-people"),
            ("211 Data", "https://www.unitedway.org/211Data"),
            ("Your Local 211 (Again)", "https://www.211.org/about-us/your-local-211"),
            ("Blog", "https://www.211.org/about-us/blog"),
            ("211 Toolkit", "https://211toolkit.unitedway.org/"),
            ("United Way Support", "https://support.unitedway.org/page/211")
        ]
        ResourceLink.objects.all().delete()
        for title, url in links:
            ResourceLink.objects.create(title=title, url=url)
        self.stdout.write(self.style.SUCCESS("Links loaded successfully."))
