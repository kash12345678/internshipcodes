from django.core.management.base import BaseCommand
from organizations.models import Organization, Source

class Command(BaseCommand):
    def handle(self, *args, **options):
        Organization.objects.all().delete()
        Source.objects.all().delete()
        
        source = Source.objects.create(
            name="211.org Complete Resources", 
            url="https://www.211.org"
        )

        services = [
            ("Bill Payment Help", "https://www.211.org/get-help/i-need-help-paying-my-bills", "Resources to assist with paying utility, rent, and other bills."),
            ("Caregiver Resources", "https://www.211.org/get-help/caregiver-resources", "Support and guidance for caregivers of children, elderly, or disabled individuals."),
            ("Local 211 Services", "https://www.211.org/about-us/your-local-211", "Find local 211 organizations that provide direct support in your community."),
            ("Disaster Recovery", "https://www.211.org/get-help/disaster-recovery", "Assistance for people recovering from natural disasters and emergencies."),
            ("Housing Expenses", "https://www.211.org/get-help/housing-expenses", "Help with rent, mortgages, and shelter resources."),
            ("Utilities Help", "https://www.211.org/get-help/utilities-expenses", "Programs for assistance with electricity, gas, water, and other utility costs."),
            ("Food Programs", "https://www.211.org/get-help/food-programs-food-benefits", "Food pantries, meal services, and government food benefit programs."),
            ("Healthcare Expenses", "https://www.211.org/get-help/healthcare-expenses", "Help covering healthcare costs including prescriptions and medical visits."),
            ("Mental Health", "https://www.211.org/get-help/mental-health", "Access to mental health professionals, therapy, and crisis hotlines."),
            ("Substance Use Help", "https://www.211.org/get-help/substance-use", "Resources for addiction recovery, detox, and counseling."),
            ("Our Partners", "https://www.211.org/partner-us/our-partners", "Information about organizations and businesses that partner with 211."),
            ("Pet Help Finder", "https://www.211.org/partner-us/pet-help-finder", "Resources for pet care, veterinary help, and pet-related services."),
            ("Astar", "https://www.211.org/astar", "Astar program details and related services from 211."),
            ("About Us", "https://www.211.org/about-us", "Learn more about 211.org and its mission."),
            ("Meet Our People", "https://www.211.org/about-us/meet-our-people", "Information about the leadership and staff behind 211."),
            ("211 Data", "https://www.unitedway.org/211Data", "Data and insights collected by 211 nationwide."),
            ("Local 211", "https://www.211.org/about-us/your-local-211", "Details on local 211 branches serving specific regions."),
            ("Blog", "https://www.211.org/about-us/blog", "Articles and stories about community services and support."),
            ("211 Toolkit", "https://211toolkit.unitedway.org/", "Toolkit and resources for communities using 211."),
            ("Support United Way", "https://support.unitedway.org/page/211", "Opportunities to support United Way and 211 through donations.")
        ]

        for name, url, desc in services:
            Organization.objects.create(
                name=name,
                website=url,
                source=source,
                description=desc
            )
            print(f"Added: {name}")

        self.stdout.write(self.style.SUCCESS("Successfully loaded all 211.org services with descriptions"))
