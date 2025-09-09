import json
from django.core.management.base import BaseCommand
from organizations.models import Source, Organization, Program, Benefit

class Command(BaseCommand):
    help = 'Load 211.org scraped data'

    def handle(self, *args, **options):
        with open('211_data.json') as f:
            data = json.load(f)

        source, _ = Source.objects.get_or_create(
            name="211.org",
            url="https://211.org"
        )

        for org_data in data['organizations']:
            org = Organization.objects.create(
                source=source,
                name=org_data['name'],
                description=org_data.get('description', ''),
                website=org_data.get('website', '')
            )
            for program_data in org_data['programs']:
                program = Program.objects.create(
                    organization=org,
                    name=program_data['name'],
                    eligibility=program_data.get('eligibility', '')
                )
                for benefit_data in program_data.get('benefits', []):
                    Benefit.objects.create(
                        program=program,
                        name=benefit_data['name'],
                        requirements=benefit_data.get('requirements', '')
                    )

        self.stdout.write(self.style.SUCCESS('Data loaded successfully!'))
