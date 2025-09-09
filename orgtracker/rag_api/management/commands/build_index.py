from django.core.management.base import BaseCommand
from rag_api.utils import create_index
from organizations.models import Organization

class Command(BaseCommand):
    help = 'Build FAISS index from organization data'

    def handle(self, *args, **options):
        orgs = Organization.objects.all()
        texts = [f"{org.name}\n{org.description}\n{org.services}" for org in orgs]
        index = create_index(texts)
        faiss.write_index(index, 'org_index.faiss')
        self.stdout.write(self.style.SUCCESS('Index built successfully'))
