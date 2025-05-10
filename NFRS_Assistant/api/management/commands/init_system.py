from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.conf import settings
import os
import logging
from api.users.models import UserProfile
from api.knowledge.models import VectorIndex

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Initialize the NFRS Assistant system with default settings and data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force reinitialization even if the system is already set up',
        )

    def handle(self, *args, **options):
        force = options.get('force', False)

        # Check if system is already initialized
        if User.objects.filter(is_superuser=True).exists() and not force:
            self.stdout.write(self.style.WARNING(
                'System appears to be already initialized. Use --force to reinitialize.'
            ))
            return

        self.stdout.write('Initializing NFRS Assistant system...')

        # Create default admin user if it doesn't exist
        self.create_default_admin()

        # Create directories if they don't exist
        self.create_directories()

        # Initialize vector index if needed
        self.initialize_vector_index()

        self.stdout.write(self.style.SUCCESS('System initialized successfully!'))

    def create_default_admin(self):
        """Create a default admin user."""
        if not User.objects.filter(username='admin').exists():
            self.stdout.write('Creating default admin user...')
            admin_user = User.objects.create_superuser(
                username='admin',
                email='admin@nfrs.org',
                password='admin123'  # This should be changed immediately
            )

            UserProfile.objects.create(
                user=admin_user,
                preferred_language='en',
                is_admin=True
            )

            self.stdout.write(self.style.SUCCESS(
                'Default admin created with username "admin" and password "admin123". '
                'Please change this password immediately!'
            ))
        else:
            self.stdout.write('Admin user already exists.')

    def create_directories(self):
        """Create necessary directories."""
        # Create media directory
        media_dir = settings.MEDIA_ROOT
        if not os.path.exists(media_dir):
            os.makedirs(media_dir)
            self.stdout.write(f'Created media directory: {media_dir}')

        # Create documents directory
        documents_dir = os.path.join(media_dir, 'documents')
        if not os.path.exists(documents_dir):
            os.makedirs(documents_dir)
            self.stdout.write(f'Created documents directory: {documents_dir}')

        # Create vector store directory
        vector_dir = settings.VECTOR_STORE_DIR
        if not os.path.exists(vector_dir):
            os.makedirs(vector_dir)
            self.stdout.write(f'Created vector store directory: {vector_dir}')

        # Create logs directory
        logs_dir = os.path.join(settings.BASE_DIR, 'logs')
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            self.stdout.write(f'Created logs directory: {logs_dir}')

    def initialize_vector_index(self):
        """Initialize vector index if needed."""
        if not VectorIndex.objects.filter(is_active=True).exists():
            self.stdout.write('Creating initial vector index record...')

            import uuid
            index_name = f"nfrs_index_{uuid.uuid4().hex[:8]}"
            index_path = os.path.join(settings.VECTOR_STORE_DIR, f"{index_name}.index")

            VectorIndex.objects.create(
                name=index_name,
                description="Initial NFRS documents vector index",
                index_file_path=index_path,
                is_active=True
            )

            self.stdout.write(self.style.SUCCESS('Vector index initialized successfully.'))
        else:
            self.stdout.write('Vector index already exists.')