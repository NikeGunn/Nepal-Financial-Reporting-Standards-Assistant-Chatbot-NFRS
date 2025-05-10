from django.apps import AppConfig


class KnowledgeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api.knowledge'
    verbose_name = 'Knowledge Base'

    def ready(self):
        # Import signal handlers to register them
        import api.knowledge.signals