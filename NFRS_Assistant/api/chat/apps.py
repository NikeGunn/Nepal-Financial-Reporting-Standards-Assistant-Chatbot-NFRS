from django.apps import AppConfig


class ChatConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api.chat'
    verbose_name = 'Chat'

    def ready(self):
        # Import signal handlers to register them
        import api.chat.signals