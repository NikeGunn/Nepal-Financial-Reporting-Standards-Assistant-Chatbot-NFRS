from django.apps import AppConfig


class UsersConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api.users'
    verbose_name = 'Users'

    def ready(self):
        # Import signal handlers to register them
        import api.users.signals