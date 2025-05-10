from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from .models import UserProfile
import logging

logger = logging.getLogger(__name__)


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """
    Signal to automatically create a UserProfile when a User is created.
    """
    if created:
        try:
            # Create user profile if it doesn't exist
            UserProfile.objects.get_or_create(
                user=instance,
                defaults={
                    'preferred_language': 'en',
                    'is_admin': False
                }
            )
            logger.info(f"Created profile for user: {instance.username}")
        except Exception as e:
            logger.error(f"Error creating profile for user {instance.username}: {e}")