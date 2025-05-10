from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from .models import Conversation, Message
import logging

logger = logging.getLogger(__name__)


@receiver(post_save, sender=Message)
def update_conversation_timestamp(sender, instance, created, **kwargs):
    """
    Signal to update the conversation's updated_at timestamp when a new message is added.
    """
    if created:
        try:
            # Update the conversation's timestamp
            conversation = instance.conversation
            conversation.save()  # This will update the updated_at field

            # If this is the first message and the conversation has no title, set it
            if conversation.title == '' and conversation.messages.count() == 1:
                # Use the first part of the message as the title
                preview = instance.content[:50]
                if len(instance.content) > 50:
                    preview += '...'

                conversation.title = preview
                conversation.save()

                logger.info(f"Updated conversation title: {conversation.id}")
        except Exception as e:
            logger.error(f"Error updating conversation timestamp: {e}")