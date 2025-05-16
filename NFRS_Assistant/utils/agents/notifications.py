"""
Notification system for the multi-agent chat interface.

This module provides classes for sending WebSocket notifications
about the multi-agent processing status to the frontend.
"""

import threading
import time
import logging
import json
from typing import List, Dict, Any, Optional
from django.conf import settings

logger = logging.getLogger(__name__)

# Try to import Django Channels/ASGI for WebSocket support
CHANNELS_AVAILABLE = False
try:
    from channels.layers import get_channel_layer
    from asgiref.sync import async_to_sync
    CHANNELS_AVAILABLE = True
    logger.info("Django Channels available for WebSocket notifications")
except ImportError:
    logger.warning("Django Channels not available. WebSocket notifications will be disabled.")


class ProgressNotifier:
    """
    Utility class for sending progress notifications to the client.

    Uses Django Channels to send WebSocket messages if available,
    otherwise silently fails without affecting the main process.
    """

    def __init__(self, conversation_id: str, user_id: str):
        """
        Initialize the notifier with conversation and user identifiers.

        Args:
            conversation_id: The ID of the current conversation
            user_id: The ID of the user to send notifications to
        """
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.channel_layer = None

        # Initialize channel layer if Channels is available
        if CHANNELS_AVAILABLE:
            try:
                self.channel_layer = get_channel_layer()
                logger.info(f"Channel layer initialized for user {user_id}")
            except Exception as e:
                logger.error(f"Error initializing channel layer: {e}")

    def send_thinking_start(self):
        """Send notification that the system has started processing."""
        self._send_notification({
            "type": "thinking_start",
            "message": "Our panel of experts is analyzing your question..."
        })

    def send_thinking_complete(self):
        """Send notification that the system has completed processing."""
        self._send_notification({
            "type": "thinking_complete",
            "message": "Processing complete"
        })

    def send_expert_selection(self, experts: List[Dict[str, str]]):
        """
        Send notification about which experts were selected for the question.

        Args:
            experts: List of expert information dictionaries
        """
        expert_names = [f"{expert['name']} ({expert.get('title', '')})"
                       for expert in experts]

        self._send_notification({
            "type": "expert_selected",
            "message": f"Consulting with financial experts: {', '.join(expert_names)}",
            "experts": experts
        })

    def send_progress_update(self, message: str, progress: float = None):
        """
        Send a generic progress update message.

        Args:
            message: Progress message to display
            progress: Optional progress value (0-1)
        """
        payload = {
            "type": "progress_update",
            "message": message
        }

        if progress is not None:
            payload["progress"] = min(max(0, progress), 1)

        self._send_notification(payload)

    def _send_notification(self, payload: Dict[str, Any]):
        """
        Send a notification via WebSocket.

        Args:
            payload: Notification data to send
        """
        if not CHANNELS_AVAILABLE or not self.channel_layer:
            # Silently return if WebSockets are not available
            return

        try:
            # Add standard fields to payload
            full_payload = {
                **payload,
                "conversation_id": self.conversation_id
            }

            # Send to user's specific channel group
            group_name = f"user_{self.user_id}"
            async_to_sync(self.channel_layer.group_send)(
                group_name,
                {
                    "type": "chat_notification",
                    "message": full_payload
                }
            )
            logger.debug(f"Notification sent to {group_name}: {payload['type']}")

        except Exception as e:
            logger.error(f"Error sending notification: {e}")


class BackgroundNotifier(threading.Thread):
    """
    Background thread that sends periodic thinking notifications.

    This is used to keep the client informed that processing is still happening
    for long-running operations.
    """

    def __init__(
        self,
        notifier: ProgressNotifier,
        interval: float = 3.0,
        max_time: float = 60.0,
        messages: List[str] = None
    ):
        """
        Initialize the background notifier.

        Args:
            notifier: ProgressNotifier to use for sending notifications
            interval: Seconds between notifications (default: 3.0)
            max_time: Maximum seconds to run (default: 60.0)
            messages: Custom thinking messages to cycle through
        """
        super().__init__()
        self.daemon = True  # Thread will exit when main thread exits
        self.notifier = notifier
        self.interval = interval
        self.max_time = max_time
        self.running = threading.Event()

        # Default thinking messages if none provided
        self.messages = messages or [
            "Our financial experts are analyzing your question...",
            "Consulting relevant NFRS and IFRS standards...",
            "Examining financial reporting requirements...",
            "Evaluating accounting principles that apply to your question...",
            "Reviewing recent updates to standards that may be relevant..."
        ]

    def run(self):
        """Run the background notification thread."""
        start_time = time.time()
        index = 0
        self.running.set()

        try:
            while self.running.is_set():
                # Check if we've exceeded max time
                if time.time() - start_time > self.max_time:
                    break

                # Send current thinking message
                message = self.messages[index % len(self.messages)]
                elapsed = time.time() - start_time
                progress = min(elapsed / self.max_time, 0.95)  # Cap at 95%

                self.notifier.send_progress_update(message, progress)

                # Wait for interval or until stopped
                self.running.wait(timeout=self.interval)

                # Move to next message
                index += 1

        except Exception as e:
            logger.error(f"Error in background notifier: {e}")

    def stop(self):
        """Stop the background notification thread."""
        self.running.clear()