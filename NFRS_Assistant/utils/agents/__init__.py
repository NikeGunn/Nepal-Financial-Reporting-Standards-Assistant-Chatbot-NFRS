"""
Multi-agent system for the NFRS Assistant.

This package contains components for implementing a multi-agent
conversational system for financial reporting standards.
"""

# Make agent components available at the package level
from .multi_agent_chat import MultiAgentChat
from .notifications import ProgressNotifier, BackgroundNotifier
from .expert_selection import select_experts_for_query