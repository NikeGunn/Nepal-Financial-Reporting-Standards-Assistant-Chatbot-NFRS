"""
Multi-agent chat system for the NFRS Assistant.

This module provides a robust multi-agent system with multiple fallback mechanisms
to ensure reliability across different LangChain versions or even without LangChain.
"""

import logging
import json
import random
import openai
import time
from typing import Dict, List, Any, Optional, Union
from django.conf import settings

logger = logging.getLogger(__name__)

# Try to import LangChain with version compatibility checks
LANGCHAIN_AVAILABLE = False
LANGCHAIN_VERSION = None
try:
    import importlib.metadata
    try:
        # Get LangChain version
        LANGCHAIN_VERSION = importlib.metadata.version("langchain")
        if LANGCHAIN_VERSION:
            logger.info(f"LangChain version {LANGCHAIN_VERSION} detected")
            LANGCHAIN_AVAILABLE = True
    except importlib.metadata.PackageNotFoundError:
        logger.warning("LangChain not found")
except ImportError:
    logger.warning("Could not check LangChain version")

# Import expert selection logic
try:
    from .expert_selection import select_experts_for_query
except ImportError:
    # Fallback implementation if expert_selection module is not available
    logger.warning("Could not import expert_selection module, using fallback")

    def select_experts_for_query(query, num_experts=3):
        """
        Fallback implementation for expert selection if the module is not available.

        Returns a default set of financial experts.
        """
        default_experts = [
            {
                "name": "Dr. Ramesh Sharma",
                "title": "Senior NFRS Specialist",
                "description": "Expert in Nepal Financial Reporting Standards with 15 years of experience."
            },
            {
                "name": "Maya Poudel",
                "title": "IFRS Compliance Officer",
                "description": "Specializes in international financial standards and compliance."
            },
            {
                "name": "Anand Joshi",
                "title": "Financial Reporting Analyst",
                "description": "Expert in analyzing financial statements and reporting requirements."
            }
        ]
        # Return a subset of the default experts
        return default_experts[:min(num_experts, len(default_experts))]


class MultiAgentChat:
    """
    Multi-agent chat system that integrates financial experts into a conversational flow.

    This class provides a robust approach with multiple fallback mechanisms:
    1. Uses LangChain when available (with version-specific adaptations)
    2. Falls back to direct OpenAI API when LangChain isn't available
    3. Reduces expert count if full panel isn't working
    4. Provides reasonable defaults in worst-case scenarios
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.3,
        max_tokens: int = 1500,
        max_experts: int = 3
    ):
        """
        Initialize the multi-agent chat system.

        Args:
            model_name: The model to use (default: gpt-3.5-turbo)
            temperature: Temperature for generation (default: 0.3)
            max_tokens: Maximum tokens for generated responses (default: 1500)
            max_experts: Maximum number of expert agents to use (default: 3)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_experts = max_experts
        self.api_key = settings.OPENAI_API_KEY

        # Validate API key
        if not self.api_key:
            logger.error("OpenAI API key not found in settings")
            raise ValueError("OpenAI API key is required for the multi-agent system")

        # Check if we can use LangChain
        self.use_langchain = LANGCHAIN_AVAILABLE

        # Try to initialize OpenAI client directly as fallback
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info("Initialized OpenAI client for direct API access")
        except ImportError:
            logger.error("Could not initialize OpenAI client")
            self.client = None

    def process_query(
        self,
        query: str,
        context: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query using the multi-agent system.

        Args:
            query: The user's question
            context: Retrieved contextual information
            conversation_history: List of previous messages

        Returns:
            Dict with the response message and metadata
        """
        logger.info(f"Processing query with multi-agent system: {query[:50]}...")

        if not conversation_history:
            conversation_history = []

        try:
            # 1. Select relevant experts based on the query
            experts = select_experts_for_query(query, num_experts=self.max_experts)

            if not experts:
                logger.warning("No experts selected, using fallback")
                # Fallback to default experts
                experts = self._get_default_experts(self.max_experts)

            logger.info(f"Selected {len(experts)} experts for the query")

            # 2. Try to use LangChain if available with version-specific handling
            if self.use_langchain:
                try:
                    response = self._langchain_multi_agent(query, context, conversation_history, experts)
                    response['expert_used'] = experts
                    return response
                except Exception as e:
                    logger.error(f"LangChain approach failed: {e}")
                    # Continue to fallback methods

            # 3. Fallback to direct OpenAI API call
            try:
                response = self._direct_api_multi_agent(query, context, conversation_history, experts)
                response['expert_used'] = experts
                return response
            except Exception as e:
                logger.error(f"Direct API multi-agent approach failed: {e}")

                # 4. Reduce number of experts and try again
                if len(experts) > 1:
                    logger.info("Attempting with reduced expert panel")
                    reduced_experts = experts[:1]  # Use just one expert
                    try:
                        response = self._direct_api_multi_agent(query, context, conversation_history, reduced_experts)
                        response['expert_used'] = reduced_experts
                        return response
                    except Exception as e:
                        logger.error(f"Reduced expert panel approach failed: {e}")

            # 5. Ultimate fallback: simple response with no experts
            logger.warning("All multi-agent approaches failed, using simple response")
            fallback_response = {
                "message": self._generate_fallback_response(query, context),
                "expert_used": []
            }
            return fallback_response

        except Exception as e:
            logger.error(f"Error in multi-agent processing: {e}")
            return {
                "message": f"I apologize, but I encountered an issue processing your question about {query[:30]}... Please try again.",
                "expert_used": []
            }

    def _langchain_multi_agent(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]],
        experts: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Use LangChain for the multi-agent approach with version compatibility.

        Args:
            query: User query
            context: Retrieved context
            conversation_history: Prior conversation
            experts: List of expert data

        Returns:
            Dict with response message
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not available")

        try:
            # Handle different LangChain versions
            if LANGCHAIN_VERSION and LANGCHAIN_VERSION.startswith("0."):
                # For older LangChain versions (0.x)
                return self._langchain_legacy_approach(query, context, conversation_history, experts)
            else:
                # For newer LangChain versions
                return self._langchain_modern_approach(query, context, conversation_history, experts)
        except Exception as e:
            logger.error(f"Error in LangChain multi-agent: {e}")
            raise

    def _langchain_legacy_approach(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]],
        experts: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Implementation for older LangChain versions."""
        try:
            # Import LangChain components for older versions
            from langchain.chat_models import ChatOpenAI
            from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
            from langchain.schema import HumanMessage, SystemMessage, AIMessage

            # Create the LLM
            llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=self.api_key
            )

            # Convert conversation history to LangChain message format
            messages = []
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
                elif msg["role"] == "system":
                    messages.append(SystemMessage(content=msg["content"]))

            # Create system message with expert panel format
            expert_descriptions = "\n".join([
                f"- {expert['name']} ({expert['title']}): {expert.get('description', '')}"
                for expert in experts
            ])

            system_template = """You are a panel of financial experts specializing in Nepal Financial Reporting Standards (NFRS) and International Financial Reporting Standards (IFRS).

The panel consists of:
{expert_descriptions}

As a panel, consult with each other and provide a thorough, authoritative answer based on the context provided. If the context doesn't contain the relevant information, acknowledge this explicitly.

FORMATTING INSTRUCTIONS:
- Always use Markdown formatting to structure your responses
- Use headers (##, ###) to organize information logically
- Use **bold** for important terms, definitions, or key points
- Use bullet points or numbered lists for series of related items
- When presenting financial data like balance sheets, income statements, or audit reports, use Markdown tables
- For equations or calculations, use proper formatting with operators and indent calculations
- For Nepal-specific content like NRB regulations, tax forms, or NFRS guidelines, create clearly formatted sections with headers
- For examples or sample reports, use code blocks with the appropriate language specification
- If referring to legislation or standards, properly format the references with sections and clauses

FINANCIAL CONTENT FORMATTING:
- Balance Sheets: Always use Markdown tables with clear headers for Assets/Liabilities
- Income Statements: Use tables with proper indentation for revenue and expense categories
- Audit Reports: Format with proper sections (Opinion, Basis for Opinion, Key Audit Matters)
- Tax Calculations: Show step-by-step calculations with clear explanations
- Financial Ratios: Include the formula and calculation steps
- NFRS/IFRS Standards: Include the standard number and title in bold

Use a clear, professional tone suitable for financial professionals. Cite specific NFRS/IFRS standards when applicable.

Context information:
{context}"""

            system_message = SystemMessage(content=system_template.format(
                expert_descriptions=expert_descriptions,
                context=context or "No specific context information available."
            ))

            # Add system message at the beginning
            all_messages = [system_message] + messages

            # Add the user's current query
            all_messages.append(HumanMessage(content=query))

            # Get the panel's response
            response = llm(all_messages)

            return {
                "message": response.content
            }

        except Exception as e:
            logger.error(f"Error in legacy LangChain approach: {e}")
            raise

    def _langchain_modern_approach(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]],
        experts: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Implementation for newer LangChain versions."""
        try:
            # Import newer LangChain components
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

            # Create the LLM
            llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=self.api_key
            )

            # Convert conversation history to LangChain message format
            messages = []
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
                elif msg["role"] == "system":
                    messages.append(SystemMessage(content=msg["content"]))

            # Create system message with expert panel format
            expert_descriptions = "\n".join([
                f"- {expert['name']} ({expert['title']}): {expert.get('description', '')}"
                for expert in experts
            ])

            system_template = """You are a panel of financial experts specializing in Nepal Financial Reporting Standards (NFRS) and International Financial Reporting Standards (IFRS).

The panel consists of:
{expert_descriptions}

As a panel, consult with each other and provide a thorough, authoritative answer based on the context provided. If the context doesn't contain the relevant information, acknowledge this explicitly.

FORMATTING INSTRUCTIONS:
- Always use Markdown formatting to structure your responses
- Use headers (##, ###) to organize information logically
- Use **bold** for important terms, definitions, or key points
- Use bullet points or numbered lists for series of related items
- When presenting financial data like balance sheets, income statements, or audit reports, use Markdown tables
- For equations or calculations, use proper formatting with operators and indent calculations
- For Nepal-specific content like NRB regulations, tax forms, or NFRS guidelines, create clearly formatted sections with headers
- For examples or sample reports, use code blocks with the appropriate language specification
- If referring to legislation or standards, properly format the references with sections and clauses

FINANCIAL CONTENT FORMATTING:
- Balance Sheets: Always use Markdown tables with clear headers for Assets/Liabilities
- Income Statements: Use tables with proper indentation for revenue and expense categories
- Audit Reports: Format with proper sections (Opinion, Basis for Opinion, Key Audit Matters)
- Tax Calculations: Show step-by-step calculations with clear explanations
- Financial Ratios: Include the formula and calculation steps
- NFRS/IFRS Standards: Include the standard number and title in bold

Use a clear, professional tone suitable for financial professionals. Cite specific NFRS/IFRS standards when applicable.

Context information:
{context}"""

            system_message = SystemMessage(content=system_template.format(
                expert_descriptions=expert_descriptions,
                context=context or "No specific context information available."
            ))

            # Add system message at the beginning
            all_messages = [system_message] + messages

            # Add the user's current query
            all_messages.append(HumanMessage(content=query))

            # Get the panel's response
            response = llm.invoke(all_messages)

            return {
                "message": response.content
            }

        except Exception as e:
            logger.error(f"Error in modern LangChain approach: {e}")
            raise

    def _direct_api_multi_agent(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]],
        experts: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Use direct OpenAI API for multi-agent approach.

        Args:
            query: User query
            context: Retrieved context
            conversation_history: Prior conversation
            experts: List of expert data

        Returns:
            Dict with response message
        """
        if not self.client:
            raise ValueError("OpenAI client is not available")

        try:
            # Create expert panel description
            expert_descriptions = "\n".join([
                f"- {expert['name']} ({expert['title']}): {expert.get('description', '')}"
                for expert in experts
            ])

            # Create system prompt
            system_prompt = f"""You are a panel of financial experts specializing in Nepal Financial Reporting Standards (NFRS) and International Financial Reporting Standards (IFRS).

The panel consists of:
{expert_descriptions}

As a panel, consult with each other and provide a thorough, authoritative answer based on the context provided. If the context doesn't contain the relevant information, acknowledge this explicitly.

FORMATTING INSTRUCTIONS:
- Always use Markdown formatting to structure your responses
- Use headers (##, ###) to organize information logically
- Use **bold** for important terms, definitions, or key points
- Use bullet points or numbered lists for series of related items
- When presenting financial data like balance sheets, income statements, or audit reports, use Markdown tables
- For equations or calculations, use proper formatting with operators and indent calculations
- For Nepal-specific content like NRB regulations, tax forms, or NFRS guidelines, create clearly formatted sections with headers
- For examples or sample reports, use code blocks with the appropriate language specification
- If referring to legislation or standards, properly format the references with sections and clauses

FINANCIAL CONTENT FORMATTING:
- Balance Sheets: Always use Markdown tables with clear headers for Assets/Liabilities
- Income Statements: Use tables with proper indentation for revenue and expense categories
- Audit Reports: Format with proper sections (Opinion, Basis for Opinion, Key Audit Matters)
- Tax Calculations: Show step-by-step calculations with clear explanations
- Financial Ratios: Include the formula and calculation steps
- NFRS/IFRS Standards: Include the standard number and title in bold

Use a clear, professional tone suitable for financial professionals. Cite specific NFRS/IFRS standards when applicable.

Context information:
{context or "No specific context information available."}"""

            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt}
            ]

            # Add conversation history
            for msg in conversation_history:
                if msg["role"] in ["user", "assistant", "system"]:
                    messages.append(msg)

            # Add the current query
            messages.append({"role": "user", "content": query})

            # Make the API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Extract the response
            response_text = response.choices[0].message.content.strip()

            return {
                "message": response_text
            }

        except Exception as e:
            logger.error(f"Error in direct API multi-agent: {e}")
            raise

    def _generate_fallback_response(self, query: str, context: str) -> str:
        """
        Generate a simple fallback response when all other methods fail.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Simple response string
        """
        try:
            # Try one last direct API call with simplified prompt
            if self.client:
                system_prompt = """You are a helpful financial assistant specializing in Nepal Financial Reporting Standards (NFRS).

Provide a professional response to the user's question based on any available context.

FORMATTING INSTRUCTIONS:
- Always use Markdown formatting to structure your responses
- Use headers (##, ###) to organize information logically
- Use **bold** for important terms, definitions, or key points
- Use bullet points or numbered lists for series of related items
- When presenting financial data like balance sheets, income statements, or audit reports, use Markdown tables
- Format Nepal-specific content clearly with proper headings and sections

Use a clear, professional tone suitable for financial professionals."""

                messages = [
                    {"role": "system", "content": system_prompt}
                ]

                # Add context if available
                if context:
                    messages.append({
                        "role": "system",
                        "content": f"Context information: {context}"
                    })

                # Add the query
                messages.append({"role": "user", "content": query})

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1000
                )

                return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error in fallback response: {e}")

        # Ultimate fallback if even the above fails
        return (
            f"## Unable to Process Request\n\n"
            f"I apologize, but I'm having difficulty processing your question about **{query[:30]}**... \n\n"
            "Our expert panel is currently unavailable. \n\n"
            "Please try again later or rephrase your question."
        )

    def _get_default_experts(self, num_experts: int = 3) -> List[Dict[str, str]]:
        """
        Get a default list of financial experts.

        Args:
            num_experts: Number of experts to return

        Returns:
            List of expert data dictionaries
        """
        default_experts = [
            {
                "name": "Dr. Ramesh Sharma",
                "title": "Senior NFRS Specialist",
                "description": "Expert in Nepal Financial Reporting Standards with 15 years of experience."
            },
            {
                "name": "Maya Poudel",
                "title": "IFRS Compliance Officer",
                "description": "Specializes in international financial standards and compliance."
            },
            {
                "name": "Anand Joshi",
                "title": "Financial Reporting Analyst",
                "description": "Expert in analyzing financial statements and reporting requirements."
            },
            {
                "name": "Dr. Sunita Thapa",
                "title": "Accounting Standards Expert",
                "description": "Specialist in aligning NFRS with international accounting standards."
            },
            {
                "name": "Rajesh Kumar Singh",
                "title": "Corporate Financial Advisor",
                "description": "Advisor to corporations on implementing NFRS in their reporting."
            }
        ]

        # Return a subset of the default experts
        return default_experts[:min(num_experts, len(default_experts))]