import logging
from django.conf import settings
from django.utils.translation import gettext as _
from django.shortcuts import get_object_or_404
from django.contrib.auth.models import User
from django.db import transaction
from django.http import Http404
from rest_framework import viewsets, status, generics
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import Conversation, Message
from .serializers import (
    ConversationListSerializer,
    ConversationDetailSerializer,
    MessageSerializer,
    ChatMessageSerializer,
    TranslateMessageSerializer
)
from api.knowledge.models import Document, SessionDocument, SessionDocumentChunk
from api.knowledge.serializers import SessionDocumentSerializer
from utils.vector_ops import search_documents
from utils.document_processor import vector_search_session_documents, cleanup_session_documents
from utils.translation import translate_text

# Import multi-agent system with proper error handling
try:
    from utils.agents.multi_agent_chat import MultiAgentChat
    from utils.agents.notifications import ProgressNotifier, BackgroundNotifier
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    logging.warning("Multi-agent system not available")
    MULTI_AGENT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Add ConversationListCreateView for compatibility
class ConversationListCreateView(generics.ListCreateAPIView):
    """
    List and create conversations for backward compatibility.
    This class exists to satisfy possible references to it in URL configurations.
    """
    permission_classes = [IsAuthenticated]
    serializer_class = ConversationListSerializer

    def get_queryset(self):
        """Return only conversations belonging to the authenticated user."""
        return Conversation.objects.filter(user=self.request.user).order_by('-updated_at')

    def perform_create(self, serializer):
        """Set the authenticated user as the owner of the conversation."""
        serializer.save(user=self.request.user)

class ConversationViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing conversations.
    """
    permission_classes = [IsAuthenticated]

    def get_serializer_class(self):
        if self.action == 'list':
            return ConversationListSerializer
        return ConversationDetailSerializer

    def get_queryset(self):
        """Return only conversations belonging to the authenticated user."""
        return Conversation.objects.filter(user=self.request.user).order_by('-updated_at')

    def perform_create(self, serializer):
        """Set the authenticated user as the owner of the conversation."""
        serializer.save(user=self.request.user)

    @action(detail=True, methods=['post'])
    def archive(self, request, pk=None):
        """Archive a conversation."""
        conversation = self.get_object()
        conversation.is_active = False
        conversation.save()
        return Response({'status': 'archived'})

    @action(detail=True, methods=['post'])
    def restore(self, request, pk=None):
        """Restore a conversation from archive."""
        conversation = self.get_object()
        conversation.is_active = True
        conversation.save()
        return Response({'status': 'restored'})

    def destroy(self, request, *args, **kwargs):
        """Delete a conversation and clean up associated session documents."""
        conversation = self.get_object()
        conversation_id = str(conversation.id)

        # Delete the conversation
        response = super().destroy(request, *args, **kwargs)

        # Clean up associated session documents
        try:
            deleted_count = cleanup_session_documents(chat_id=conversation_id)
            logger.info(f"Deleted {deleted_count} session documents for conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Error cleaning up session documents for conversation {conversation_id}: {e}")

        return response


class ChatMessageView(viewsets.ViewSet):
    """
    API endpoint for processing chat messages.
    """
    permission_classes = [IsAuthenticated]

    def create(self, request):
        """Process an incoming chat message and generate a response."""
        serializer = ChatMessageSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        user_message = serializer.validated_data['message']
        conversation_id = serializer.validated_data.get('conversation_id')
        language = serializer.validated_data.get('language', 'en')
        use_multi_agent = serializer.validated_data.get('use_multi_agent', True)
        session_id = serializer.validated_data.get('session_id')

        # Get or create conversation
        if conversation_id:
            try:
                conversation = Conversation.objects.get(
                    id=conversation_id,
                    user=request.user
                )
            except Conversation.DoesNotExist:
                return Response(
                    {'error': 'Conversation not found'},
                    status=status.HTTP_404_NOT_FOUND
                )
        else:
            # Create a new conversation with a title based on the first message
            title = user_message[:50] + ('...' if len(user_message) > 50 else '')
            conversation = Conversation.objects.create(
                user=request.user,
                title=title,
                language=language
            )

        # Save the user's message
        user_message_obj = Message.objects.create(
            conversation=conversation,
            role='user',
            content=user_message
        )

        # Initialize notification system for WebSocket updates if using multi-agent
        notifier = None
        background_notifier = None

        if MULTI_AGENT_AVAILABLE and use_multi_agent:
            try:
                notifier = ProgressNotifier(
                    conversation_id=str(conversation.id),
                    user_id=str(request.user.id)
                )                # Start thinking notification
                notifier.send_thinking_start()

                # Start background notification thread
                background_notifier = BackgroundNotifier(notifier, max_time=120)
                background_notifier.start()
            except Exception as e:
                logger.error(f"Error initializing notification system: {e}")

        try:
            # Get context from relevant documents
            context, sources = self._get_context_for_query(user_message)

            # Get context from session documents if available - prioritize session documents
            # Always include session document context even if the query doesn't directly relate
            session_context, session_docs = self._get_session_document_context(
                user_message,
                session_id=session_id,
                chat_id=str(conversation.id)
            )

            # Combine contexts - place session context first as it's higher priority
            combined_context = ""
            if session_context:
                combined_context = session_context
            if context:
                combined_context = combined_context + "\n\n" + context if combined_context else context

            # Generate assistant response based on mode
            if MULTI_AGENT_AVAILABLE and use_multi_agent:
                response_data = self._generate_multi_agent_response(
                    user_message,
                    combined_context,
                    conversation,
                    notifier
                )
                response_text = response_data["message"]
                experts_used = response_data.get("expert_used", [])
            else:
                # Use standard RAG approach if multi-agent is not available or not requested
                response_text = self._generate_standard_response(user_message, combined_context, conversation)
                experts_used = []

            # Save the assistant's response
            assistant_message = Message.objects.create(
                conversation=conversation,
                role='assistant',
                content=response_text
            )

            # Update message with knowledge sources
            if sources:
                assistant_message.knowledge_sources.set(sources)

            # Update conversation title if this is the first exchange
            if conversation.messages.count() <= 2:  # User + Assistant message
                updated_title = user_message[:50] + ('...' if len(user_message) > 50 else '')
                conversation.title = updated_title
                conversation.save()

            # Prepare response object
            response = {
                'message': response_text,
                'conversation_id': conversation.id,
                'sources': [{'id': doc.id, 'title': doc.title} for doc in sources]
            }

            # Add session document sources if available
            if session_docs:
                session_sources = [{'id': doc.id, 'title': doc.title, 'type': 'session'} for doc in session_docs]
                response['session_sources'] = session_sources

            # Add experts if multi-agent was used
            if experts_used:
                response['experts'] = [
                    {'name': expert.get('name', ''), 'title': expert.get('title', '')}
                    for expert in experts_used
                ]
                response['multi_agent'] = True

            return Response(response)

        except Exception as e:
            logger.error(f"Error processing chat message: {e}", exc_info=True)
            return Response(
                {'error': _('An error occurred while processing your message.')},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        finally:
            # Stop background notification thread if it was started
            if background_notifier:
                background_notifier.stop()

            # Send completion notification
            if notifier:
                notifier.send_thinking_complete()

    def _get_session_document_context(self, query, session_id=None, chat_id=None):
        """
        Get relevant context from session documents based on the query.

        Args:
            query: The user query
            session_id: Optional browser session ID
            chat_id: Optional chat/conversation ID

        Returns:
            tuple: (context_text, list_of_session_document_objects)
        """
        if not session_id and not chat_id:
            return "", []

        try:
            # First, check if there are any session documents for this session/chat
            filter_kwargs = {}
            if session_id:
                filter_kwargs["session_id"] = session_id
            if chat_id:
                filter_kwargs["chat_id"] = chat_id

            session_documents = SessionDocument.objects.filter(**filter_kwargs)
            if not session_documents.exists():
                return "", []
              # Get all document summaries first
            summary_context_parts = []
            for doc in session_documents:
                if doc.content_preview and len(doc.content_preview) > 50:  # Ensure it's a real summary
                    summary_context_parts.append(
                        f"### Document Summary: {doc.title} ###\n"
                        f"{doc.content_preview}\n"
                    )

            # Search for relevant document chunks using vector search
            # Increased top_k to get more comprehensive context from session documents
            search_results = vector_search_session_documents(
                query_text=query,
                session_id=session_id,
                chat_id=chat_id,
                top_k=7  # Increased from 5 to 7 to provide more comprehensive context
            )

            # Get unique documents and format context
            chunk_context_parts = []
            session_docs = []
            seen_doc_ids = set()

            # search_results is a list of dictionaries, not a dictionary with a 'results' key
            if search_results and isinstance(search_results, list):
                for result in search_results:
                    doc_id = result.get('document_id')
                    if doc_id and doc_id not in seen_doc_ids:
                        try:
                            document = SessionDocument.objects.get(id=doc_id)
                            if document not in session_docs:
                                session_docs.append(document)
                                seen_doc_ids.add(doc_id)
                        except SessionDocument.DoesNotExist:
                            continue                    # Format this chunk as context with more detailed structure
                    chunk_context_parts.append(
                        f"## Excerpt from: {result.get('document_title', 'Unknown Document')} ##\n"
                        f"Page/Section: {result.get('page_number', 'N/A')}\n"
                        f"Relevance: {round(result.get('similarity', 0) * 100, 1)}% match\n"
                        f"Content: {result.get('content', '')}\n"
                    )

            # Combine summary context and chunk context
            all_context_parts = []

            # First add document summaries for all documents in the session
            if summary_context_parts:
                summary_section = "### DOCUMENT SUMMARIES ###\n\n" + "\n\n".join(summary_context_parts)
                all_context_parts.append(summary_section)

            # Then add specific chunks that match the query
            if chunk_context_parts:
                chunk_section = "### RELEVANT DOCUMENT SECTIONS ###\n\n" + "\n\n".join(chunk_context_parts)
                all_context_parts.append(chunk_section)

            # If we found no documents but have summaries, use the session documents
            if not session_docs and session_documents:
                session_docs = list(session_documents)

            # Combine all context parts
            context_text = "\n\n".join(all_context_parts)            # Add header to clearly identify this as session document content
            if context_text:
                context_text = "### SESSION DOCUMENT KNOWLEDGE BASE ###\n\n" + context_text

            # Add instruction to the AI to use this knowledge
            if context_text:
                context_text += "\n\n## IMPORTANT INSTRUCTIONS FOR HANDLING SESSION DOCUMENTS ##\n"
                context_text += "The information above comes from documents the user has uploaded specifically for this session. Follow these guidelines:\n"
                context_text += "1. This is high-priority knowledge that should override any conflicting information in your general knowledge.\n"
                context_text += "2. If the user's question relates to these documents, base your answer primarily on their content.\n"
                context_text += "3. When citing information from these documents, mention the document title.\n"
                context_text += "4. If the session documents don't contain relevant information for a question, use your general knowledge but acknowledge the limitations."

            return context_text, session_docs

        except Exception as e:
            logger.error(f"Error retrieving session document context: {e}")
            return "", []

    def _get_context_for_query(self, query):
        """
        Get relevant context from documents based on the query.

        Returns:
            tuple: (context_text, list_of_document_objects)
        """
        # Search for relevant documents
        search_results = search_documents(query)

        # Format context from search results
        context_parts = []
        sources = []

        for result in search_results:
            document = Document.objects.get(id=result['document_id'])

            # Add document to sources if not already included
            if document not in sources:
                sources.append(document)

            # Format this chunk as context
            context_parts.append(
                f"Document: {document.title}\n"
                f"Content: {result['content']}\n"
            )

        # Combine all context parts
        context_text = "\n\n".join(context_parts)

        return context_text, sources

    def _generate_standard_response(self, query, context, conversation):
        """Generate a response using the standard RAG approach."""
        # Get conversation history
        history = self._get_conversation_history(conversation)        # Build a standard system prompt for financial assistant
        system_prompt = """You are a helpful and knowledgeable financial assistant specializing in Nepal Financial Reporting Standards (NFRS) and International Financial Reporting Standards (IFRS).

Answer questions based on the provided context information.

PRIORITY INSTRUCTION: When answering questions, you must prioritize information from SESSION DOCUMENT KNOWLEDGE BASE sections over any other knowledge. These documents have been specifically uploaded by the user for this conversation and represent the most current and relevant information available. If information in these session documents contradicts your general knowledge, treat the session document information as authoritative.

If the context doesn't contain the relevant information, say so clearly rather than making up information. When citing information from session documents, reference the document title to help the user understand the source.

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

Use a clear, professional tone suitable for financial professionals."""

        try:
            # Try to use OpenAI API directly as fallback
            from openai import OpenAI

            # Prepare messages with system prompt, context, history, and query
            messages = [
                {"role": "system", "content": system_prompt}
            ]

            # Add conversation history
            for msg in history:
                messages.append(msg)            # Add context as system message if available
            if context:
                # Extract and highlight the SESSION DOCUMENT KNOWLEDGE BASE section if present
                if "### SESSION DOCUMENT KNOWLEDGE BASE ###" in context:
                    messages.append({
                        "role": "system",
                        "content": "ATTENTION: The following context contains SESSION DOCUMENT KNOWLEDGE BASE information that should be treated as the highest priority source of information:"
                    })

                messages.append({
                    "role": "system",
                    "content": f"Here is relevant context information to help answer the query:\n\n{context}"
                })

            # Add the user's query
            messages.append({"role": "user", "content": query})

            # Initialize OpenAI client
            client = OpenAI(api_key=settings.OPENAI_API_KEY)

            # Generate response
            response = client.chat.completions.create(
                model=getattr(settings, 'CHAT_MODEL', 'gpt-3.5-turbo'),
                messages=messages,
                temperature=0.3,
                max_tokens=1500
            )

            # Return the generated text
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error in standard response generation: {e}")
            # Ultimate fallback - return a simple message if all else fails
            return f"I apologize, but I encountered an issue processing your query about {query[:30]}... Please try again."

    def _generate_multi_agent_response(self, query, context, conversation, notifier=None):
        """
        Generate a response using the multi-agent system.

        Args:
            query: The user's query
            context: Context from relevant documents
            conversation: Conversation object
            notifier: Optional ProgressNotifier for WebSocket updates

        Returns:
            Dictionary with response data
        """        # Get conversation history
        history = self._get_conversation_history(conversation)

        # Initialize the multi-agent system
        multi_agent = MultiAgentChat(
            model_name=getattr(settings, 'CHAT_MODEL', 'gpt-3.5-turbo'),
            temperature=0.3,
            max_tokens=1500,
            system_instructions="PRIORITY INSTRUCTION: When answering questions, you must prioritize information from SESSION DOCUMENT KNOWLEDGE BASE sections over any other knowledge. These documents have been specifically uploaded by the user for this conversation and represent the most current and relevant information available. If information in these session documents contradicts your general knowledge, treat the session document information as authoritative."
        )

        # Process the query
        response_data = multi_agent.process_query(
            query=query,
            context=context,
            conversation_history=history
        )

        # Send notification about selected experts if notifier is available
        if notifier and 'expert_used' in response_data and response_data['expert_used']:
            notifier.send_expert_selection(response_data['expert_used'])

        return response_data

    def _build_standard_prompt(self, query, context, history):
        """
        Build the prompt for the standard RAG approach.

        Args:
            query: The user's query
            context: Context information
            history: Conversation history

        Returns:
            Formatted prompt string
        """
        # TODO: Implement actual prompt building
        # This is just a placeholder
        return f"Context: {context}\n\nQuestion: {query}"

    def _get_conversation_history(self, conversation, max_messages=10):
        """
        Get the conversation history formatted for the AI model.

        Args:
            conversation: Conversation object
            max_messages: Maximum number of messages to include

        Returns:
            List of message dictionaries
        """
        # Get the most recent messages in the conversation
        messages = conversation.messages.order_by('-created_at')[:max_messages]

        # Convert to format expected by LLM
        history = []
        for msg in reversed(messages):
            history.append({
                'role': msg.role,
                'content': msg.content
            })

        return history

    @action(detail=False, methods=['post'])
    def translate(self, request):
        """Translate a message to the specified language."""
        serializer = TranslateMessageSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        message_id = serializer.validated_data['message_id']
        target_language = serializer.validated_data['target_language']

        try:
            message = Message.objects.get(id=message_id)
            # Ensure the message belongs to a conversation owned by the user
            if message.conversation.user != request.user:
                return Response(
                    {'error': 'Message not found'},
                    status=status.HTTP_404_NOT_FOUND
                )

            # Translate the message
            translated_text = translate_text(message.content, target_language)

            return Response({
                'original': message.content,
                'translated': translated_text,
                'language': target_language
            })

        except Message.DoesNotExist:
            return Response(
                {'error': 'Message not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Translation error: {e}", exc_info=True)
            return Response(
                {'error': _('An error occurred during translation.')},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['get'])
    def session_documents(self, request):
        """
        Get session documents for a specific session ID or chat ID.
        Endpoint: {{base_url}}/api/v1/chat/messages/session-documents/
        """
        # Get query parameters
        session_id = request.query_params.get('session_id')
        chat_id = request.query_params.get('chat_id')

        # Require at least one filter parameter
        if not session_id and not chat_id:
            return Response(
                {"error": "Either session_id or chat_id query parameter is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Build filters
        filters = {'uploaded_by': request.user}
        if session_id:
            filters['session_id'] = session_id
        if chat_id:
            filters['chat_id'] = chat_id

        # Get documents with prefetched chunks for better performance
        documents = SessionDocument.objects.filter(**filters).prefetch_related('chunks')

        if not documents.exists():
            return Response([])

        # Serialize the documents with their chunks
        serializer = SessionDocumentSerializer(documents, many=True)

        return Response(serializer.data)