from rest_framework import generics, status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from django.conf import settings
import os
import openai
from openai import OpenAI  # Add import for new OpenAI client
import json
import numpy as np
from google.cloud import translate_v2 as translate
from .models import Conversation, Message
from .serializers import (
    ConversationListSerializer, ConversationDetailSerializer,
    MessageSerializer, ChatMessageSerializer, TranslateMessageSerializer
)
from api.knowledge.models import Document, DocumentChunk, VectorIndex
from utils.vector_ops import vector_search as perform_vector_search
import logging

# Add logger
logger = logging.getLogger(__name__)


class ConversationListCreateView(generics.ListCreateAPIView):
    """
    API endpoint for listing and creating conversations.
    """
    serializer_class = ConversationListSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class ConversationDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    API endpoint for retrieving, updating, and deleting conversations.
    """
    serializer_class = ConversationDetailSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if getattr(self, 'swagger_fake_view', False):
            return Conversation.objects.none()
        return Conversation.objects.filter(user=self.request.user)


class MessageDetailView(generics.RetrieveAPIView):
    """
    API endpoint for retrieving individual messages.
    """
    serializer_class = MessageSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if getattr(self, 'swagger_fake_view', False):
            return Message.objects.none()
        return Message.objects.filter(conversation__user=self.request.user)


class ChatMessageView(APIView):
    """
    API endpoint for sending and receiving chat messages.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = ChatMessageSerializer(data=request.data)
        if serializer.is_valid():
            message_content = serializer.validated_data['message']
            conversation_id = serializer.validated_data.get('conversation_id')
            language = serializer.validated_data.get('language', 'en')

            # Get or create conversation
            if conversation_id:
                try:
                    conversation = Conversation.objects.get(
                        id=conversation_id,
                        user=request.user
                    )
                except Conversation.DoesNotExist:
                    return Response(
                        {"error": "Conversation not found"},
                        status=status.HTTP_404_NOT_FOUND
                    )
            else:
                conversation = Conversation.objects.create(
                    user=request.user,
                    language=language
                )

            # Save user message
            user_message = Message.objects.create(
                conversation=conversation,
                role='user',
                content=message_content
            )

            # Translate message if needed (from Nepali to English)
            search_query = message_content
            if language == 'ne':
                try:
                    search_query = self.translate_text(message_content, target_language='en')
                except Exception as e:
                    logger.error(f"Translation error: {e}")
                    # Continue with original text if translation fails
                    search_query = message_content

            # Perform vector search to find relevant documents
            relevant_chunks = self.vector_search(search_query, top_k=3)

            # Build context from relevant document chunks
            context = ""
            referenced_documents = []

            if relevant_chunks:
                context = "Information from NFRS documents:\n\n"
                for chunk in relevant_chunks:
                    context += f"{chunk['content']}\n\n"
                    # Add document to referenced documents if not already there
                    if chunk['document_id'] not in [doc.id for doc in referenced_documents]:
                        try:
                            document = Document.objects.get(id=chunk['document_id'])
                            referenced_documents.append(document)
                        except Document.DoesNotExist:
                            pass

            # Prepare messages for OpenAI
            messages = [
                {"role": "system", "content": f"You are an NFRS Assistant, knowledgeable about Nepal Forest Research and Survey. Answer questions based on the provided context. If the answer is not in the context, say you don't know. Don't mention the context directly in your answer. Today's date is {settings.CURRENT_DATE if hasattr(settings, 'CURRENT_DATE') else 'not specified'}."},
            ]

            # Add context if available
            if context:
                messages.append({"role": "system", "content": context})

            # Add conversation history (up to 5 recent messages)
            history_messages = conversation.messages.order_by('-created_at')[:10]
            for msg in reversed(list(history_messages)):
                if msg.id != user_message.id:  # Skip the current message
                    messages.append({"role": msg.role, "content": msg.content})

            # Add the current user message
            messages.append({"role": "user", "content": message_content})

            # Call OpenAI API
            try:
                # Initialize the OpenAI client with the API key
                client = OpenAI(api_key=settings.OPENAI_API_KEY)

                # Call the completions API using the new format
                response = client.chat.completions.create(
                    model=settings.CHAT_MODEL,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7,
                )

                # Extract the assistant's response using the new response format
                assistant_response = response.choices[0].message.content.strip()

                # Translate response if language is Nepali
                if language == 'ne':
                    try:
                        assistant_response = self.translate_text(assistant_response, target_language='ne')
                    except Exception as e:
                        logger.error(f"Translation error: {e}")
                        # Continue with English response if translation fails

                # Save assistant message
                assistant_message = Message.objects.create(
                    conversation=conversation,
                    role='assistant',
                    content=assistant_response
                )

                # Add referenced documents
                if referenced_documents:
                    for doc in referenced_documents:
                        assistant_message.knowledge_sources.add(doc)

                # Update conversation's updated_at timestamp
                conversation.save()

                # Return the response
                return Response({
                    "message": assistant_response,
                    "conversation_id": conversation.id,
                    "sources": [{"id": doc.id, "title": doc.title} for doc in referenced_documents]
                })

            except Exception as e:
                return Response(
                    {"error": str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def vector_search(self, query, top_k=3):
        """
        Perform vector search using the vector operations utility.
        With fallback for missing/empty vector index.
        """
        try:
            # Use the vector_search function from utils.vector_ops
            results = perform_vector_search(query, top_k=top_k)

            # If we got results, return them
            if results:
                return results

            # Log the issue but don't fail the request
            logger.warning("Vector search returned empty results. This might be due to missing vector index or no relevant content.")

            # Check if we have any active index in the database
            from api.knowledge.models import VectorIndex, Document

            # Check if there are any indices in the database
            if not VectorIndex.objects.filter(is_active=True).exists():
                logger.error("No active vector index found in database")

                # Try to see if we need to create a new index
                index_files = [f for f in os.listdir(settings.VECTOR_STORE_DIR)
                              if f.endswith('.index') and os.path.isfile(os.path.join(settings.VECTOR_STORE_DIR, f))]

                if index_files:
                    # Found index files but no database entry - create one
                    index_file = index_files[0]  # Use the first one found
                    logger.info(f"Creating database entry for existing index file: {index_file}")

                    index_path = os.path.join(settings.VECTOR_STORE_DIR, index_file)
                    index_name = os.path.splitext(index_file)[0]

                    # Create vector index entry
                    VectorIndex.objects.create(
                        name=index_name,
                        description="Auto-recovered NFRS documents vector index",
                        index_file_path=index_path,
                        is_active=True
                    )

                    # Try the search again
                    return perform_vector_search(query, top_k=top_k)

            # If we have no vector index or results, fall back to a basic keyword search
            # This ensures the user still gets some response even without vector search
            logger.info("Falling back to basic keyword search")
            documents = Document.objects.filter(
                status='processed',
                is_deleted=False
            ).order_by('-created_at')[:5]  # Get 5 most recent documents

            basic_results = []
            for doc in documents:
                # Get a chunk from this document
                chunks = DocumentChunk.objects.filter(document=doc)[:1]
                if chunks.exists():
                    chunk = chunks.first()
                    basic_results.append({
                        'score': 0.5,  # Default score
                        'document_id': doc.id,
                        'chunk_id': chunk.id,
                        'content': f"Document: {doc.title}\n\n{chunk.content[:500]}",
                        'page_number': chunk.page_number
                    })

            if basic_results:
                return basic_results

            return []

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    def translate_text(self, text, target_language='ne'):
        """
        Translate text between English and Nepali.
        """
        try:
            # Set Google credentials path - first check for project root path
            google_credentials_path = os.path.join(settings.BASE_DIR, 'google-credentials.json')
            if os.path.exists(google_credentials_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_path
            elif hasattr(settings, 'GOOGLE_APPLICATION_CREDENTIALS') and settings.GOOGLE_APPLICATION_CREDENTIALS:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = settings.GOOGLE_APPLICATION_CREDENTIALS

            # Check if credentials file exists
            if not os.path.exists(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')):
                logger.error(f"Google credentials file not found at {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
                raise FileNotFoundError(f"Google credentials file not found at {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")

            client = translate.Client()
            source_language = 'ne' if target_language == 'en' else 'en'

            result = client.translate(
                text,
                target_language=target_language,
                source_language=source_language
            )

            return result['translatedText']
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # If translation fails, return original text
            return text


class TranslateMessageView(APIView):
    """
    API endpoint for translating messages.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = TranslateMessageSerializer(data=request.data)
        if serializer.is_valid():
            message_id = serializer.validated_data['message_id']
            target_language = serializer.validated_data['target_language']

            try:
                # Get the message and check user permissions
                message = Message.objects.get(id=message_id, conversation__user=request.user)

                # Use the more robust utility function from utils.translation
                from utils.translation import translate_text
                source_language = 'en' if target_language == 'ne' else 'ne'

                translated_text = translate_text(
                    message.content,
                    target_language=target_language,
                    source_language=source_language
                )

                return Response({
                    "original_text": message.content,
                    "translated_text": translated_text,
                    "source_language": source_language,
                    "target_language": target_language
                })

            except Message.DoesNotExist:
                return Response(
                    {"error": "Message not found"},
                    status=status.HTTP_404_NOT_FOUND
                )
            except Exception as e:
                logger.error(f"Translation error: {e}")
                return Response(
                    {"error": str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)