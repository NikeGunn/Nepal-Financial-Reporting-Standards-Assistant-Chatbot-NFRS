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
from django.contrib.auth.models import User
from datetime import date
from .models import Conversation, Message
from .serializers import (
    ConversationListSerializer, ConversationDetailSerializer,
    MessageSerializer, ChatMessageSerializer, TranslateMessageSerializer
)
from api.knowledge.models import Document, DocumentChunk, VectorIndex
from utils.vector_ops import vector_search as perform_vector_search
import logging

# Add LangChain imports
try:
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        AIMessagePromptTemplate
    )
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Falling back to standard OpenAI implementation.")

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
    Enhanced with LangChain for chain of thought reasoning.
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

            # Check if query is related to finance/NFRS before processing
            is_on_topic, confidence = self.check_topic_relevance(message_content)

            if not is_on_topic and confidence > 0.7:
                # Return a polite off-topic response if the query is clearly not related to finance
                off_topic_response = (
                    "I apologize, but I am specialized in financial reporting, auditing, and accounting matters "
                    "under NFRS and IFRS. Please ask a question related to finance or auditing, and I'd be happy "
                    "to assist with professional insights."
                )

                # Save assistant message
                assistant_message = Message.objects.create(
                    conversation=conversation,
                    role='assistant',
                    content=off_topic_response
                )

                # Update conversation's updated_at timestamp
                conversation.save()

                return Response({
                    "message": off_topic_response,
                    "conversation_id": conversation.id,
                    "sources": []
                })

            # Translate message if needed (from Nepali to English)
            search_query = message_content
            original_language = language
            if language == 'ne':
                try:
                    search_query = self.translate_text(message_content, target_language='en')
                except Exception as e:
                    logger.error(f"Translation error: {e}")
                    # Continue with original text if translation fails
                    search_query = message_content

            # Perform vector search to find relevant documents
            relevant_chunks = self.vector_search(search_query, top_k=5)  # Increased from 3 to 5 for better context

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

            # Get current date for prompt
            current_date = date.today().strftime("%B %d, %Y")

            # Use LangChain if available
            if LANGCHAIN_AVAILABLE:
                try:
                    return self.process_with_langchain(
                        message_content,
                        conversation,
                        user_message,
                        context,
                        referenced_documents,
                        original_language,
                        current_date
                    )
                except Exception as e:
                    logger.error(f"LangChain processing failed: {str(e)}. Falling back to standard processing.")
                    # Fall back to standard processing

            # Standard OpenAI processing (if LangChain is not available or failed)
            return self.process_with_openai(
                message_content,
                conversation,
                user_message,
                context,
                referenced_documents,
                language,
                current_date
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def process_with_langchain(self, message_content, conversation, user_message, context, referenced_documents, language, current_date):
        """
        Process the message using LangChain for chain of thought reasoning.
        """
        try:
            # Create the enhanced system prompt
            system_prompt = (
                "You are a highly qualified and deeply experienced **Chartered Accountant (CA)** and **Licensed Auditor**, "
                "accredited by the **Institute of Chartered Accountants of Nepal (ICAN)**, with over 15 years of rigorous, hands-on experience "
                "in corporate advisory, statutory audits, internal control evaluations, and financial compliance across both private and public sectors.\n\n"

                "## You Have Full Power to Show valuation, measurement, and disclosure requirements questions and options. "
                "You also have the authority to ask users targeted diagnostic questions to uncover whether correct valuation methods, measurement techniques, and disclosures "
                "are being used. You apply deep domain expertise to guide, question, and solve core financial, audit, tax, and compliance problems "
                "in the context of Nepal. Your recommendations are practical, strategic, and framed with the authority of a senior Chartered Accountant.\n"

                "## 👤 Your Role & Expertise\n"
                "You serve as a **core financial intelligence system** within a Retrieval-Augmented Generation (RAG) framework, "
                "offering deep interpretative insights into:\n"
                "- **Valuation, measurement, and disclosure requirements** under NFRS and IFRS\n"
                "- **Auditing practices** aligned with International Standards on Auditing (ISA)\n"
                "- **Regulatory compliance** with Nepal Rastra Bank (NRB) directives, Company Law, and Tax Regulations\n\n"

                "Your guidance is comparable to a **senior audit partner in a Big Four firm**, with the ability to:\n"
                "1. Probe into client practices using structured, context-driven diagnostic questions\n"
                "2. Detect gaps in accounting treatment, risk disclosures, provisioning, and revenue recognition\n"
                "3. Recommend corrective actions tailored to industry, size, and risk appetite of the entity\n"
                "4. Educate users by connecting real-world financial behavior with conceptual standards\n\n"

                "## 🧠 Core Areas of Technical Mastery:\n"
                "- **Valuation of assets, intangibles, and financial instruments** (NFRS 9, NFRS 13)\n"
                "- **Impairment testing, depreciation models, amortization schedules** (NFRS 36, NFRS 16)\n"
                "- **Fair presentation and true & fair view disclosures** in financial statements\n"
                "- **Judgment-intensive areas** like revenue recognition (NFRS 15), lease accounting (NFRS 16), and expected credit loss (NFRS 9)\n"
                "- **NRB compliance**, capital adequacy, and provisioning norms\n"
                "- **Forensic indicators**, fraud detection, and risk-based audit procedures\n"
                "- **Deferred tax**, **contingent liabilities**, and **going concern assessments** under Nepalese legal and financial environment\n\n"

                "## 🌍 Nepal-Specific Deep Domain Thinking:\n"
                "You answer only financial questions **related to Nepal**, including:\n"
                "- Financial reporting under **NFRS/NAS**\n"
                "- Compliance with **Nepal Rastra Bank (NRB)** directives for BFIs\n"
                "- Tax treatments under **Nepal Income Tax Act, VAT Act**, and local indirect taxes\n"
                "- Regulatory obligations under **Company Act 2063**, **BAFIA**, and **AML/KYC frameworks**\n"
                "- Real-world challenges like under-disclosure, loan misclassification, tax evasion indicators, improper provisioning, and undocumented adjustments\n\n"

                "You think and solve like a **revolutionary Nepali CA** who is transforming the profession with modern, AI-powered tools. Your vision is to bring financial clarity to 100 million users by providing:\n"
                "- 📈 Strategic clarity\n"
                "- 🧠 Expert-level diagnostics\n"
                "- 🔒 Regulatory confidence\n"
                "- 💰 Profit-focused corrections\n"
                "- ⚠️ Risk mitigations with professional insight\n\n"

                "## 📌 AI Functions within the RAG System:\n"
                "You assist users by leveraging FAISS-indexed documents from:\n"
                "- NFRS and IFRS official standards\n"
                "- NRB circulars and sectoral compliance mandates\n"
                "- ISAs, audit manuals, and templates\n"
                "- Interpretative commentaries from Nepalese financial experts\n\n"

                "## 🧭 When Answering:\n"
                "1. FIRST, take a deep breath and break down the question into its core components. Analyze what accounting standards, methods, or regulations are relevant.\n"
                "2. SECOND, reflect on how a senior Chartered Accountant in Nepal would approach this problem in practice, not just in theory.\n"
                "3. THIRD, consider any hidden risks, compliance issues, or reporting implications that might not be immediately obvious.\n"
                "4. FOURTH, identify whether the issue is primarily about **valuation**, **measurement**, **disclosure**, **compliance**, or **tax treatment**.\n"
                "5. FIFTH, apply a multi-layer analytical framework:\n"
                "   - **Standard layer**: What do the relevant NFRS, NAS, or IFRS standards literally state?\n"
                "   - **Regulatory layer**: What additional requirements apply from NRB, IRD, ICAN, or other regulators?\n"
                "   - **Practical layer**: How are these applied in Nepal's business environment?\n"
                "   - **Audit layer**: What would an auditor scrutinize most closely? What documentation is needed?\n"
                "   - **Risk layer**: What financial or compliance risks could emerge from different approaches?\n"
                "6. SIXTH, consider industry-specific implications (BFIs, manufacturing, hospitality, etc.) and size-based variations (listed vs. private, large vs. SME).\n"
                "7. SEVENTH, reason about temporal considerations - reporting timeline, financial year impacts, transition periods, future regulatory changes.\n"
                "8. EIGHTH, explore alternative treatments and their accounting, tax, and business consequences.\n"
                "9. FINALLY, synthesize your insights into a comprehensive, well-reasoned response with practical, actionable guidance.\n\n"

                "## 📝 Response Guidelines (Markdown + Emojis):\n"
                "- Use **bold** for standards, sections, and judgments\n"
                "- Use **##** for clear section headings\n"
                "- Use **bullets, numbered lists, tables** where relevant\n"
                "- Use emojis such as:\n"
                "  - 📉 for losses or financial red flags\n"
                "  - 📊 for analysis or financial ratios\n"
                "  - ✅ for compliant behavior or best practices\n"
                "  - ⚠️ for non-compliance, risk, or red flags\n"
                "  - 💰 for profitability guidance\n"
                "  - 🔍 for investigative insights or auditor mindset\n"
                "- Maintain a **formal, educative, solution-driven tone**\n\n"

                "## 🎯 Your Objectives:\n"
                "1. Help users solve deep financial and compliance problems with NFRS, NRB, and IFRS alignment\n"
                "2. Increase their business profitability and reduce risk\n"
                "3. Strengthen their understanding of disclosure, measurement, valuation, tax, and audit\n"
                "4. Act as the CA mind behind Nepal's next-gen financial advisory revolution\n\n"

                "## ❌ Scope Limitations:\n"
                "- Politely decline non-financial, non-Nepal-related questions:\n"
                "'I specialize in financial reporting, auditing, and regulatory compliance under NFRS, IFRS, NRB, and ISA within Nepalese context.'\n\n"

                f"📅 Today's date is {current_date}."
            )

            # If we have context from documents, add it to the system prompt
            if context:
                system_prompt += "\n\n## 📚 Relevant NFRS/IFRS Document Context:\n" + context
            else:
                system_prompt += "\n\n## ⚠️ Notice: No specific NFRS/IFRS document context was found for this query. Please rely on your general knowledge while being clear about any limitations."

            # Create system message template
            system_message = SystemMessagePromptTemplate.from_template(system_prompt)

            # Create the full prompt template
            message_templates = [system_message]

            # Add conversation history (up to 10 recent messages)
            history_messages = conversation.messages.order_by('-created_at')[:10]
            for msg in reversed(list(history_messages)):
                if msg.id != user_message.id:  # Skip the current message
                    if msg.role == 'user':
                        message_templates.append(HumanMessagePromptTemplate.from_template(msg.content))
                    elif msg.role == 'assistant':
                        message_templates.append(AIMessagePromptTemplate.from_template(msg.content))

            # Add the current user message
            message_templates.append(HumanMessagePromptTemplate.from_template(message_content))

            # Create the final chat prompt template
            chat_prompt = ChatPromptTemplate.from_messages(message_templates)

            # Initialize the ChatOpenAI model with fallback defaults if settings are missing
            llm = ChatOpenAI(
                temperature=0.3,
                model=getattr(settings, 'CHAT_MODEL', 'gpt-3.5-turbo'),
                api_key=getattr(settings, 'OPENAI_API_KEY', ''),
                max_tokens=1000,  # Increased for more comprehensive responses
            )

            # Create LangChain LLM chain with chain-of-thought
            chain = LLMChain(
                llm=llm,
                prompt=chat_prompt,
                verbose=getattr(settings, 'DEBUG', False)  # Only verbose in debug mode
            )

            # Run the chain
            assistant_response = chain.run({})

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
            # Log the error and re-raise to trigger fallback
            logger.error(f"Error in LangChain processing: {e}")
            raise

    def process_with_openai(self, message_content, conversation, user_message, context, referenced_documents, language, current_date):
        """
        Process the message using standard OpenAI API - fallback for when LangChain isn't available or fails.
        """
        try:
            # Prepare messages for OpenAI with enhanced system prompt for better guardrails
            system_prompt = (
                "You are a highly qualified and deeply experienced **Chartered Accountant (CA)** and **Licensed Auditor**, "
                "accredited by the **Institute of Chartered Accountants of Nepal (ICAN)**, with over 15 years of rigorous, hands-on experience "
                "in corporate advisory, statutory audits, internal control evaluations, and financial compliance across both private and public sectors.\n\n"

                "## You Have Full Power to Show valuation ,measurement and disclosure requirements questions and options And also have power to ask the user relevant question to know if the user has used valuation methods,measurement and disclosure and many more deep core knowledge and technique which is rquired on accounting, auditing, financial auditing, balance report auditing and suggesting directly what have to do and what not have to do, you can share deep knowledge and core technique how to tackle problem can get out of it and you are here to suggest them as a senior chattered accounted, suggest them in a respectfull way\n"

                "## 👤 Your Role & Expertise\n"
                "You serve as a **core financial intelligence system** within a Retrieval-Augmented Generation (RAG) framework, "
                "offering deep interpretative insights into:\n"
                "- **Valuation, measurement, and disclosure requirements** under NFRS and IFRS\n"
                "- **Auditing practices** aligned with International Standards on Auditing (ISA)\n"
                "- **Regulatory compliance** with Nepal Rastra Bank (NRB) directives, Company Law, and Tax Regulations\n\n"

                "Your guidance is comparable to a **senior audit partner in a Big Four firm**, with the ability to:\n"
                "1. Probe into client practices using structured, context-driven diagnostic questions\n"
                "2. Detect gaps in accounting treatment, risk disclosures, provisioning, and revenue recognition\n"
                "3. Recommend corrective actions tailored to industry, size, and risk appetite of the entity\n"
                "4. Educate users by connecting real-world financial behavior with conceptual standards\n\n"

                "## 🧠 Core Areas of Technical Mastery:\n"
                "- **Valuation of assets, intangibles, and financial instruments** (NFRS 9, NFRS 13)\n"
                "- **Impairment testing, depreciation models, amortization schedules** (NFRS 36, NFRS 16)\n"
                "- **Fair presentation and true & fair view disclosures** in financial statements\n"
                "- **Judgment-intensive areas** like revenue recognition (NFRS 15), lease accounting (NFRS 16), and expected credit loss (NFRS 9)\n"
                "- **NRB compliance**, capital adequacy, and provisioning norms\n"
                "- **Forensic indicators**, fraud detection, and risk-based audit procedures\n\n"

                "## 📌 AI Functions within the RAG System:\n"
                "As a virtual Chartered Accountant, you assist users by leveraging FAISS-indexed documents from:\n"
                "- NFRS and IFRS official standards\n"
                "- NRB circulars and sectoral compliance mandates\n"
                "- ISAs, audit manuals, risk-assessment templates\n"
                "- Interpretative commentaries from regulatory and academic sources\n\n"

                "## 🧭 When Answering:\n"
                "- Begin by identifying whether the issue is **valuation**, **measurement**, or **disclosure-related**.\n"
                "- Ask deep, structured follow-up questions:\n"
                "   - 'What valuation basis have you adopted for X?' \n"
                "   - 'Is this fair value derived using observable inputs (Level 1 or 2), or based on unobservable assumptions (Level 3)?'\n"
                "   - 'Have you disclosed assumptions and sensitivity analysis as required by NFRS 13?'\n"
                "- Provide audit-quality suggestions for remediation, compliance, or improvement.\n\n"

                "## 📝 Response Guidelines (Markdown + Emojis):\n"
                "- Use **bold** for standards, sections, and judgments\n"
                "- Use **##** for clear section headings\n"
                "- Use **bullets, numbered lists, tables** where relevant\n"
                "- Use emojis such as:\n"
                "  - 📉 for losses or financial red flags\n"
                "  - 📊 for analysis or financial ratios\n"
                "  - ✅ for compliant behavior or best practices\n"
                "  - ⚠️ for non-compliance, risk, or red flags\n"
                "  - 💰 for profitability guidance\n"
                "  - 🔍 for investigative insights or auditor mindset\n"
                "- Maintain a **formal, educative, solution-driven tone**\n\n"

                "## 🎯 Your Objectives:\n"
                "1. Help users solve deep financial and compliance problems with NFRS, NRB, and IFRS alignment\n"
                "2. Increase their business profitability and reduce risk\n"
                "3. Strengthen their understanding of disclosure, measurement, valuation, tax, and audit\n"
                "4. Act as the CA mind behind Nepal’s next-gen financial advisory revolution\n\n"

                "## ❌ Scope Limitations:\n"
                "- Decline questions outside financial, audit, tax, or accounting domains politely:\n"
                "'I specialize in financial reporting, auditing, and regulatory compliance under NFRS, IFRS, NRB, and ISA.'\n\n"

                f"📅 Today's date is {current_date}."
            )

            messages = [
                {"role": "system", "content": system_prompt},
            ]

            # Add context if available
            if context:
                messages.append({"role": "system", "content": context})
            else:
                # If no relevant documents found, add a note to be careful with responses
                messages.append({"role": "system", "content": (
                    "No specific NFRS/IFRS document context was found for this query. "
                    "Please only respond with general financial information if appropriate, "
                    "and be clear about limitations in your knowledge."
                )})

            # Add conversation history (up to 10 recent messages)
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
                    max_tokens=800,  # Slightly increased
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
                logger.error(f"OpenAI processing error: {e}")
                return Response(
                    {"error": str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        except Exception as e:
            logger.error(f"Error in fallback OpenAI processing: {e}")
            return Response(
                {"error": f"Error processing request: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    # Keep existing methods
    def check_topic_relevance(self, query_text):
        """
        Determines if a query is related to finance/NFRS or completely off-topic.
        Returns a tuple of (is_on_topic, confidence_score).
        """
        # Finance-related keywords to check against
        finance_keywords = [
            'nfrs', 'ifrs', 'balance sheet', 'profit', 'loss', 'audit', 'tax', 'revenue', 'expense',
            'asset', 'liability', 'equity', 'depreciation', 'financial statement', 'compliance',
            'nfrs', 'nepal financial reporting standard', 'financial', 'reporting', 'accounting',
            'standard', 'ifrs', 'gaap', 'balance sheet', 'income statement', 'cash flow',
            'audit', 'tax', 'revenue', 'expense', 'asset', 'liability', 'equity', 'depreciation',
            'amortization', 'accrual', 'fiscal', 'budget', 'finance', 'debit', 'credit', 'journal',
            'ledger', 'reconciliation', 'statement', 'disclosure', 'compliance', 'regulation',
            'nepal', 'financial statement', 'accounting policy', 'profit', 'loss', 'capital',
            'investment', 'dividend', 'interest', 'loan', 'debt', 'equity', 'shares', 'stock',
            'ican', 'auditor', 'cpa', 'chartered accountant', 'banking', 'investment'
        ]

        # Clearly off-topic categories
        off_topic_keywords = [
            'recipe', 'movie', 'sports', 'weather', 'travel', 'fashion', 'pet', 'game',
            'recipe', 'cooking', 'movie', 'song', 'actor', 'sports', 'game', 'play',
            'weather', 'travel', 'vacation', 'hotel', 'flight', 'dating', 'relationship',
            'exercise', 'workout', 'diet', 'weight', 'fashion', 'clothing', 'restaurant',
            'gardening', 'plant', 'pet', 'dog', 'cat', 'animal', 'wildlife'
        ]

        query_lower = query_text.lower()

        # Check for finance-related terms
        finance_matches = sum(1 for kw in finance_keywords if kw in query_lower)

        # Check for off-topic terms
        off_topic_matches = sum(1 for kw in off_topic_keywords if kw in query_lower)

        # Simple heuristic algorithm to determine relevance
        is_finance_related = finance_matches > 0
        is_clearly_off_topic = off_topic_matches > 0 and finance_matches == 0

        # Confidence calculation (simple version)
        if is_finance_related:
            confidence = 0.3 + min(0.7, finance_matches * 0.1)  # Scale with matches but cap at 1.0
        elif is_clearly_off_topic:
            confidence = 0.3 + min(0.7, off_topic_matches * 0.1)
        else:
            confidence = 0.5  # Neutral when we're not sure

        return (not is_clearly_off_topic, confidence)

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
                processing_status='completed',  # Only use fully processed documents
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


class UserConversationsView(generics.ListAPIView):
    """
    API endpoint for fetching conversations by username.
    This allows frontends to retrieve conversations for a specific user.
    """
    serializer_class = ConversationListSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # Get the username from the query parameters
        username = self.request.query_params.get('username', None)

        # Default to current user if no username provided
        if not username:
            return Conversation.objects.filter(user=self.request.user)

        # Check if the requesting user has admin permissions
        if not hasattr(self.request.user, 'profile') or not self.request.user.profile.is_admin:
            # Non-admin users can only see their own conversations
            if username != self.request.user.username:
                return Conversation.objects.none()
            return Conversation.objects.filter(user=self.request.user)

        # Admin users can see conversations for any user
        try:
            user = User.objects.get(username=username)
            return Conversation.objects.filter(user=user)
        except User.DoesNotExist:
            return Conversation.objects.none()