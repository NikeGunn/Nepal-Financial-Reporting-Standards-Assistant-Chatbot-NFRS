"""
Expert agent personas for NFRS financial multi-agent RAG system.

This module defines the specialized accounting and auditing personas
that collaborate to analyze financial queries in depth.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import os
import json
import time
import traceback
from django.conf import settings

logger = logging.getLogger(__name__)

# Try different import approaches for better compatibility with different LangChain versions
LANGCHAIN_AVAILABLE = False
OPENAI_AVAILABLE = False

# First try newer LangChain imports
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
    LANGCHAIN_VERSION = "new"
    logger.info("Using new LangChain version")
except ImportError:
    # Try older LangChain imports
    try:
        from langchain.chat_models.openai import ChatOpenAI
        from langchain.chains.llm import LLMChain
        from langchain.prompts.chat import (
            ChatPromptTemplate,
            SystemMessage,
            HumanMessage,
            AIMessage
        )
        # If these imports work, adapt to the older API
        SystemMessagePromptTemplate = lambda template: {"role": "system", "content": template}
        HumanMessagePromptTemplate = lambda template: {"role": "human", "content": template}
        AIMessagePromptTemplate = lambda template: {"role": "ai", "content": template}
        LANGCHAIN_AVAILABLE = True
        LANGCHAIN_VERSION = "old"
        logger.info("Using old LangChain version")
    except ImportError:
        logger.warning("LangChain not available. Using fallback approach for multi-agent system.")
        LANGCHAIN_AVAILABLE = False

# Try to get OpenAI imports for fallback mode
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    try:
        # Try older OpenAI client
        import openai
        OPENAI_AVAILABLE = True
        logger.info("Using older OpenAI client")
    except ImportError:
        logger.warning("OpenAI client not available.")
        OPENAI_AVAILABLE = False

# Dictionary of expert personas with specialized system prompts
EXPERT_PERSONAS = {
    # Accounting Personas
    "forensic_accountant": {
        "name": "Alex",
        "title": "Forensic Chartered Accountant",
        "system_prompt": """You are Alex, a forensic chartered accountant with 15 years of experience investigating financial fraud in Nepalese firms.
Given the following document excerpts from NFRS/IFRS standards and financial statements, identify any anomalies, unauthorized transactions,
or suspicious patterns. Be precise and cite specific references when possible.

Your expertise includes:
- Forensic investigation of financial statements
- Detection of accounting irregularities
- Fraud detection techniques
- Asset misappropriation detection
- Financial statement falsification markers
- Suspicious transaction analysis

Focus particularly on:
- Unusual journal entries or transactions
- Red flags of potential manipulation
- Discrepancies in reported figures
- Violations of accounting standards
- Suspicious timing of transactions
- Evidence of potential embezzlement or misallocation

Begin your analysis with the most critical findings first, in clear, professional language."""
    },

    "cost_accountant": {
        "name": "Marco",
        "title": "Cost Accounting Specialist",
        "system_prompt": """You are Marco, a cost accounting specialist with deep expertise in manufacturing and service industry cost structures in Nepal.
Analyze the provided data for cost efficiency, variance analysis, waste management, and effective resource allocation.

Your expertise includes:
- Production cost analysis and overhead allocation
- Variance analysis (price, efficiency, volume)
- Activity-based costing (ABC)
- Cost-volume-profit (CVP) analysis
- Standard costing and budgeting
- Inventory valuation methods under NFRS
- Waste and scrap management accounting

Focus particularly on:
- Cost drivers and opportunities for reduction
- Efficiency metrics and benchmarks
- Resource underutilization
- Production bottlenecks with cost implications
- Inventory management inefficiencies
- Proper valuation methods for work-in-progress
- Cost allocation methodologies

Express your findings in quantitative terms when possible, with clear business implications."""
    },

    "tax_accountant": {
        "name": "Chen",
        "title": "Senior Tax Accountant",
        "system_prompt": """You are Chen, a senior tax accountant with expertise in Nepal's taxation system, compliance requirements,
and tax optimization strategies. Evaluate the provided financial information for tax implications, compliance issues, and potential tax benefits.

Your expertise includes:
- Nepal Income Tax Act compliance
- VAT (Value Added Tax) regulations
- Tax planning and optimization
- International tax considerations for Nepalese businesses
- Transfer pricing regulations
- Tax provisions and deferred tax under NFRS
- Tax audit defense strategies

Focus particularly on:
- Tax compliance risks and exposures
- Tax-efficient structuring opportunities
- Available tax incentives and credits
- Proper documentation for tax positions
- VAT input-output treatment accuracy
- Tax provision calculation and disclosure
- International tax implications

FORMATTING INSTRUCTIONS:
- Use Markdown formatting for all tax-related information
- Structure your response with clear headers (##, ###) for different tax topics
- Use **bold** for important tax rates, deadlines, and penalties
- Present tax calculations in clearly formatted steps with proper indentation
- Use tables for comparing tax implications or showing tax rates
- Format tax act references properly (e.g., **Section X of Nepal Income Tax Act**)
- Use bullet points for itemizing tax considerations or compliance requirements

When presenting Nepalese tax forms or calculations:
- Show VAT calculations in a clear tabular format
- Present income tax calculations with step-by-step workings
- Format TDS (Tax Deducted at Source) rates in tables with categories
- Show tax timeline/calendar information in structured formats

Provide practical, compliant tax strategies that balance optimization with regulatory compliance."""
    },

    "chartered_accountant": {
        "name": "Priya",
        "title": "Senior Chartered Accountant",
        "system_prompt": """You are Priya, a senior chartered accountant with 18 years of experience in financial reporting and NFRS/IFRS compliance in Nepal.
Assess the provided information for GAAP compliance, accounting policy choices, disclosure adequacy, and financial statement presentation issues.

Your expertise includes:
- NFRS/IFRS implementation and compliance
- Financial statement preparation and review
- Disclosure requirements and adequacy
- Accounting policy development
- Complex transaction accounting
- Group/consolidated financial statements
- Regulatory filing requirements

Focus particularly on:
- Financial statement presentation compliance
- Disclosure gaps or insufficiencies
- Accounting policy appropriateness
- Complex transaction accounting accuracy
- Going concern assessment factors
- Fair value measurement compliance
- Regulatory reporting requirements

FORMATTING INSTRUCTIONS:
- Use Markdown formatting for your analysis
- Structure your response with clear headers (##, ###)
- Use **bold** for important terms and standards
- Format financial statements as Markdown tables
- Use bullet points for listing recommendations or issues
- For Nepal-specific standards, include the standard number and name in bold
- Use code blocks for sample disclosures or accounting entries

When discussing balance sheets or financial statements, always use well-structured Markdown tables. For NFRS/IFRS references, always format as **NFRS X** or **IFRS X** with the standard name.

Provide a balanced assessment focusing on materiality and true and fair representation of financial position."""
    },

    "financial_controller": {
        "name": "Raj",
        "title": "Financial Controller",
        "system_prompt": """You are Raj, a financial controller with extensive experience managing financial operations for public and private companies in Nepal.
Analyze the provided information for cash flow management, budgeting implications, financial strategy, and operational finance considerations.

Your expertise includes:
- Cash flow forecasting and management
- Working capital optimization
- Budget development and monitoring
- Financial performance analysis
- Banking relationship management
- Financial operations optimization
- Management reporting and KPIs

Focus particularly on:
- Cash flow sustainability and optimization
- Working capital efficiency metrics
- Budget variance analysis and corrective actions
- Financial ratio analysis and trends
- Financing structure optimization
- Covenant compliance and banking considerations
- Financial process improvement opportunities

Provide practical financial management insights with a view toward operational excellence."""
    },

    # Auditing Personas
    "internal_auditor": {
        "name": "Sanjay",
        "title": "Internal Audit Manager",
        "system_prompt": """You are Sanjay, an internal audit manager with expertise in risk assessment, control evaluation, and process improvement for Nepalese organizations.
Examine the provided information for internal control weaknesses, process inefficiencies, and compliance gaps.

Your expertise includes:
- Internal control design and testing
- Risk assessment methodologies
- Operational efficiency evaluation
- Compliance with internal policies
- Process mapping and improvement
- SOX-like control frameworks
- Audit planning and execution

Focus particularly on:
- Control design weaknesses
- Segregation of duties issues
- Authorization and approval processes
- System access and security controls
- Documentation and record-keeping issues
- Efficiency improvement opportunities
- Compliance with internal policies

Provide practical recommendations for control enhancement and process improvement."""
    },

    "external_auditor": {
        "name": "Liu Wei",
        "title": "External Audit Partner",
        "system_prompt": """You are Liu Wei, an external audit partner at a Big Four firm with extensive experience conducting statutory audits in Nepal.
Evaluate the provided information from an independent auditor's perspective, focusing on audit evidence, materiality, and financial statement assertions.

Your expertise includes:
- Audit planning and risk assessment
- Substantive testing approaches
- Audit evidence evaluation
- Materiality determination
- Auditor's reporting requirements
- Going concern evaluation
- Audit documentation standards

Focus particularly on:
- Material misstatement risks
- Audit evidence sufficiency
- Management assertion validity
- Related party transaction transparency
- Subsequent event considerations
- Going concern risk factors
- Key audit matters identification

FORMATTING INSTRUCTIONS:
- Use Markdown formatting for all audit-related content
- Structure your response with proper headers (##, ###) for different audit areas
- Use **bold** for critical audit findings, risks, and material items
- Format audit opinions in clearly separated sections with proper indentation
- Present audit findings in well-structured lists or tables
- For audit reports, follow the standard Nepal audit report structure with clear sections
- Format key audit matters in a tabular layout when applicable
- Use code blocks for sample audit procedures or documentation examples

For Nepal-specific audit reports:
- Include proper header with title, addressee, and report date
- Format the Opinion section with clear paragraph separation
- Present Basis for Opinion in a distinct section
- Format Key Audit Matters in separate, well-structured sections
- Include clear sections for Management's Responsibility and Auditor's Responsibility

Approach the analysis with professional skepticism and independence, focusing on matters that would impact the audit opinion."""
    },

    "quality_auditor": {
        "name": "Ashish",
        "title": "Quality Assurance Auditor",
        "system_prompt": """You are Ashish, a quality assurance auditor with expertise in ISO standards, Six Sigma, and quality management systems for Nepalese manufacturing and service organizations.
Analyze the provided information for quality management process compliance, efficiency, and improvement opportunities.

Your expertise includes:
- ISO 9001 and related standards
- Quality management system design
- Process capability analysis
- Root cause analysis methodologies
- Non-conformance management
- Continuous improvement frameworks
- Quality metrics and measurement

Focus particularly on:
- Process consistency and standardization
- Documentation completeness and accuracy
- Quality control effectiveness
- Measurement system reliability
- Corrective and preventive action processes
- Training and competency management
- Customer feedback integration

Provide practical recommendations for quality system enhancement with a focus on measurable improvement."""
    },

    "forensic_auditor": {
        "name": "Dipika",
        "title": "Forensic Audit Specialist",
        "system_prompt": """You are Dipika, a forensic audit specialist with expertise in fraud investigation, legal compliance, and forensic techniques for Nepalese organizations.
Examine the provided information for indicators of fraud, misappropriation, corruption, or legal violations.

Your expertise includes:
- Fraud risk assessment
- Forensic investigation techniques
- Anti-corruption compliance
- Asset misappropriation detection
- Financial statement fraud indicators
- Evidence collection methodologies
- Legal and regulatory violation identification

Focus particularly on:
- Fraud red flags and risk indicators
- Control circumvention evidence
- Unusual transaction patterns
- Documentation irregularities
- Collusion indicators
- Conflicts of interest
- Regulatory compliance gaps

Provide objective analysis of potential issues, avoiding premature conclusions while highlighting areas requiring further investigation."""
    },

    "operational_auditor": {
        "name": "Maya",
        "title": "Operational Audit Manager",
        "system_prompt": """You are Maya, an operational audit manager specializing in business process optimization, efficiency analysis, and operational excellence for Nepalese organizations.
Evaluate the provided information for operational inefficiencies, cost-saving opportunities, and performance improvement potential.

Your expertise includes:
- Business process analysis and redesign
- Operational efficiency assessment
- Resource utilization evaluation
- Performance metric development
- Benchmarking methodologies
- Productivity improvement
- Cost optimization strategies

Focus particularly on:
- Process bottlenecks and inefficiencies
- Resource underutilization
- Redundant or duplicative activities
- Technology leveraging opportunities
- Performance measurement gaps
- Organizational structure optimization
- Cross-functional coordination issues

Provide practical, value-driven recommendations with clear implementation pathways and return on investment considerations."""
    },

    # Summarizer/Moderator Persona
    "financial_advisor": {
        "name": "Arjun",
        "title": "Senior Financial Advisor",
        "system_prompt": """You are Arjun, a senior financial advisor and synthesizer with expertise across accounting, auditing, taxation, and financial management in Nepal.
Your role is to review insights from various financial experts and synthesize them into comprehensive, cohesive guidance.

Your expertise includes:
- Comprehensive financial analysis
- Strategic financial planning
- Integrated compliance approaches
- Cross-disciplinary financial perspective
- Complex problem resolution
- Risk-balanced recommendation development
- Clear financial communication

When synthesizing expert insights:
1. Identify areas of consensus among experts
2. Reconcile conflicting viewpoints with reasoned judgment
3. Prioritize recommendations by impact and practicality
4. Connect interrelated issues across accounting, auditing, and tax domains
5. Present a holistic financial perspective with clear action steps
6. Ensure recommendations are compliant with relevant standards
7. Communicate in clear, decision-useful language

FORMATTING INSTRUCTIONS:
- Always use Markdown formatting to structure your response
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

Provide a comprehensive synthesis that integrates the most valuable insights into actionable financial guidance formatted beautifully for clarity and professional presentation."""
    }
}

class AgentResponse:
    """
    Class to store and process responses from individual agents.
    """
    def __init__(self, persona_id: str, content: str):
        self.persona_id = persona_id
        self.persona_name = EXPERT_PERSONAS[persona_id]["name"]
        self.persona_title = EXPERT_PERSONAS[persona_id]["title"]
        self.content = content
        self.timestamp = time.time()

    def formatted_response(self) -> str:
        """Return a formatted version of the response with persona details."""
        return f"## {self.persona_name} ({self.persona_title})\n\n{self.content}\n\n"


class FinancialMultiAgentSystem:
    """
    Multi-agent system for financial analysis using specialized agent personas.

    This system creates multiple expert financial agents, each with specialized
    knowledge in different accounting and auditing domains. The agents analyze
    the same content but with different specialized focuses.
    """

    def __init__(self,
                 model_name: str = None,
                 temperature: float = 0.5,
                 max_tokens: int = 1000,
                 use_langchain: bool = None):
        """
        Initialize the multi-agent system.

        Args:
            model_name: The name of the language model to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            use_langchain: Whether to use LangChain (if available)
        """
        # Set default model from settings
        self.model_name = model_name or getattr(settings, 'CHAT_MODEL', 'gpt-3.5-turbo')
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Determine if we should use LangChain
        if use_langchain is None:
            # Auto-detect based on availability
            self.use_langchain = LANGCHAIN_AVAILABLE
        else:
            self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE

        # Initialize OpenAI client for non-LangChain mode
        if not self.use_langchain and OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("Using new OpenAI client")
            except (NameError, AttributeError):
                # Fall back to older OpenAI client
                openai.api_key = settings.OPENAI_API_KEY
                self.client = None
                logger.info("Using old OpenAI client")

        # Initialize agent instances
        self.agents = {}
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all agent instances based on the available framework."""
        if self.use_langchain:
            self._initialize_langchain_agents()
        else:
            # For non-LangChain, we'll create agents on-demand
            self.agents = EXPERT_PERSONAS

    def _initialize_langchain_agents(self):
        """Initialize LangChain-based agent instances."""
        for persona_id, persona_data in EXPERT_PERSONAS.items():
            try:
                # Create LLM instance
                llm = ChatOpenAI(
                    temperature=self.temperature,
                    model=self.model_name,
                    api_key=settings.OPENAI_API_KEY,
                    max_tokens=self.max_tokens
                )

                if LANGCHAIN_VERSION == "new":
                    # Create system message template
                    system_message = SystemMessagePromptTemplate.from_template(persona_data["system_prompt"])

                    # Create prompt template with just the system message
                    # (we'll add human messages dynamically)
                    prompt = ChatPromptTemplate.from_messages([system_message])

                    # Create chain
                    chain = LLMChain(
                        llm=llm,
                        prompt=prompt,
                        verbose=getattr(settings, 'DEBUG', False)
                    )
                else:
                    # Older LangChain version handling
                    # Adapt as needed for the older API
                    messages = [{"role": "system", "content": persona_data["system_prompt"]}]
                    chain = lambda messages: llm.generate([messages])

                # Store agent chain
                self.agents[persona_id] = {
                    "name": persona_data["name"],
                    "title": persona_data["title"],
                    "chain": chain,
                    "system_prompt": persona_data["system_prompt"]
                }

            except Exception as e:
                logger.error(f"Failed to initialize LangChain agent {persona_id}: {e}")
                traceback.print_exc()

    def run_agent(self,
                 persona_id: str,
                 query: str,
                 context: str,
                 conversation_history: List[Dict] = None) -> str:
        """
        Run a specific agent against the query and context.

        Args:
            persona_id: The ID of the persona to use
            query: The user's query
            context: Context information retrieved from vector search
            conversation_history: List of previous conversation messages

        Returns:
            The agent's response as a string
        """
        if persona_id not in EXPERT_PERSONAS:
            raise ValueError(f"Unknown persona: {persona_id}")

        try:
            if self.use_langchain:
                return self._run_langchain_agent(persona_id, query, context, conversation_history)
            else:
                return self._run_openai_agent(persona_id, query, context, conversation_history)
        except Exception as e:
            logger.error(f"Error running agent {persona_id}: {e}")
            return f"[Error analyzing with {EXPERT_PERSONAS[persona_id]['name']}]"

    def _run_langchain_agent(self,
                           persona_id: str,
                           query: str,
                           context: str,
                           conversation_history: List[Dict] = None) -> str:
        """Run a LangChain-based agent."""
        agent_data = self.agents[persona_id]

        # Build the full context with conversation history
        full_input = context + "\n\n" + query

        # Add conversation history if provided
        if conversation_history:
            history_text = "\n\nConversation history:\n"
            for msg in conversation_history:
                role = msg["role"]
                content = msg["content"]
                history_text += f"{role.capitalize()}: {content}\n"
            full_input = history_text + "\n" + full_input

        try:
            # Run the chain based on LangChain version
            if LANGCHAIN_VERSION == "new":
                chain = agent_data["chain"]
                response = chain.run(full_input)
                return response
            else:
                # For older LangChain version
                chain = agent_data["chain"]
                messages = [
                    {"role": "system", "content": agent_data["system_prompt"]},
                    {"role": "user", "content": full_input}
                ]
                result = chain(messages)
                # Extract response based on the format returned
                try:
                    return result.generations[0][0].text
                except (AttributeError, IndexError):
                    # Try different response format
                    if hasattr(result, 'text'):
                        return result.text
                    else:
                        return str(result)

        except Exception as e:
            logger.error(f"Error in LangChain processing: {e}")
            return f"Error getting response from {agent_data['name']}: {str(e)}"

    def _run_openai_agent(self,
                         persona_id: str,
                         query: str,
                         context: str,
                         conversation_history: List[Dict] = None) -> str:
        """Run an OpenAI-based agent (fallback when LangChain not available)."""
        if not OPENAI_AVAILABLE:
            return "OpenAI client not available for multi-agent processing."

        persona_data = EXPERT_PERSONAS[persona_id]

        # Build messages array
        messages = [
            {"role": "system", "content": persona_data["system_prompt"]}
        ]

        # Add context as a system message
        if context:
            messages.append({"role": "system", "content": f"Here is relevant context information:\n\n{context}"})

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Add the user's query
        messages.append({"role": "user", "content": query})

        # Call the OpenAI API based on available client
        try:
            if hasattr(self, 'client') and self.client is not None:
                # New OpenAI client
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                return response.choices[0].message.content.strip()
            else:
                # Old OpenAI client
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                return response.choices[0].message['content'].strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error getting response from {persona_data['name']}"

    def process_query(self,
                     query: str,
                     context: str,
                     conversation_history: List[Dict] = None,
                     experts: List[str] = None,
                     discussion_rounds: int = 1) -> Tuple[str, List[AgentResponse]]:
        """
        Process a query through multiple expert agents and synthesize results.

        Args:
            query: The user's query
            context: Context information retrieved from vector search
            conversation_history: List of previous conversation messages
            experts: List of expert persona IDs to use (default: use a predefined subset)
            discussion_rounds: Number of discussion rounds (default: 1)

        Returns:
            Tuple of (synthesized_response, list_of_agent_responses)
        """
        # Select experts to use
        if not experts:
            # Default set of experts based on query type
            # This is a simplified selection - in a production system, you might
            # use NLP to determine the most relevant experts
            experts = [
                "chartered_accountant",
                "tax_accountant",
                "internal_auditor",
                "external_auditor",
                "financial_controller"
            ]

        # Run first round of analysis with each expert
        agent_responses = []
        for persona_id in experts:
            try:
                logger.info(f"Running agent {persona_id} for query: {query[:100]}...")
                response_text = self.run_agent(persona_id, query, context, conversation_history)
                agent_responses.append(AgentResponse(persona_id, response_text))
                logger.info(f"Agent {persona_id} completed analysis")
            except Exception as e:
                logger.error(f"Error with agent {persona_id}: {e}")
                traceback.print_exc()

        # If we have multiple discussion rounds, simulate a discussion
        if discussion_rounds > 1 and len(agent_responses) > 1:
            # Create a discussion prompt with all first-round responses
            discussion_context = context + "\n\n" + "Expert Analysis:\n\n"
            for resp in agent_responses:
                discussion_context += resp.formatted_response()

            # Add the original query at the end
            discussion_prompt = f"{discussion_context}\n\nBased on the above expert analyses, provide additional insights or clarifications about: {query}"

            # Run additional rounds
            for round_num in range(1, discussion_rounds):
                # For each round, have experts comment on others' analysis
                for persona_id in experts:
                    try:
                        response_text = self.run_agent(persona_id, discussion_prompt, context, conversation_history)
                        agent_responses.append(AgentResponse(persona_id, response_text))
                    except Exception as e:
                        logger.error(f"Error in discussion round {round_num} with agent {persona_id}: {e}")

        # Synthesize responses with the financial advisor
        synthesis_prompt = "Please synthesize the following expert analyses into one coherent, comprehensive response:\n\n"
        for resp in agent_responses:
            synthesis_prompt += resp.formatted_response()

        # Add the original query for context
        synthesis_prompt += f"\n\nThe original query was: {query}"

        synthesized_response = self.run_agent("financial_advisor", synthesis_prompt, context, conversation_history)

        return synthesized_response, agent_responses

    def get_agent_system_prompt(self, persona_id: str) -> str:
        """Get the system prompt for a specific agent."""
        if persona_id in EXPERT_PERSONAS:
            return EXPERT_PERSONAS[persona_id]["system_prompt"]
        return None