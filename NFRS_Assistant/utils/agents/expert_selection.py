"""
Expert selection module for the NFRS Assistant.

This module provides functionality to select relevant financial experts
based on the content and context of a user query.
"""

import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Define expert personas with specializations
EXPERT_PERSONAS = [
    {
        "name": "Dr. Ramesh Sharma",
        "title": "Senior NFRS Specialist",
        "description": "Expert in Nepal Financial Reporting Standards with 15 years of experience.",
        "specialties": ["NFRS", "financial statements", "compliance", "audit", "nepal standards"],
    },
    {
        "name": "Maya Poudel",
        "title": "IFRS Compliance Officer",
        "description": "Specializes in international financial standards and compliance.",
        "specialties": ["IFRS", "international standards", "compliance", "global reporting"],
    },
    {
        "name": "Anand Joshi",
        "title": "Financial Reporting Analyst",
        "description": "Expert in analyzing financial statements and reporting requirements.",
        "specialties": ["financial analysis", "financial statements", "reporting", "ratios"],
    },
    {
        "name": "Dr. Prabha Singh",
        "title": "Tax Accounting Specialist",
        "description": "Expert in tax implications of accounting standards.",
        "specialties": ["taxation", "tax accounting", "deferred tax", "tax planning"],
    },
    {
        "name": "Bikash Thapa",
        "title": "Banking Sector Specialist",
        "description": "Specializes in NFRS application in the banking and finance sectors.",
        "specialties": ["banking", "financial institutions", "NRB directives", "credit loss"],
    },
    {
        "name": "Sarita Ghimire",
        "title": "Fixed Assets & Depreciation Expert",
        "description": "Expert in accounting for property, plant, equipment and depreciation.",
        "specialties": ["fixed assets", "depreciation", "PPE", "asset valuation", "impairment"],
    },
    {
        "name": "Dr. Kamal Neupane",
        "title": "Revenue Recognition Specialist",
        "description": "Expert in revenue recognition principles and practices under NFRS/IFRS.",
        "specialties": ["revenue recognition", "contracts", "NFRS 15", "IFRS 15", "performance obligations"],
    }
]

def extract_query_topics(query: str) -> List[str]:
    """
    Extract relevant financial topics from a user query.

    Args:
        query: The user's question or request

    Returns:
        List of identified topics
    """
    # Convert to lowercase for better matching
    query_lower = query.lower()

    # Define common financial reporting topics
    topics = [
        "financial statements", "balance sheet", "income statement", "cash flow",
        "depreciation", "revenue recognition", "tax", "taxation", "assets", "liabilities",
        "equity", "audit", "compliance", "ifrs", "nfrs", "disclosures", "notes",
        "recognition", "measurement", "reporting", "accounting policies", "estimates",
        "fair value", "impairment", "consolidation", "banking", "financial instruments",
        "leases", "inventory", "provisions", "contingent", "intangible", "deferred tax",
    ]

    # Find topics in the query
    found_topics = []
    for topic in topics:
        if topic in query_lower:
            found_topics.append(topic)

    # If no specific topics found, return some general topics
    if not found_topics:
        found_topics = ["financial reporting", "accounting standards"]

    return found_topics

def select_expert_personas(topics: List[str], num_experts: int = 3) -> List[Dict[str, Any]]:
    """
    Select the most relevant expert personas based on topics.

    Args:
        topics: List of financial topics
        num_experts: Number of experts to select

    Returns:
        List of selected expert personas
    """
    # Score each expert based on topic relevance
    scored_experts = []
    for expert in EXPERT_PERSONAS:
        score = 0
        for topic in topics:
            for specialty in expert["specialties"]:
                if topic in specialty.lower() or specialty.lower() in topic:
                    score += 1

        scored_experts.append((score, expert))

    # Sort by score (descending)
    scored_experts.sort(reverse=True, key=lambda x: x[0])

    # Get the top N experts
    selected_experts = [expert for _, expert in scored_experts[:num_experts]]

    # Return at least one expert even if none matched
    if not selected_experts:
        selected_experts = [EXPERT_PERSONAS[0]]

    # Remove the specialties field from the output
    for expert in selected_experts:
        expert_copy = expert.copy()
        if "specialties" in expert_copy:
            del expert_copy["specialties"]

    # Return experts without the specialties field
    return [
        {k: v for k, v in expert.items() if k != "specialties"}
        for expert in selected_experts
    ]

def estimate_discussion_rounds(query: str) -> int:
    """
    Estimate the number of discussion rounds needed based on query complexity.

    Args:
        query: The user's question

    Returns:
        Estimated number of rounds (1-3)
    """
    # Simple heuristic based on query length and complexity indicators
    complexity = 1

    # Length-based complexity
    if len(query) > 150:
        complexity += 1

    # Keyword-based complexity
    complex_indicators = [
        "compare", "difference", "versus", "pros and cons", "advantages",
        "disadvantages", "implications", "impact", "analyze", "recommend",
        "should we", "how would", "explain in detail", "comprehensive"
    ]

    for indicator in complex_indicators:
        if indicator in query.lower():
            complexity += 1
            break

    # Cap at 3 rounds
    return min(complexity, 3)

def select_experts_for_query(query: str, num_experts: int = 3) -> List[Dict[str, str]]:
    """
    Main function to select experts based on a user query.

    Args:
        query: The user's question
        num_experts: Number of experts to select (default: 3)

    Returns:
        List of expert data dictionaries
    """
    try:
        # Extract topics from the query
        topics = extract_query_topics(query)
        logger.info(f"Extracted topics: {topics}")

        # Select relevant experts
        selected_experts = select_expert_personas(topics, num_experts)
        logger.info(f"Selected {len(selected_experts)} experts for the query")

        return selected_experts
    except Exception as e:
        logger.error(f"Error in expert selection: {e}")
        # Return default experts on error
        default_experts = EXPERT_PERSONAS[:min(num_experts, len(EXPERT_PERSONAS))]
        return [
            {k: v for k, v in expert.items() if k != "specialties"}
            for expert in default_experts
        ]