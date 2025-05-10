"""
Translation utilities for the NFRS Assistant.
"""
from google.cloud import translate_v2 as translate
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

def translate_text(text, target_language, source_language=None):
    """
    Translate text between languages using Google Cloud Translation API.

    Args:
        text (str): The text to translate
        target_language (str): The target language code (e.g., 'en', 'ne')
        source_language (str, optional): The source language code. If None, it will be auto-detected.

    Returns:
        str: The translated text
    """
    if not text:
        return text

    try:
        client = translate.Client()

        # If source language not provided, determine based on target
        if not source_language:
            if target_language == 'en':
                source_language = 'ne'
            elif target_language == 'ne':
                source_language = 'en'

        result = client.translate(
            text,
            target_language=target_language,
            source_language=source_language
        )

        return result['translatedText']
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text  # Return original text if translation fails


def is_nepali_text(text):
    """
    Detect if text is primarily in Nepali.

    Args:
        text (str): The text to analyze

    Returns:
        bool: True if the text appears to be in Nepali, False otherwise
    """
    # Check if text contains Devanagari Unicode characters
    # Devanagari Unicode range: 0900-097F
    nepali_char_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    total_chars = len(text.strip())

    if total_chars == 0:
        return False

    # Consider it Nepali if at least 30% of characters are Devanagari
    return (nepali_char_count / total_chars) > 0.3


def detect_language(text):
    """
    Detect the language of the given text.

    Args:
        text (str): The text to analyze

    Returns:
        str: The detected language code ('en' or 'ne')
    """
    if is_nepali_text(text):
        return 'ne'
    return 'en'