"""
Translation utilities for the NFRS Assistant.
"""
from google.cloud import translate_v2 as translate
from django.conf import settings
import logging
import os
import requests
from typing import Dict, Any

logger = logging.getLogger(__name__)

def translate_text(text, target_language, source_language=None):
    """
    Translate text between languages using Google Cloud Translation API.
    Falls back to a basic translation method if the API is not available.

    Args:
        text (str): The text to translate
        target_language (str): The target language code (e.g., 'en', 'ne')
        source_language (str, optional): The source language code. If None, it will be auto-detected.

    Returns:
        str: The translated text
    """
    if not text:
        return text

    # If source language not provided, determine based on target
    if not source_language:
        if target_language == 'en':
            source_language = 'ne'
        elif target_language == 'ne':
            source_language = 'en'

    try:
        # Try Google Cloud Translation first
        try:
            # Set credentials path
            google_credentials_path = os.path.join(settings.BASE_DIR, 'google-credentials.json')
            if os.path.exists(google_credentials_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_path

            client = translate.Client()
            result = client.translate(
                text,
                target_language=target_language,
                source_language=source_language
            )
            return result['translatedText']
        except Exception as google_error:
            logger.warning(f"Google Translation failed: {google_error}. Trying fallback method...")

            # Fallback to basic translation
            translated = fallback_translation(text, source_language, target_language)
            if translated:
                return translated

            # If all translation methods fail, log the error and return original text
            raise Exception(f"All translation methods failed. Original error: {google_error}")

    except Exception as e:
        logger.error(f"Translation error: {e}")
        return f"{text} (Translation service unavailable)"  # Return original text with note if translation fails


def fallback_translation(text: str, source_lang: str, target_lang: str) -> str:
    """
    A simple fallback translation mechanism using a basic dictionary for common
    Nepali-English phrases. This is very limited but provides some functionality
    when the Google API is unavailable.

    Args:
        text (str): Text to translate
        source_lang (str): Source language code ('en' or 'ne')
        target_lang (str): Target language code ('en' or 'ne')

    Returns:
        str: Translated text or empty string if translation failed
    """
    # Very basic dictionary of common phrases
    en_to_ne = {
        "hello": "नमस्ते",
        "thank you": "धन्यवाद",
        "yes": "हो",
        "no": "होइन",
        "forest": "वन",
        "tree": "रूख",
        "what is your name": "तपाईंको नाम के हो",
        "my name is": "मेरो नाम हो",
        "how are you": "तपाईं कस्तो हुनुहुन्छ",
        "i am fine": "म ठिक छु",
        "good morning": "शुभ प्रभात",
        "good night": "शुभ रात्री"
    }

    # Flip dictionary for ne->en
    ne_to_en = {v: k for k, v in en_to_ne.items()}

    # Choose the right dictionary based on direction
    translation_dict = en_to_ne if source_lang == 'en' and target_lang == 'ne' else ne_to_en

    # Simple word/phrase replacement
    # This is extremely basic and won't handle grammar or complex sentences properly
    lower_text = text.lower()
    for src, tgt in translation_dict.items():
        if src.lower() in lower_text:
            return text.replace(src, tgt)

    # If no match found, return empty to trigger warning about limited translation
    return ""


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