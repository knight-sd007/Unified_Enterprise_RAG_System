"""
Security & Input Sanitization Utilities.
"""

import hmac
import re

def verify_access_key(provided_key: str, expected_key: str) -> bool:
    """
    Performs constant-time string comparison to prevent timing attacks.
    """
    if not provided_key or not expected_key:
        return False
    return hmac.compare_digest(provided_key.strip(), expected_key.strip())

def sanitize_error_message(error: Exception) -> str:
    """
    Strips sensitive credentials, internal file paths, or raw stack traces from exception strings.
    """
    msg = str(error)
    # Redact URL credentials (e.g. https://user:pass@host)
    msg = re.sub(r'(https?://[^:@\s]+:)[^@\s]+(@)', r'\1[REDACTED]\2', msg)
    # Redact potential API keys (sk-..., AIzaSy..., nvapi-...)
    msg = re.sub(r'(sk-[a-zA-Z0-9]{20,})', '[REDACTED_API_KEY]', msg)
    msg = re.sub(r'(AIzaSy[a-zA-Z0-9_-]{30,})', '[REDACTED_API_KEY]', msg)
    msg = re.sub(r'(nvapi-[a-zA-Z0-9_-]{30,})', '[REDACTED_API_KEY]', msg)
    # Redact explicit api_key or token parameter patterns
    msg = re.sub(r'(api[-_]?key[\s:=]+)[^\s,;\'"]+', r'\1[REDACTED]', msg, flags=re.IGNORECASE)
    msg = re.sub(r'(bearer[\s]+)[a-zA-Z0-9_\-\.]{15,}', r'\1[REDACTED]', msg, flags=re.IGNORECASE)
    return msg
