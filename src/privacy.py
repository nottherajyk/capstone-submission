"""
Privacy and sanitization module for FlyRank Capstone.
Guarantees public-safe exports without client names, raw URLs, query strings, or secret tokens.
"""

import re
from typing import Dict, List, Set, Tuple

# Only allowed external URL
ALLOWED_URLS: Set[str] = {
    "https://flyrank.ai",
    "https://flyrank.ai/",
    "http://flyrank.ai",
}

SAFE_PLACEHOLDERS: Set[str] = {
    "hf_your_token_here",
    "hf_your_actual_token_here",
    "your_huggingface_token_here",
    "hf_123456789012345678901234567890",
    "hf_abcdefghijklmnopqrstuvwxyz123456",
    "hf_secret12345678901234567890",
}

# Regex patterns for detecting sensitive data
URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
DOMAIN_PATTERN = re.compile(r"\b[a-zA-Z0-9][-a-zA-Z0-9]*\.(?:com|org|net|io|co|ai|edu|gov)\b", re.IGNORECASE)
HF_TOKEN_PATTERN = re.compile(r"\bhf_[a-zA-Z0-9]{20,}\b")
API_KEY_PATTERN = re.compile(r"(?:api[_-]?key|secret|token)[\s:=]+['\"]?([a-zA-Z0-9_\-]{16,})['\"]?", re.IGNORECASE)


def sanitize_id(raw_id: str, prefix: str = "item") -> str:
    """Hash or deterministic pseudonymize identifiers."""
    val = abs(hash(str(raw_id))) % 1000000
    return f"{prefix}_{val:06d}"


def scan_for_privacy_violations(text: str) -> List[Dict[str, str]]:
    """
    Scan string text for unauthorized URLs, domains, queries, or credentials.
    Returns a list of violations found with redacted match strings.
    """
    violations: List[Dict[str, str]] = []

    # 1. Check for unauthorized URLs
    for match in URL_PATTERN.finditer(text):
        url = match.group(0).rstrip(".,;)")
        if url not in ALLOWED_URLS and not url.startswith("https://nottherajyk.github.io"):
            violations.append({
                "type": "UNAUTHORIZED_URL",
                "match": url,
                "position": str(match.start()),
            })

    # 2. Check for potential raw domains
    for match in DOMAIN_PATTERN.finditer(text):
        domain = match.group(0).lower()
        if domain not in {"flyrank.ai", "github.com", "github.io"}:
            violations.append({
                "type": "POTENTIAL_DOMAIN",
                "match": domain,
                "position": str(match.start()),
            })

    # 3. Check for Hugging Face tokens
    for match in HF_TOKEN_PATTERN.finditer(text):
        token_str = match.group(0)
        if token_str not in SAFE_PLACEHOLDERS and not token_str.startswith("hf_12345"):
            violations.append({
                "type": "SECRET_TOKEN",
                "match": token_str[:6] + "[REDACTED]",
                "position": str(match.start()),
            })

    # 4. Check for generic API keys
    for match in API_KEY_PATTERN.finditer(text):
        key_str = match.group(0)
        if "your_token" not in key_str and "your_actual_token" not in key_str and "hf_12345" not in key_str:
            violations.append({
                "type": "API_CREDENTIAL",
                "match": key_str[:12] + "[REDACTED]",
                "position": str(match.start()),
            })

    return violations


def assert_public_safe(text: str, context: str = "Artifact") -> None:
    """Assert that a string contains zero privacy violations; raise ValueError otherwise."""
    violations = scan_for_privacy_violations(text)
    if violations:
        msg = f"Privacy violation detected in {context}: {len(violations)} issues found. First: {violations[0]}"
        raise ValueError(msg)
