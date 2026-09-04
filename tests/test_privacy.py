"""
Tests for privacy scanning and secret token detection.
Verifies blocking of external domains, credentials, and tokens while allowing flyrank.ai.
"""

import pytest

from src.privacy import assert_public_safe, scan_for_privacy_violations


def test_flyrank_ai_url_is_permitted():
    safe_text = "This research was developed using the https://flyrank.ai platform data."
    violations = scan_for_privacy_violations(safe_text)
    assert len(violations) == 0
    # Should not raise
    assert_public_safe(safe_text)


def test_unauthorized_urls_and_domains_detected():
    bad_text = "Data downloaded from https://private-client-store.com/api/v1/export"
    violations = scan_for_privacy_violations(bad_text)
    assert len(violations) > 0
    with pytest.raises(ValueError) as exc:
        assert_public_safe(bad_text)
    assert "Privacy violation" in str(exc.value)


def test_secret_tokens_detected():
    # Hugging Face token pattern: hf_ followed by >=34 chars
    token_text = "Exported with HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz1234567890"
    violations = scan_for_privacy_violations(token_text)
    assert any(v["type"] == "SECRET_TOKEN" for v in violations)

    api_key_text = "Configured with api_key='sk_live_12345678901234567890'"
    violations2 = scan_for_privacy_violations(api_key_text)
    assert any(v["type"] == "API_CREDENTIAL" for v in violations2)
