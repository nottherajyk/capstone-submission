"""
Tests for Hugging Face authentication, dotenv handling, and secret scanning.
Uses fake tokens ONLY for unit test assertions.
"""

import os
import pytest

from src.data import get_hf_token
from src.privacy import assert_public_safe, scan_for_privacy_violations


def test_missing_hf_token_raises_clear_error(monkeypatch):
    """Test 1: Missing HF_TOKEN produces a clear actionable error."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError) as exc_info:
        get_hf_token(raise_error=True)

    err_msg = str(exc_info.value)
    assert "HF_TOKEN is not configured" in err_msg
    assert "local .env file" in err_msg


def test_env_var_read_correctly(monkeypatch):
    """Test 2 & 4: Environment variable HF_TOKEN is read correctly."""
    fake_token = "hf_123456789012345678901234567890"
    monkeypatch.setenv("HF_TOKEN", fake_token)

    retrieved = get_hf_token(raise_error=True)
    assert retrieved == fake_token


def test_token_never_in_errors(monkeypatch):
    """Test 3: Token is never returned in exception messages."""
    fake_token = "hf_secret12345678901234567890"
    monkeypatch.setenv("HF_TOKEN", fake_token)

    try:
        from src.data import load_dataset
        # Intentional invalid path to trigger error path
        load_dataset("non_existent_file.csv")
    except Exception as e:
        err_msg = str(e)
        assert fake_token not in err_msg


def test_secret_scanner_catches_fake_token():
    """Test 5: Secret scanner catches real-looking fake tokens and redacts output."""
    fake_token_str = "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz123456"
    violations = scan_for_privacy_violations(fake_token_str)

    assert len(violations) > 0
    # Assert token match is redacted
    for v in violations:
        if v["type"] == "SECRET_TOKEN":
            assert "abcdefghijkl" not in v["match"]
            assert "[REDACTED]" in v["match"]


def test_secret_scanner_allows_placeholder():
    """Test 6: Secret scanner allows safe placeholder in .env.example."""
    safe_env_example = "HF_TOKEN=hf_your_token_here"
    violations = scan_for_privacy_violations(safe_env_example)

    # Should not flag hf_your_token_here as a leaked token
    token_violations = [v for v in violations if v["type"] == "SECRET_TOKEN"]
    assert len(token_violations) == 0
    assert_public_safe(safe_env_example)
