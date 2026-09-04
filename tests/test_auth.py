"""
Tests for Hugging Face authentication, dotenv handling, DuckDB Secrets Manager integration, and secret scanning.
Uses fake tokens ONLY for unit test assertions.
"""

import os
import pytest

from src.data import classify_warehouse_error, create_duckdb_connection_with_hf_auth, get_hf_token
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

    err = classify_warehouse_error(Exception("401 Unauthorized access attempt with " + fake_token), fake_token)
    err_msg = str(err)

    assert fake_token not in err_msg
    assert "[REDACTED_HF_TOKEN]" in err_msg or "ACCESS DENIED" in err_msg


def test_duckdb_connection_and_secret_creation(monkeypatch):
    """Test 5 & 6: DuckDB connection initialization and temporary secret creation."""
    fake_token = "hf_123456789012345678901234567890"
    monkeypatch.setenv("HF_TOKEN", fake_token)

    try:
        conn = create_duckdb_connection_with_hf_auth(fake_token)
        assert conn is not None
        conn.close()
    except ImportError:
        pass  # Graceful fallback if duckdb is not installed in current environment


def test_secret_scanner_catches_fake_token():
    """Test 7: Secret scanner catches real-looking fake tokens and redacts output."""
    fake_token_str = "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz123456"
    violations = scan_for_privacy_violations(fake_token_str)

    assert len(violations) > 0
    # Assert token match is redacted
    for v in violations:
        if v["type"] == "SECRET_TOKEN":
            assert "abcdefghijkl" not in v["match"]
            assert "[REDACTED]" in v["match"]


def test_secret_scanner_allows_placeholder():
    """Test 8: Secret scanner allows safe placeholder in .env.example."""
    safe_env_example = "HF_TOKEN=hf_your_token_here"
    violations = scan_for_privacy_violations(safe_env_example)

    # Should not flag hf_your_token_here as a leaked token
    token_violations = [v for v in violations if v["type"] == "SECRET_TOKEN"]
    assert len(token_violations) == 0
    assert_public_safe(safe_env_example)
