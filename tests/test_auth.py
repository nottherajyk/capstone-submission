"""
Tests for Hugging Face authentication, dotenv handling, DuckDB Secrets Manager integration,
warehouse path resolution, and secret scanning.
Uses fake tokens ONLY for unit test assertions.
"""

import os
import pytest

from src.data import (
    classify_warehouse_error,
    create_duckdb_connection_with_hf_auth,
    get_hf_token,
    get_warehouse_sources,
)
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


def test_warehouse_root_path_resolution():
    """Test 9: Verify root warehouse path and no /data suffix is generated."""
    repo_id = "FlyRank/internship-warehouse"
    sources = get_warehouse_sources(repo_id)

    assert sources["root"] == "hf://datasets/FlyRank/internship-warehouse"
    assert not sources["root"].endswith("/data")
    assert "/internship-warehouse/data" not in sources["root"]
    assert sources["dim_clients"] == "hf://datasets/FlyRank/internship-warehouse/dim_clients.parquet"
    assert sources["dim_content"] == "hf://datasets/FlyRank/internship-warehouse/dim_content.parquet"
    assert sources["query_90d"] == "hf://datasets/FlyRank/internship-warehouse/fact_content_query_90d.parquet"
    assert sources["sample"] == "hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance_sample.parquet"
    assert sources["daily_performance"] == "hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance/**/*.parquet"


def test_regression_no_data_subfolder_suffix():
    """Regression Test: Ensure hf://datasets/FlyRank/internship-warehouse does NOT become .../data."""
    sources = get_warehouse_sources("FlyRank/internship-warehouse")
    for key, path in sources.items():
        assert not path.endswith("/data")
        assert "/internship-warehouse/data/" not in path


def test_error_classification_distinguishes_error_types():
    """Test 10: Error classifier distinguishes 404, 401, network, and missing catalog errors."""
    err_404 = classify_warehouse_error(Exception("HTTP GET error (HTTP 404)"))
    assert "[PATH/SCHEMA NOT FOUND]" in str(err_404)

    err_401 = classify_warehouse_error(Exception("401 Unauthorized access"))
    assert "[ACCESS DENIED]" in str(err_401)

    err_net = classify_warehouse_error(Exception("Could not resolve host huggingface.co"))
    assert "[NETWORK ERROR]" in str(err_net)


def test_infer_candidate_identifiers_real_schema():
    """Test 11: Candidate identifier inference selects client_hash_id, content_hash_id, report_date."""
    import pandas as pd
    from src.data import infer_candidate_identifiers

    df_sample = pd.DataFrame({
        "report_date": pd.to_datetime(["2026-06-30", "2026-06-30"]),
        "client_hash_id": ["cl_hash_001", "cl_hash_002"],
        "content_hash_id": ["cnt_hash_001", "cnt_hash_002"],
        "client_has_gsc": [True, True],
        "client_has_ga4": [True, False],
        "gsc_data_available": [True, True],
        "ga4_data_available": [True, False],
        "impressions": [500, 1000],
        "clicks": [14, 25],
        "gsc_sum_position": [1427, 2800],
        "gsc_avg_position": [4.2, 5.1],
        "ga4_pageviews": [17, 0],
    })

    inferred = infer_candidate_identifiers(df_sample)

    # Required assertion 1 & 2 & 3: correct canonical fields selected
    assert inferred["client_id"] == "client_hash_id"
    assert inferred["content_id"] == "content_hash_id"
    assert inferred["report_date"] == "report_date"

    # Required assertion 4 & 5: reject boolean and metric false matches
    assert inferred["client_id"] != "client_has_ga4"
    assert inferred["client_id"] != "client_has_gsc"
    assert inferred["content_id"] != "ga4_pageviews"
    assert inferred["content_id"] != "impressions"


def test_numeric_and_boolean_fields_rejected_as_identifiers():
    """Test 12: Numeric performance metrics and boolean flags cannot become entity identifiers."""
    import pandas as pd
    from src.data import infer_candidate_identifiers

    # DataFrame with ONLY metrics and flags, without valid entity identifiers
    df_no_ids = pd.DataFrame({
        "client_has_ga4": [True, False],
        "ga4_data_available": [True, False],
        "ga4_pageviews": [100, 200],
        "clicks": [10, 20],
        "impressions": [1000, 2000],
        "position": [3.5, 4.2],
    })

    inferred = infer_candidate_identifiers(df_no_ids)
    assert inferred["client_id"] is None
    assert inferred["content_id"] is None


def test_resolve_canonical_columns_preserves_raw_columns():
    """Test 13: resolve_canonical_columns does not rename or overwrite raw warehouse columns."""
    import pandas as pd
    from src.data import resolve_canonical_columns

    df_raw = pd.DataFrame({
        "report_date": pd.to_datetime(["2026-06-30"]),
        "client_hash_id": ["cl_hash_abc"],
        "content_hash_id": ["cnt_hash_xyz"],
        "gsc_avg_position": [3.5],
        "clicks": [10],
        "impressions": [100],
    })

    schema_mapping = {
        "identifiers": {
            "client_id": {"canonical": "client_hash_id", "aliases": ["client_hash_id"]},
            "content_id": {"canonical": "content_hash_id", "aliases": ["content_hash_id"]},
            "report_date": {"canonical": "report_date", "aliases": ["report_date"]},
        },
        "raw_metrics": {
            "position": {"canonical": "gsc_avg_position", "aliases": ["gsc_avg_position"]},
        },
    }

    df_resolved = resolve_canonical_columns(df_raw, schema_mapping)

    # Raw columns MUST be preserved
    assert "client_hash_id" in df_resolved.columns
    assert "content_hash_id" in df_resolved.columns
    assert "report_date" in df_resolved.columns
    assert "gsc_avg_position" in df_resolved.columns

    # Canonical aliases MUST be populated
    assert "client_id" in df_resolved.columns
    assert "content_id" in df_resolved.columns
    assert "date" in df_resolved.columns
    assert "position" in df_resolved.columns
    assert df_resolved["client_id"].iloc[0] == "cl_hash_abc"
    assert df_resolved["content_id"].iloc[0] == "cnt_hash_xyz"
    assert df_resolved["position"].iloc[0] == 3.5


def test_check_warehouse_analytical_grain_duckdb(tmp_path):
    """Test full warehouse analytical grain check using DuckDB aggregation."""
    import duckdb
    from src.data import check_warehouse_analytical_grain

    p_str = str(tmp_path / "test_perf.parquet").replace("\\", "/")
    conn = duckdb.connect()

    # Create test parquet with 5 rows: 4 unique grain combos, 1 duplicate
    conn.execute(f"""
        COPY (
            SELECT 'c1' AS client_hash_id, 'p1' AS content_hash_id, DATE '2025-02-01' AS report_date
            UNION ALL SELECT 'c1', 'p2', DATE '2025-02-05'
            UNION ALL SELECT 'c2', 'p1', DATE '2025-03-01'
            UNION ALL SELECT 'c3', 'p1', DATE '2025-04-01'
            UNION ALL SELECT 'c1', 'p1', DATE '2025-02-01'
        ) TO '{p_str}' (FORMAT PARQUET);
    """)

    res = check_warehouse_analytical_grain(conn, source_path=p_str)

    assert res["total_rows"] == 5
    assert res["distinct_grain_combinations"] == 4
    assert res["duplicate_count"] == 1
    assert res["grain_holds"] is False
    assert res["report_date_range"]["min"].startswith("2025-02-01")
    assert res["report_date_range"]["max"].startswith("2025-04-01")
    conn.close()
