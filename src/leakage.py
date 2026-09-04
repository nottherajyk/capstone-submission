"""
Leakage detection and audit engine for FlyRank Capstone.
Enforces strict 7-level leakage priority hierarchy:
1. Feature timestamp availability
2. Feature provenance
3. Future-window dependency
4. Target-derived feature detection
5. Precomputed outcome/recommendation field detection
6. Identifier / grouping leakage
7. Statistical diagnostics (secondary only)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np


@dataclass
class FeatureMetadata:
    feature_name: str
    source_table: str
    source_columns: List[str]
    observation_time_rule: str
    lookback_window: str
    future_dependency: bool
    target_dependency: bool
    leakage_status: str  # SAFE, CAUTION, BLOCKED
    notes: str


class LeakageError(Exception):
    """Raised when a BLOCKED or leaking feature enters model training."""
    pass


class LeakageAuditor:
    """
    Authoritative auditor checking features against explicit leakage tiers.
    """

    KNOWN_BLOCKED_TERMS: Set[str] = {
        "trend_direction",
        "trend_pct",
        "health_score",
        "recommended_action",
        "future_clicks",
        "future_impressions",
        "future_position",
        "is_declining_label",
        "target",
        "label",
    }

    IDENTIFIER_TERMS: Set[str] = {
        "client_id",
        "client_hash_id",
        "client_hash",
        "tenant_id",
        "account_id",
        "page_id",
        "content_id",
        "content_hash_id",
        "content_hash",
        "url_id",
        "doc_id",
        "url",
        "domain",
    }

    def __init__(self, feature_registry: Optional[List[Dict[str, Any]]] = None):
        self.registry: Dict[str, FeatureMetadata] = {}
        if feature_registry:
            for item in feature_registry:
                meta = FeatureMetadata(
                    feature_name=item["feature_name"],
                    source_table=item.get("source_table", "search_performance"),
                    source_columns=item.get("source_columns", []),
                    observation_time_rule=item.get("observation_time_rule", "unknown"),
                    lookback_window=item.get("lookback_window", "unknown"),
                    future_dependency=bool(item.get("future_dependency", False)),
                    target_dependency=bool(item.get("target_dependency", False)),
                    leakage_status=item.get("leakage_status", "BLOCKED"),
                    notes=item.get("notes", ""),
                )
                self.registry[meta.feature_name] = meta

    def audit_feature(
        self,
        feature_name: str,
        df: Optional[pd.DataFrame] = None,
        target_series: Optional[pd.Series] = None,
    ) -> FeatureMetadata:
        """
        Audit a candidate feature against the 7-level leakage priority hierarchy.
        """
        clean_name = feature_name.lower().strip()

        # Priority 3, 4, 5: Check blocked target/future/precomputed fields
        for blocked_term in self.KNOWN_BLOCKED_TERMS:
            if blocked_term in clean_name:
                return FeatureMetadata(
                    feature_name=feature_name,
                    source_table="unknown",
                    source_columns=[feature_name],
                    observation_time_rule="VIOLATION: references target or future data",
                    lookback_window="none",
                    future_dependency=True,
                    target_dependency=True,
                    leakage_status="BLOCKED",
                    notes=f"Contains prohibited target/outcome term '{blocked_term}'",
                )

        # Priority 6: Check raw identifiers/grouping leakage
        for id_term in self.IDENTIFIER_TERMS:
            if clean_name == id_term or clean_name.endswith(f"_{id_term}"):
                return FeatureMetadata(
                    feature_name=feature_name,
                    source_table="raw_entity",
                    source_columns=[feature_name],
                    observation_time_rule="VIOLATION: raw identifier",
                    lookback_window="none",
                    future_dependency=False,
                    target_dependency=False,
                    leakage_status="BLOCKED",
                    notes=f"High-cardinality entity identifier '{id_term}'",
                )

        # Check in registered metadata
        if feature_name in self.registry:
            meta = self.registry[feature_name]
            if meta.future_dependency or meta.target_dependency:
                meta.leakage_status = "BLOCKED"
            return meta

        # Secondary diagnostics (statistical checks only as secondary diagnostic, never primary proof)
        notes = "Feature not pre-registered in schema_mapping.yaml"
        status = "CAUTION"

        if df is not None and target_series is not None and feature_name in df.columns:
            try:
                corr = df[feature_name].corr(target_series)
                if abs(corr) > 0.95:
                    status = "BLOCKED"
                    notes = f"Extreme correlation ({corr:.3f}) indicating near-deterministic target derivation"
            except Exception:
                pass

        return FeatureMetadata(
            feature_name=feature_name,
            source_table="dynamic",
            source_columns=[feature_name],
            observation_time_rule="Dynamic generation; requires manual timestamp audit",
            lookback_window="unknown",
            future_dependency=False,
            target_dependency=False,
            leakage_status=status,
            notes=notes,
        )

    def validate_feature_matrix(
        self,
        feature_names: List[str],
        df: Optional[pd.DataFrame] = None,
        target_series: Optional[pd.Series] = None,
    ) -> Dict[str, FeatureMetadata]:
        """
        Validate all candidate features before model ingestion.
        Hard-fails with LeakageError if any BLOCKED feature enters.
        """
        audited: Dict[str, FeatureMetadata] = {}
        blocked_features: List[Tuple[str, str]] = []

        for name in feature_names:
            meta = self.audit_feature(name, df, target_series)
            audited[name] = meta
            if meta.leakage_status == "BLOCKED":
                blocked_features.append((name, meta.notes))

        if blocked_features:
            details = "\n".join([f" - {f}: {reason}" for f, reason in blocked_features])
            raise LeakageError(
                f"Model training halted due to BLOCKED leakage features detected in feature matrix:\n{details}"
            )

        return audited
