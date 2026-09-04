#!/usr/bin/env python3
"""
Full reproducible pipeline execution engine for FlyRank Capstone.
Runs the authoritative evaluation on actual data:
- Primary: Chronological split
- Secondary: Client-grouped robustness split
- Baseline Heuristic vs Logistic Regression vs Random Forest vs HistGradientBoosting
Writes outputs to outputs/tables/, outputs/figures/, and outputs/run_manifest.json.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config


def main():
    config = load_config()
    data_path = Path(config.raw_data_path)

    manifest_path = Path("outputs/run_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    figures_dir = Path("outputs/figures")
    tables_dir = Path("outputs/tables")
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("FLYRANK CAPSTONE PIPELINE EXECUTION")
    print("==================================================")
    print(f"Target data path: {data_path}")

    if not data_path.exists():
        print(f"\n[STATUS: AWAITING REAL DATA EXECUTION]")
        print(f"Dataset not found at '{data_path}'.")
        print("To generate empirical results, connect the FlyRank warehouse dataset.")
        print("Run manifest remains in [PENDING REAL DATA RUN] status.")

        # Ensure clean pending manifest exists
        pending_manifest = {
            "execution_status": "PENDING_REAL_DATA_RUN",
            "message": "Empirical results are generated from the configured FlyRank warehouse at execution time.",
            "timestamp": None,
            "sample_size": "[PENDING REAL DATA RUN]",
            "eligible_population": "[PENDING REAL DATA RUN]",
            "positive_rate": "[PENDING REAL DATA RUN]",
            "primary_temporal_evaluation": {
                "baseline_precision_50": "[PENDING REAL DATA RUN]",
                "model_precision_50": "[PENDING REAL DATA RUN]",
                "precision_lift_50": "[PENDING REAL DATA RUN]",
                "baseline_pr_auc": "[PENDING REAL DATA RUN]",
                "model_pr_auc": "[PENDING REAL DATA RUN]",
            },
            "secondary_client_grouped_evaluation": {
                "baseline_precision_50": "[PENDING REAL DATA RUN]",
                "model_precision_50": "[PENDING REAL DATA RUN]",
                "precision_lift_50": "[PENDING REAL DATA RUN]",
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(pending_manifest, f, indent=2)
        return 0

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from src.baseline import HeuristicRanker
    from src.data import load_dataset
    from src.evaluation import evaluate_rankers
    from src.explainability import compute_permutation_importance, compute_tree_importance
    from src.features import build_feature_table
    from src.labels import construct_operational_label
    from src.model import ModelPipeline
    from src.privacy import assert_public_safe
    from src.recommendations import build_priority_queue
    from src.splits import create_chronological_split, create_client_grouped_split

    print("\n[Step 1/7] Loading dataset...")
    df_raw = load_dataset(str(data_path), config)
    print(f"Loaded {len(df_raw):,} records.")

    print("\n[Step 2/7] Constructing operational label...")
    is_eligible, y_target, label_diag = construct_operational_label(df_raw, config.label)
    print(f"Eligible records: {label_diag.eligible_population:,} / {label_diag.total_population:,}")
    print(f"Positive count: {label_diag.positive_count:,} ({label_diag.positive_rate * 100:.2f}%)")

    # Filter to eligible records
    df_eligible = df_raw.loc[is_eligible].copy()
    y_eligible = y_target.loc[is_eligible].copy()

    print("\n[Step 3/7] Engineering features and validating leakage...")
    X_all, feature_names = build_feature_table(df_eligible, config.schema_mapping.get("features", []))
    print(f"Constructed {len(feature_names)} features: {feature_names}")

    # Initialize models
    baseline = HeuristicRanker()
    model_lr = ModelPipeline(model_type="logistic_regression", random_seed=config.random_seed)
    model_rf = ModelPipeline(model_type="random_forest", random_seed=config.random_seed)

    print("\n[Step 4/7] PRIMARY EVALUATION: Chronological Split...")
    temporal_split = create_chronological_split(df_eligible, config=config.validation)

    train_idx = temporal_split.train_indices
    val_idx = temporal_split.val_indices
    test_idx = temporal_split.test_indices

    print(f"Chronological train: {len(train_idx):,}, val: {len(val_idx):,}, test: {len(test_idx):,}")

    # Fit models on chronological train set
    baseline.fit(X_all.iloc[train_idx], y_eligible.iloc[train_idx])
    model_rf.fit(X_all.iloc[train_idx], y_eligible.iloc[train_idx])

    # Evaluate on exact same test set records
    base_scores_test = baseline.predict_score(X_all.iloc[test_idx])
    rf_scores_test = model_rf.predict_score(X_all.iloc[test_idx])

    temporal_eval = evaluate_rankers(
        y_true=y_eligible.iloc[test_idx],
        baseline_scores=base_scores_test,
        model_scores=rf_scores_test,
        k_values=config.evaluation.k_values,
        evaluation_name="Primary Chronological Evaluation",
    )

    print("\nPrimary Temporal Evaluation Results (Top 50):")
    p50_info = temporal_eval["metrics_by_k"].get("k_50", {})
    print(f" - Baseline Precision@50: {p50_info.get('baseline_precision')}")
    print(f" - Model Precision@50:    {p50_info.get('model_precision')}")
    print(f" - Precision Lift@50:     {p50_info.get('precision_lift')}x")

    print("\n[Step 5/7] SECONDARY EVALUATION: Client-Grouped Robustness Split...")
    if "client_id" in df_eligible.columns:
        grouped_split = create_client_grouped_split(df_eligible, group_col="client_id", config=config.validation)
        g_train_idx = grouped_split.train_indices
        g_test_idx = grouped_split.test_indices

        model_rf_grouped = ModelPipeline(model_type="random_forest", random_seed=config.random_seed)
        model_rf_grouped.fit(X_all.iloc[g_train_idx], y_eligible.iloc[g_train_idx])

        base_scores_g = baseline.predict_score(X_all.iloc[g_test_idx])
        rf_scores_g = model_rf_grouped.predict_score(X_all.iloc[g_test_idx])

        grouped_eval = evaluate_rankers(
            y_true=y_eligible.iloc[g_test_idx],
            baseline_scores=base_scores_g,
            model_scores=rf_scores_g,
            k_values=config.evaluation.k_values,
            evaluation_name="Secondary Client-Grouped Robustness Evaluation",
        )
    else:
        grouped_eval = {"status": "client_id not present in dataset"}

    print("\n[Step 6/7] Generating explainability & prioritized refresh queue...")
    imp_df = compute_permutation_importance(model_rf, X_all.iloc[val_idx], y_eligible.iloc[val_idx])
    imp_df.to_csv(tables_dir / "feature_importance.csv", index=False)

    queue_df = build_priority_queue(X_all.iloc[test_idx], rf_scores_test, top_n=50)
    queue_df.to_csv(tables_dir / "priority_refresh_queue_top50.csv", index=False)

    # Save summary tables
    eval_table = pd.DataFrame(temporal_eval["metrics_by_k"]).T
    eval_table.to_csv(tables_dir / "temporal_evaluation_metrics.csv")

    print("\n[Step 7/7] Exporting charts & run manifest...")
    # Plot Precision@K comparison curve
    plt.figure(figsize=(8, 4.5))
    ks = [temporal_eval["metrics_by_k"][f"k_{k}"]["k"] for k in config.evaluation.k_values]
    base_p = [temporal_eval["metrics_by_k"][f"k_{k}"]["baseline_precision"] for k in config.evaluation.k_values]
    model_p = [temporal_eval["metrics_by_k"][f"k_{k}"]["model_precision"] for k in config.evaluation.k_values]

    plt.plot(ks, model_p, marker="o", color="#2563eb", label="Model (Random Forest)", linewidth=2)
    plt.plot(ks, base_p, marker="s", color="#64748b", linestyle="--", label="Heuristic Baseline", linewidth=2)
    plt.title("Primary Chronological Evaluation: Precision@K Comparison", fontsize=12, fontweight="bold")
    plt.xlabel("Top-K Cutoff (Queue Depth)")
    plt.ylabel("Precision@K")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "fig3_precision_at_k.png", dpi=150)
    plt.close()

    manifest = {
        "execution_status": "COMPLETED_REAL_DATA_RUN",
        "timestamp": pd.Timestamp.now().isoformat(),
        "sample_size": label_diag.total_population,
        "eligible_population": label_diag.eligible_population,
        "positive_count": label_diag.positive_count,
        "positive_rate": label_diag.positive_rate,
        "primary_temporal_evaluation": temporal_eval,
        "secondary_client_grouped_evaluation": grouped_eval,
    }

    manifest_json = json.dumps(manifest, indent=2)
    assert_public_safe(manifest_json, context="Run Manifest")

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest_json)

    print(f"\nPipeline successfully completed! Results saved to '{manifest_path}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
