#!/usr/bin/env python3
"""
Research paper renderer for FlyRank Capstone.
Reads outputs/run_manifest.json and dynamically injects live empirical evidence into paper/index.html.
If run_manifest.json is absent or in PENDING_REAL_DATA_RUN state, leaves all metrics strictly
as [PENDING REAL DATA RUN].
"""

import json
import re
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.privacy import assert_public_safe


def render_paper():
    manifest_path = Path("outputs/run_manifest.json")
    paper_path = Path("paper/index.html")

    if not manifest_path.exists():
        print("Manifest not found. Paper remains in [PENDING REAL DATA RUN] status.")
        return 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    status = manifest.get("execution_status", "PENDING_REAL_DATA_RUN")
    print(f"Manifest status: {status}")

    if not paper_path.exists():
        print(f"Paper index file '{paper_path}' not found.")
        return 1

    with open(paper_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    if status == "COMPLETED_REAL_DATA_RUN":
        print("Injecting actual warehouse empirical results into paper/index.html...")
        temp_eval = manifest.get("primary_temporal_evaluation", {})
        m50 = temp_eval.get("metrics_by_k", {}).get("k_50", {})

        p_base = m50.get("baseline_precision", "[PENDING REAL DATA RUN]")
        p_model = m50.get("model_precision", "[PENDING REAL DATA RUN]")
        lift = m50.get("precision_lift", "[PENDING REAL DATA RUN]")
        sample_size = manifest.get("sample_size", "[PENDING REAL DATA RUN]")
        pos_rate = manifest.get("positive_rate", "[PENDING REAL DATA RUN]")

        # Dynamic replacement of pre-run tags
        html_content = html_content.replace(
            "id=\"val-primary-base-p50\">[PENDING REAL DATA RUN]",
            f"id=\"val-primary-base-p50\">{p_base}"
        )
        html_content = html_content.replace(
            "id=\"val-primary-model-p50\">[PENDING REAL DATA RUN]",
            f"id=\"val-primary-model-p50\">{p_model}"
        )
        html_content = html_content.replace(
            "id=\"val-primary-lift-p50\">[PENDING REAL DATA RUN]",
            f"id=\"val-primary-lift-p50\">{lift}x"
        )
        html_content = html_content.replace(
            "id=\"val-sample-size\">[PENDING REAL DATA RUN]",
            f"id=\"val-sample-size\">{sample_size:,}" if isinstance(sample_size, int) else f"id=\"val-sample-size\">{sample_size}"
        )

        # Assert public safety before writing
        assert_public_safe(html_content, context="Rendered Research Paper")

        with open(paper_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print("Paper successfully updated with verified empirical metrics.")
    else:
        print("Paper confirmed in authoritative pre-run state with [PENDING REAL DATA RUN] placeholders.")

    return 0


if __name__ == "__main__":
    sys.exit(render_paper())
