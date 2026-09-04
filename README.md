# FlyRank Applied Search Intelligence — Capstone Repository

[![Validation Checks](https://github.com/nottherajyk/capstone-submission/actions/workflows/deploy-paper.yml/badge.svg)](https://github.com/nottherajyk/capstone-submission/actions)

> **Research Question:**  
> *"Can historical search-performance signals identify content that is likely to experience meaningful future deterioration, enabling a more effective refresh-prioritization queue?"*

This repository houses the complete, reproducible machine learning system and research paper for content refresh prioritization in organic search.

**Core Methodological Principle: Zero Hard-Coded Empirical Results**  
Empirical results are generated from the configured FlyRank warehouse at execution time. Prior to running against the connected dataset, all empirical metrics default strictly to `[PENDING REAL DATA RUN]`.

---

## 1. System Architecture

```text
flyrank-capstone/
├── work/                         # 7 reproducible research notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_label_validation.ipynb
│   ├── 05_baseline_and_model.ipynb
│   ├── 06_evaluation_and_error_analysis.ipynb
│   └── 07_capstone.ipynb
│
├── src/                          # Modular production logic
│   ├── config.py                 # Configuration & schema mapping loader
│   ├── data.py                   # Dataset discovery & column resolver
│   ├── features.py               # Zero-division safe feature engineering
│   ├── labels.py                 # Operational future-window deterioration target
│   ├── leakage.py                # 7-tier priority leakage prevention engine
│   ├── splits.py                 # Primary chronological & secondary grouped splits
│   ├── baseline.py               # Transparent heuristic ranker
│   ├── model.py                  # Logistic Regression & Random Forest pipelines
│   ├── evaluation.py             # Same-population Precision@K & lift evaluator
│   ├── recommendations.py        # Top-50 queue generator with reason codes
│   ├── explainability.py         # Permutation & tree feature importance
│   └── privacy.py                # PII & credential scanner
│
├── configs/                      # Authoritative runtime & schema configs
│   ├── default.yaml
│   └── schema_mapping.yaml
│
├── tests/                        # Automated unit tests
│   ├── test_features.py
│   ├── test_labels.py
│   ├── test_leakage.py
│   ├── test_splits.py
│   ├── test_metrics.py
│   ├── test_privacy.py
│   └── test_auth.py
│
├── scripts/                      # Discovery, pipeline, paper & validation scripts
│   ├── inspect_dataset.py
│   ├── run_pipeline.py
│   ├── generate_paper.py
│   └── validate_repo.py
│
├── outputs/                      # Generated tables, figures & run manifest
│   ├── figures/
│   ├── tables/
│   └── run_manifest.json
│
├── paper/                        # Scientific research paper web app
│   ├── index.html
│   ├── assets/
│   │   ├── styles.css
│   │   └── script.js
│   └── figures/
│
├── submission/
│   └── paper_url.txt             # Single-line deployment target URL
│
├── .github/workflows/
│   └── deploy-paper.yml          # GitHub Pages automated deployment
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .gitignore
```

---

## 2. Validation Design

To prevent conflating operational effectiveness with cross-tenant transferability, evaluations are decoupled into two distinct regimes:

1. **Primary Evaluation — Chronological Split**:
   - Data is sorted strictly by time: Train (earliest 60%), Validation (middle 20%), Test (latest 20%).
   - Preserves temporal realism: models predict future decay using historical observations.
   - Client overlap between time periods is explicitly permitted.
2. **Secondary Robustness Evaluation — Client-Grouped Split**:
   - Independent `GroupShuffleSplit` on `client_id` enforcing $\text{Train Clients} \cap \text{Test Clients} = \emptyset$.
   - Evaluates cross-domain generalization to completely unseen website architectures.

*Primary temporal metrics and secondary client-grouped metrics are reported independently; they are never combined or averaged.*

---

## 3. Leakage Prevention Protocol

Features are audited in strict order of priority:
1. **Timestamp Availability**: Features must be strictly observable at or before cutoff $T$.
2. **Feature Provenance**: Documented derivation path from raw metrics.
3. **Future-Window Dependency**: Immediate disqualification of future-derived fields.
4. **Target-Derived Feature Detection**: Immediate blocking of trend derivatives (e.g. `trend_direction`).
5. **Precomputed Outcome Detection**: Blocking proprietary downstream action tags.
6. **Entity Identifier Exclusion**: Blocking client IDs and URLs from entering feature matrix.
7. **Secondary Diagnostics**: Correlation and mutual information used only as auxiliary checks (never as proof of absence of leakage).

*The pipeline raises a hard `LeakageError` if any `BLOCKED` feature enters model fitting.*

---

## 4. Current Empirical Benchmark

| Evaluation Regime | Heuristic Baseline Precision@50 | Model Precision@50 (Random Forest) | Precision Lift |
|---|---|---|---|
| **Primary Temporal Evaluation** | `[PENDING REAL DATA RUN]` | `[PENDING REAL DATA RUN]` | `[PENDING REAL DATA RUN]` |
| **Secondary Client-Grouped Robustness** | `[PENDING REAL DATA RUN]` | `[PENDING REAL DATA RUN]` | `[PENDING REAL DATA RUN]` |

*All empirical figures are populated dynamically via `scripts/run_pipeline.py` into `outputs/run_manifest.json`.*

---

## 5. Quickstart & Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit tests
pytest tests/

# 3. Inspect dataset schema
python scripts/inspect_dataset.py

# 4. Execute end-to-end pipeline (generates tables, figures, manifest)
python scripts/run_pipeline.py

# 5. Inject live results into research paper
python scripts/generate_paper.py

# 6. Run repository acceptance validation
python scripts/validate_repo.py
```

---

## 6. Hugging Face Authentication & Local Setup

The FlyRank warehouse dataset hosted at `FlyRank/internship-warehouse` is gated. Access requires a valid Hugging Face User Access Token (`HF_TOKEN`). Authentication is performed dynamically using DuckDB's Secrets Manager (`CREATE SECRET (TYPE HUGGINGFACE)`) over the `httpfs` extension without hardcoded credentials or arbitrary header settings.

> **Security Rule:** Never commit your `HF_TOKEN` or `.env` file to version control. The token must only be supplied via your local environment.

### Obtaining your Token
1. Request access on Hugging Face at [`FlyRank/internship-warehouse`](https://huggingface.co/datasets/FlyRank/internship-warehouse).
2. Generate a read token in your Hugging Face Account Settings (**Settings → Access Tokens → New Token → Read**).

### Local Configuration Options (Windows & Cross-Platform)

#### Option A: Local `.env` file (Recommended for local development)
Create a `.env` file in the repository root (copied from `.env.example`):

```env
HF_TOKEN=hf_your_actual_token_here
```

*Note: `.env` is ignored by `.gitignore` and will never be committed to Git.*

#### Option B: Windows PowerShell Environment Variable
```powershell
$env:HF_TOKEN="hf_your_actual_token_here"
python scripts/inspect_dataset.py
```

#### Option C: Command Prompt (cmd.exe)
```cmd
set HF_TOKEN=hf_your_actual_token_here
python scripts/inspect_dataset.py
```

### Verifying Access
Verify your configuration by running the dataset inspection script:
```bash
python scripts/inspect_dataset.py
```
If configured correctly, the script detects `HF_TOKEN`, connects to the FlyRank warehouse, and displays non-sensitive schema metadata.

---

## 7. Honest Research Framing

- **Decision-Support Focus**: This work models observable page-level engagement signals to assist human editorial teams.
- **No Causal Claims**: Content updates are not guaranteed to cause organic traffic recovery; ranking results are influenced by external competition, indexation shifts, and search engine algorithm updates.
- **Data Credit**: Research developed using anonymized data structures provided by [FlyRank](https://flyrank.ai).
