"""
Configuration loader and schema validator for FlyRank Capstone.
Loads settings from configs/default.yaml and configs/schema_mapping.yaml.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
try:
    import yaml
except ImportError:
    yaml = None


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Lightweight fallback YAML parser for environments before pip install."""
    result: Dict[str, Any] = {}
    current_dict = result
    stack = [(0, result)]

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        while stack and indent < stack[-1][0]:
            stack.pop()

        current_dict = stack[-1][1]

        if ":" in stripped:
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()

            if not val:
                new_dict: Dict[str, Any] = {}
                if isinstance(current_dict, dict):
                    current_dict[key] = new_dict
                stack.append((indent + 2, new_dict))
            else:
                # Type conversion
                if val.lower() == "true":
                    v_parsed = True
                elif val.lower() == "false":
                    v_parsed = False
                elif val.lower() in ("null", "none"):
                    v_parsed = None
                elif val.startswith("[") and val.endswith("]"):
                    inner = val[1:-1].strip()
                    v_parsed = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
                else:
                    try:
                        v_parsed = int(val) if val.isdigit() else float(val)
                    except ValueError:
                        v_parsed = val.strip("'\"")
                if isinstance(current_dict, dict):
                    current_dict[key] = v_parsed

    return result


def _read_yaml_file(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    if yaml is not None:
        return yaml.safe_load(content) or {}
    return _parse_simple_yaml(content)


@dataclass
class LabelConfig:
    lookback_days: int = 28
    future_horizon_days: int = 28
    min_lookback_impressions: float = 100.0
    min_lookback_clicks: float = 10.0
    min_future_active_days: int = 14
    click_deterioration_threshold: float = 0.20
    position_deterioration_threshold: float = 2.0
    exclusion_rules: Dict[str, bool] = field(
        default_factory=lambda: {
            "exclude_zero_traffic": True,
            "exclude_sparse_history": True,
            "exclude_missing_dates": True,
        }
    )


@dataclass
class ValidationConfig:
    primary_method: str = "chronological"
    primary_train_ratio: float = 0.60
    primary_val_ratio: float = 0.20
    primary_test_ratio: float = 0.20
    secondary_method: str = "client_grouped"
    secondary_test_size: float = 0.20
    secondary_group_col: str = "client_id"


@dataclass
class EvaluationConfig:
    k_values: List[int] = field(default_factory=lambda: [10, 25, 50, 100])
    primary_k: int = 50
    tie_breaking: str = "first"


@dataclass
class AppConfig:
    project_name: str = "flyrank-applied-search-intelligence"
    random_seed: int = 42
    raw_data_path: str = "data/raw/content_refresh_anonymized.csv"
    label: LabelConfig = field(default_factory=LabelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    model_params: Dict[str, Any] = field(default_factory=dict)
    schema_mapping: Dict[str, Any] = field(default_factory=dict)


def load_config(
    config_path: Optional[str] = None,
    schema_path: Optional[str] = None,
) -> AppConfig:
    """Load configuration files with environment variable overrides."""
    root = Path(__file__).resolve().parent.parent

    if config_path is None:
        config_path = str(root / "configs" / "default.yaml")
    if schema_path is None:
        schema_path = str(root / "configs" / "schema_mapping.yaml")

    cfg_dict = _read_yaml_file(Path(config_path))
    schema_dict = _read_yaml_file(Path(schema_path))

    project_cfg = cfg_dict.get("project", {})
    data_cfg = cfg_dict.get("data", {})
    lbl_cfg = cfg_dict.get("label", {})
    val_cfg = cfg_dict.get("validation", {})
    eval_cfg = cfg_dict.get("evaluation", {})
    model_cfg = cfg_dict.get("models", {})

    seed = int(os.getenv("RANDOM_SEED", project_cfg.get("random_seed", 42)))
    raw_path = os.getenv("FLYRANK_DATA_PATH", data_cfg.get("raw_path", "data/raw/content_refresh_anonymized.csv"))

    label_obj = LabelConfig(
        lookback_days=lbl_cfg.get("lookback_days", 28),
        future_horizon_days=lbl_cfg.get("future_horizon_days", 28),
        min_lookback_impressions=lbl_cfg.get("min_lookback_impressions", 100.0),
        min_lookback_clicks=lbl_cfg.get("min_lookback_clicks", 10.0),
        min_future_active_days=lbl_cfg.get("min_future_active_days", 14),
        click_deterioration_threshold=lbl_cfg.get("click_deterioration_threshold", 0.20),
        position_deterioration_threshold=lbl_cfg.get("position_deterioration_threshold", 2.0),
        exclusion_rules=lbl_cfg.get("exclusion_rules", {}),
    )

    primary_val = val_cfg.get("primary", {})
    secondary_val = val_cfg.get("secondary", {})

    validation_obj = ValidationConfig(
        primary_method=primary_val.get("method", "chronological"),
        primary_train_ratio=primary_val.get("train_ratio", 0.60),
        primary_val_ratio=primary_val.get("val_ratio", 0.20),
        primary_test_ratio=primary_val.get("test_ratio", 0.20),
        secondary_method=secondary_val.get("method", "client_grouped"),
        secondary_test_size=secondary_val.get("test_size", 0.20),
        secondary_group_col=secondary_val.get("group_col", "client_id"),
    )

    eval_obj = EvaluationConfig(
        k_values=eval_cfg.get("k_values", [10, 25, 50, 100]),
        primary_k=eval_cfg.get("primary_k", 50),
        tie_breaking=eval_cfg.get("tie_breaking", "first"),
    )

    return AppConfig(
        project_name=project_cfg.get("name", "flyrank-applied-search-intelligence"),
        random_seed=seed,
        raw_data_path=raw_path,
        label=label_obj,
        validation=validation_obj,
        evaluation=eval_obj,
        model_params=model_cfg,
        schema_mapping=schema_dict,
    )
