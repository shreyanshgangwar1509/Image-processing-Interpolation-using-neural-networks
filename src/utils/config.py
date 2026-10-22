"""Config loading, recursive base inheritance, and string override utilities."""
import copy
from pathlib import Path
from typing import Any, Dict, List, Union
import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges override dict into base dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a YAML configuration file.

    If the YAML contains `_base_: <relative_path>`, loads that base config first
    and recursively deep-merges the current file over it. Supports chained bases.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "_base_" in cfg:
        base_rel_path = cfg.pop("_base_")
        base_abs_path = (config_path.parent / base_rel_path).resolve()
        base_cfg = load_config(base_abs_path)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Applies dot-notation key=value overrides to cfg dictionary in-place.

    Values are parsed using yaml.safe_load so types (int, float, bool, list) are converted.
    """
    if not overrides:
        return cfg

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'key=value'.")
        key_str, val_str = override.split("=", 1)
        parsed_val = yaml.safe_load(val_str)

        keys = key_str.strip().split(".")
        curr = cfg
        for k in keys[:-1]:
            if k not in curr or not isinstance(curr[k], dict):
                curr[k] = {}
            curr = curr[k]
        curr[keys[-1]] = parsed_val

    return cfg
