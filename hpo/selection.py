"""Publish confirmed HPO parameters for automatic full-experiment loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modules.common import json_default, write_json
from modules.hpo_selection import HPO_SETTING_NAMES, save_selection

from .regularized_config import SearchConfig


def export_selected_parameters(
    config: SearchConfig,
    selected: dict[str, Any],
    directory: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    payload = {
        **selected,
        "selection_schema_version": 1,
        "model_variant": config.model_variant,
        "model_family": config.model_family,
        "source_hpo_output_dir": str(config.output_dir),
        "hpo_settings": {name: getattr(config, name) for name in HPO_SETTING_NAMES},
    }
    payload = json.loads(json.dumps(payload, default=json_default))
    path = save_selection(config.model_variant, payload, directory)
    write_json(config.output_dir / config.selection_filename, payload)
    write_json(config.output_dir / "confirmation" / "selected_parameters.json", payload)
    return payload, path
