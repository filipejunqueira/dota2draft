# dota2draft/config_loader.py

import yaml
import os
from rich.console import Console
from .exceptions import ConfigurationError

console = Console()

CONFIG_FILE_PATH = "config.yaml"

DEFAULT_CONFIG = {
    "database_path": "opendota_league_info.db",
    "csv_output_path": "lanes.csv",
    "nn_artifacts_path": "nn_artifacts/",
    "log_file_path": "dota2draft.log",
    "model_weights_path": "dota_draft_predictor_weights.pth",
    "kpi_parameters": {
        "analysis_minute_mark": 10,
        "kill_death_analysis_time_limit_seconds": 600,
        "early_tower_kill_time_limit_seconds": 720,
        "score_weights": {
            "major_gold_lead_per_laner_threshold": 750,
            "minor_gold_lead_per_laner_threshold": 300,
            "major_xp_lead_per_laner_threshold": 1000,
            "minor_xp_lead_per_laner_threshold": 500,
            "points_for_major_lead": 2,
            "points_for_minor_lead": 1,
            "kill_difference_for_major_points": 2,
            "points_for_major_kill_difference": 2,
            "kill_difference_for_minor_points": 1,
            "points_for_minor_kill_difference": 1,
            "points_for_early_tower_kill": 3
        },
        "display_secondary_minute_mark": 8
    },
    "nn_training_defaults": {
        "num_heroes_input": 132,
        "hidden_layer_size1": 128,
        "hidden_layer_size2": 256,
        "hidden_layer_size3": 128,
        "hidden_layer_size4": 64,
        "output_layer_size": 6,
        "epochs": 50,
        "batch_size": 10,
        "learning_rate": 0.001,
        "random_seed": 42
    }
}

_config_cache = None

def _deep_merge_config(default: dict, override: dict) -> dict:
    """
    Deep merge configuration dictionaries.
    Override values take precedence over defaults, but missing keys from defaults are preserved.
    """
    result = default.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_config(result[key], value)
        else:
            result[key] = value
    
    return result

def load_config() -> dict:
    """Loads the YAML configuration file.

    If the file doesn't exist, it creates one with default values.
    Returns a dictionary representing the configuration.
    Caches the loaded config to avoid repeated file I/O.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if not os.path.exists(CONFIG_FILE_PATH):
        console.print(f"[yellow]Warning: Configuration file '{CONFIG_FILE_PATH}' not found. Creating with default values.[/yellow]")
        try:
            with open(CONFIG_FILE_PATH, 'w') as f:
                yaml.dump(DEFAULT_CONFIG, f, sort_keys=False)
            _config_cache = DEFAULT_CONFIG
            return DEFAULT_CONFIG
        except IOError as e:
            console.print(f"[bold red]Error: Could not write default config file '{CONFIG_FILE_PATH}': {e}[/bold red]")
            console.print("[yellow]Using internal default configuration values.[/yellow]")
            _config_cache = DEFAULT_CONFIG # Use internal defaults if write fails
            return DEFAULT_CONFIG

    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            loaded_config = yaml.safe_load(f)
            if not isinstance(loaded_config, dict):
                console.print(f"[bold red]Error: Config file '{CONFIG_FILE_PATH}' is malformed. Expected a dictionary.[/bold red]")
                raise yaml.YAMLError("Config is not a dictionary")
            
            # Deep merge with defaults to ensure all keys are present
            config = _deep_merge_config(DEFAULT_CONFIG, loaded_config)

            _config_cache = config
            return config
    except (yaml.YAMLError, IOError) as e:
        console.print(f"[bold red]Error loading or parsing config file '{CONFIG_FILE_PATH}': {e}[/bold red]")
        console.print("[yellow]Using internal default configuration values.[/yellow]")
        _config_cache = DEFAULT_CONFIG # Fallback to internal defaults
        return DEFAULT_CONFIG

# Load config on module import to make it available immediately
CONFIG = load_config()
