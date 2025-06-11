# dota2draft/config_loader.py

import yaml
import os
from rich.console import Console

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
        "learning_rate": 0.001
    }
}

_config_cache = None

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
            
            # Simple merge with defaults to ensure all keys are present (does not deeply merge nested dicts perfectly)
            # For a more robust solution, a deep merge utility would be better.
            config = DEFAULT_CONFIG.copy()
            config.update(loaded_config)
            # Ensure nested dicts like kpi_parameters and nn_training_defaults are also handled
            if 'kpi_parameters' in loaded_config and isinstance(loaded_config['kpi_parameters'], dict):
                config['kpi_parameters'] = DEFAULT_CONFIG['kpi_parameters'].copy()
                config['kpi_parameters'].update(loaded_config['kpi_parameters'])
            if 'nn_training_defaults' in loaded_config and isinstance(loaded_config['nn_training_defaults'], dict):
                config['nn_training_defaults'] = DEFAULT_CONFIG['nn_training_defaults'].copy()
                config['nn_training_defaults'].update(loaded_config['nn_training_defaults'])

            _config_cache = config
            return config
    except (yaml.YAMLError, IOError) as e:
        console.print(f"[bold red]Error loading or parsing config file '{CONFIG_FILE_PATH}': {e}[/bold red]")
        console.print("[yellow]Using internal default configuration values.[/yellow]")
        _config_cache = DEFAULT_CONFIG # Fallback to internal defaults
        return DEFAULT_CONFIG

# Load config on module import to make it available immediately
CONFIG = load_config()
