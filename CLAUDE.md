# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dota 2 Draft Assistant - A command-line application that fetches Dota 2 match data, analyzes hero drafts, and uses PyTorch neural networks to predict lane scores and match outcomes.

## Development Commands

### Testing
```bash
pytest tests/                    # Run all tests
pytest tests/test_cli.py        # Test CLI functionality  
pytest tests/test_db.py         # Test database operations
```

### Setup & Dependencies
```bash
pip install -r requirements.txt # Install dependencies
pip install -e .                # Install package in development mode
```

### Main Application Commands
```bash
# Data management
python dota2draft_cli.py refresh-static
python dota2draft_cli.py leagues fetch <league_id>
python dota2draft_cli.py leagues list

# Analysis & ML
python dota2draft_cli.py analyze-lanes <match_id>
python dota2draft_cli.py export-analysis <league_id>  
python dota2draft_cli.py nn train --csv-file lanes.csv --epochs 100
python dota2draft_cli.py nn predict "Radiant Pick: Axe; Dire Pick: Juggernaut"

# Interactive mode
python dota2draft_cli_interactive.py
```

## Architecture

### Core Package Structure (`dota2draft/`)
- **api.py** - OpenDota API client with rate limiting (60 req/min)
- **db.py** - SQLite database management (opendota_league_info.db)
- **core.py** - Data service orchestration layer
- **analysis.py** - Lane analysis and visualization logic  
- **model.py** - PyTorch neural network (132 input → 128/256/128/64 hidden → 6 output)
- **config_loader.py** - YAML configuration management

### Entry Points
- **dota2draft_cli.py** - Main CLI with typer/rich
- **dota2draft_cli_interactive.py** - Interactive questionnaire interface

### Key Files
- **config.yaml** - Main configuration (database paths, ML parameters, KPIs)
- **opendota_league_info.db** - SQLite database (matches, heroes, teams, leagues)
- **dota_draft_predictor_weights.pth** - Trained model weights
- **nn_artifacts/** - Training plots and evaluation results

## Data Flow
1. **Fetch**: OpenDota API → SQLite database
2. **Analyze**: Database → Lane analysis → CSV export  
3. **Train**: CSV data → PyTorch model → Saved weights
4. **Predict**: Draft input + Model → Lane score predictions

## Technology Stack
- **CLI**: typer, rich, questionary
- **ML**: torch, scikit-learn, numpy
- **Data**: requests, PyYAML, matplotlib, seaborn
- **Database**: SQLite with custom schema
- **Testing**: pytest, pytest-mock

## Configuration
All settings in `config.yaml` including:
- Database and output paths
- Neural network architecture parameters
- KPI thresholds for lane analysis
- Logging configuration