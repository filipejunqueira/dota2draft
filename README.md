# Dota 2 Draft Assistant

This project is a command-line application designed to fetch, analyze, and predict Dota 2 match outcomes based on hero drafts. It provides a suite of tools to gather data from the OpenDota API, store it in a local SQLite database, perform detailed laning-phase analysis, and train a neural network to predict lane scores.

## Project Structure

The project has been refactored into a modular Python package for better organization and maintainability:

```
.dota2_draft_nn/
├── dota2draft/               # Main application package
│   ├── __init__.py
│   ├── api.py                # Handles communication with the OpenDota API.
│   ├── analysis.py           # Contains logic for lane analysis and data visualization.
│   ├── core.py               # Orchestrates data flow between API, DB, and other components.
│   ├── db.py                 # Manages the local SQLite database.
│   └── model.py              # Defines the PyTorch neural network and related functions.
├── dota2draft_cli.py         # The single command-line entry point for the application.
├── opendota_league_info.db   # Local SQLite database for caching API data.
├── lanes.csv                 # Default output file for lane analysis exports.
├── requirements.txt          # Lists all Python dependencies.
└── README.md                 # This file.
```

## Setup

1.  **Clone the repository** (if you haven't already).

2.  **Install dependencies**: It is recommended to use a virtual environment. Navigate to the project root directory and run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

All functionalities are accessed through `dota2draft_cli.py`. The CLI is organized into logical groups of commands (`leagues`, `heroes`, `players`, `nn`). You can see a list of all available commands by running:

```bash
python dota2draft_cli.py --help
```

### Data Fetching & Management

First, you need to populate your local database with data from the OpenDota API.

1.  **Refresh Static Data**: It's good practice to start by getting the latest list of heroes, teams, and leagues.
    ```bash
    python dota2draft_cli.py refresh-static
    ```

2.  **Fetch League Matches**: Download and store all match details for a specific league. You can find league IDs on the OpenDota website or by using the `leagues` commands.

    The command features a progress bar, skips matches that are already downloaded (unless `--force-refresh` is used), and provides a summary of the operation.
    ```bash
    # Example: Fetch all matches for league ID 15728 (ESL One Birmingham 2024)
    python dota2draft_cli.py fetch-league 15728
    ```

### Exploring the Database

The CLI provides several commands to query the data you've downloaded.

#### Leagues Commands (`leagues`)

Manage and view information about professional leagues.

-   **List All Leagues**: `python dota2draft_cli.py leagues list`
-   **Search for a League**: `python dota2draft_cli.py leagues search "International"`
-   **List Downloaded Leagues**: `python dota2draft_cli.py leagues downloaded`

#### Hero Commands (`heroes`)

View hero statistics and manage hero nicknames.

-   **Get Hero Stats**: Calculate hero pick/ban/win rates. You can filter by league or date.
    ```bash
    # Get hero stats across all matches
    python dota2draft_cli.py heroes stats

    # Get hero stats for a specific league and after a certain date (post-patch analysis)
    python dota2draft_cli.py heroes stats --league-id 15728 --after-date 2024-01-01
    ```
-   **Set Hero Nickname**: Assign a custom, memorable nickname to a hero. This nickname can be used in other commands, like `nn predict`.
    ```bash
    # Assign 'AM' to Anti-Mage
    python dota2draft_cli.py heroes set-nickname "Anti-Mage" "AM"

    # You can also use the hero's ID
    python dota2draft_cli.py heroes set-nickname 1 "Magina"
    ```
-   **List Hero Nicknames**: View all nicknames assigned to a specific hero.
    ```bash
    python dota2draft_cli.py heroes list-nicknames "Anti-Mage"
    ```
-   **Remove Hero Nickname**: Delete a specific nickname from a hero.
    ```bash
    python dota2draft_cli.py heroes remove-nickname "Anti-Mage" "Magina"
    ```

#### Player Commands (`players`)

View player statistics and manage player nicknames.

-   **Get Player Stats**: View detailed stats for a specific player.
    ```bash
    # Use the player's account ID
    python dota2draft_cli.py players stats 12345678

    # Use a nickname you've assigned
    python dota2draft_cli.py players stats "MyFavoritePlayer"
    ```
-   **Set Player Nickname**: Assign a fixed nickname to a player's account ID. This is useful for tracking players who frequently change their in-game name.
    ```bash
    python dota2draft_cli.py players set-nickname 12345678 "MyFavoritePlayer"
    ```
-   **List Player Nicknames**: View all nicknames assigned to a specific player.
    ```bash
    python dota2draft_cli.py players list-nicknames 12345678
    ```
-   **Remove Player Nickname**: Delete a specific nickname from a player.
    ```bash
    python dota2draft_cli.py players remove-nickname 12345678 "MyFavoritePlayer"
    ```

### Match-Specific Analysis

-   **Analyze Lanes**: Perform a detailed laning-phase analysis for a match.
    ```bash
    python dota2draft_cli.py analyze-lanes <MATCH_ID>
    ```
-   **Export Analysis to CSV**: Run the lane analysis for all matches of a league and export the results to a CSV file. This CSV is required for training the neural network.
    ```bash
    python dota2draft_cli.py export-analysis <LEAGUE_ID>
    ```

### Neural Network (`nn`)

The `nn` subcommand contains all model-related operations.

1.  **Train the Model**: Train the neural network using the data from a CSV file.
    ```bash
    python dota2draft_cli.py nn train --csv-file lanes.csv --epochs 100
    ```
    This will save the trained model weights to `dota_draft_predictor_weights.pth`. All generated PNG image files (e.g., training loss plot) will be saved in the `nn_artifacts/` directory.

2.  **Make Predictions**: Use the trained model to predict lane scores for a given draft. You can use official hero names or any nicknames you've set.
    ```bash
    # Using official names
    python dota2draft_cli.py nn predict "Radiant Pick: Axe; Dire Pick: Juggernaut"

    # Using a hero nickname
    python dota2draft_cli.py nn predict "Radiant Pick: AM; Dire Pick: Juggernaut"
    ```

## Configuration

The application's behavior can be customized through the `config.yaml` file. This file allows you to set default paths for the database, CSV outputs, and model weights, as well as configure parameters for the lane analysis KPIs and neural network architecture.

Many configuration values can be overridden at runtime using CLI options (e.g., `--out` for `export-analysis`, `--model-file` for `nn` commands).
