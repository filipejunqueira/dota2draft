# dota2draft_cli.py

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import csv
import os
import json # Added for json output in analyze_lanes

# Import modules from the new package structure
from dota2draft.db import DBManager
from dota2draft.api import OpenDotaAPIClient
from dota2draft.core import DataService
from dota2draft.analysis import perform_core_lane_analysis_for_match, visualize_json_structure, _get_hero_display_name_from_hero_id
from dota2draft.model import (
    DraftLanePredictor, load_and_preprocess_data, train_model, evaluate_model, 
    save_model_weights, load_model_weights, predict_draft, 
    plot_training_loss, plot_evaluation_results
)
from dota2draft.config_loader import CONFIG
from dota2draft.logger_config import logger # Import the configured logger
import torch
import torch.nn as nn
import torch.optim as optim

# --- Initialization ---
console = Console() # Keep for Rich-specific output like tables/panels
app = typer.Typer(
    help=Panel(
        "[bold green]Dota 2 Draft Assistant[/bold green]\n" 
        "A CLI tool to fetch match data, analyze lane outcomes, and train a predictive model.",
        title="Welcome",
        border_style="blue"
    )
)
nn_app = typer.Typer(help="Commands for Neural Network training and prediction.")
leagues_app = typer.Typer(help="Commands for viewing and searching leagues.")
stats_app = typer.Typer(help="Commands for database statistics.")

app.add_typer(nn_app, name="nn")
app.add_typer(leagues_app, name="leagues")
app.add_typer(stats_app, name="stats")

# Instantiate core components
# These will now use the logger internally where appropriate
db_manager = DBManager()
api_client = OpenDotaAPIClient()
data_service = DataService(db_manager, api_client)

# --- Helper Functions for Display ---
def display_draft_info(match_data: dict, hero_map: dict):
    if not match_data.get('picks_bans'):
        logger.warning("No pick/ban information available for this match.")
        console.print("[yellow]No pick/ban information available for this match.[/yellow]")
        return

    table = Table(title=f"Draft for Match ID: {match_data['match_id']}", show_header=True, header_style="bold magenta")
    table.add_column("Order", style="dim")
    table.add_column("Team")
    table.add_column("Action")
    table.add_column("Hero")

    for action in sorted(match_data['picks_bans'], key=lambda x: x['order']):
        team = "[green]Radiant[/green]" if action['team'] == 0 else "[bold red]Dire[/bold red]"
        action_type = "Pick" if action['is_pick'] else "Ban"
        hero_name = _get_hero_display_name_from_hero_id(action['hero_id'], hero_map)
        table.add_row(str(action['order']), team, action_type, hero_name)
    
    console.print(table)

# --- Main CLI Commands ---

# --- Leagues Commands ---
@leagues_app.command(name="list", help="Display all leagues in the database.")
def list_leagues():
    logger.info("Fetching all leagues from the database.")
    leagues = db_manager.get_all_leagues()
    if not leagues:
        logger.warning("No leagues found in the database. Use 'refresh-static' to fetch them.")
        console.print("[yellow]No leagues found in the database. Use 'refresh-static' to fetch them.[/yellow]")
        return

    table = Table(title="All Stored Leagues", show_header=True, header_style="bold magenta")
    table.add_column("League ID", style="cyan")
    table.add_column("League Name")
    table.add_column("Tier", style="green")

    for league in leagues:
        table.add_row(str(league['leagueid']), league['name'], league['tier'])
    
    console.print(table)

@leagues_app.command(name="search", help="Search for leagues by name.")
def search_leagues(keyword: str = typer.Argument(..., help="Keyword to search for in league names.")):
    logger.info(f"Searching for leagues with keyword: '{keyword}'")
    leagues = db_manager.get_leagues_by_name(keyword)
    if not leagues:
        logger.warning(f"No leagues found matching '{keyword}'.")
        console.print(f"[yellow]No leagues found matching '{keyword}'.[/yellow]")
        return

    table = Table(title=f"Leagues Matching '{keyword}'", show_header=True, header_style="bold magenta")
    table.add_column("League ID", style="cyan")
    table.add_column("League Name")
    table.add_column("Tier", style="green")

    for league in leagues:
        table.add_row(str(league['leagueid']), league['name'], league['tier'])
    
    console.print(table)

# --- Stats Commands ---
@stats_app.command(name="heroes", help="Display hero statistics from stored matches.")
def hero_stats(
    league_id: int = typer.Option(None, "--league-id", help="Filter stats for a specific league ID."),
    after_date: str = typer.Option(None, "--after-date", help="Only include matches after this date (YYYY-MM-DD).")
):
    logger.info(f"Calculating hero stats. League filter: {league_id if league_id else 'None'}. Date filter: {after_date}")
    stats = db_manager.get_hero_stats(league_id, after_date)
    if not stats:
        logger.warning("No hero stats could be calculated. Ensure matches are fetched.")
        console.print("[yellow]No hero stats could be calculated. Fetch matches using 'fetch-league' first.[/yellow]")
        return

    total_matches = stats[0]['total_matches'] if stats else 0
    title = f"Hero Statistics (Total Matches: {total_matches})"
    if league_id:
        title += f" - League ID: {league_id}"
    if after_date:
        title += f" - After {after_date}"

    table = Table(title=title, show_header=True, header_style="bold blue")
    table.add_column("Hero")
    table.add_column("Picks", justify="right")
    table.add_column("Bans", justify="right")
    table.add_column("Pick Rate", justify="right")
    table.add_column("Ban Rate", justify="right")
    table.add_column("Wins", justify="right")
    table.add_column("Win Rate %", justify="right")

    for hero in sorted(stats, key=lambda x: x['hero_name']):
        pick_rate = (hero['picks'] / total_matches * 100) if total_matches > 0 else 0
        ban_rate = (hero['bans'] / total_matches * 100) if total_matches > 0 else 0
        win_rate = (hero['wins'] / hero['picks'] * 100) if hero['picks'] > 0 else 0
        table.add_row(
            hero['hero_name'],
            str(hero['picks']),
            str(hero['bans']),
            f"{pick_rate:.2f}%",
            f"{ban_rate:.2f}%",
            str(hero['wins']),
            f"{win_rate:.2f}%"
        )
    console.print(table)

@stats_app.command(name="players", help="Display player statistics from stored matches.")
def player_stats(
    league_id: int = typer.Option(None, "--league-id", help="Filter stats for a specific league ID."),
    after_date: str = typer.Option(None, "--after-date", help="Only include matches after this date (YYYY-MM-DD).")
):
    logger.info(f"Calculating player stats. League filter: {league_id if league_id else 'None'}. Date filter: {after_date}")
    stats = db_manager.get_player_stats(league_id, after_date)
    if not stats:
        logger.warning("No player stats could be calculated. Ensure matches are fetched.")
        console.print("[yellow]No player stats could be calculated. Fetch matches using 'fetch-league' first.[/yellow]")
        return

    title = "Player Statistics"
    if league_id:
        title += f" - League ID: {league_id}"
    if after_date:
        title += f" - After {after_date}"

    table = Table(title=title, show_header=True, header_style="bold green")
    table.add_column("Player Name")
    table.add_column("Matches Played", justify="right")
    table.add_column("Wins", justify="right")
    table.add_column("Win Rate %", justify="right")

    for player in sorted(stats, key=lambda x: x['name']):
        win_rate = (player['wins'] / player['matches_played'] * 100) if player['matches_played'] > 0 else 0
        table.add_row(
            player['name'],
            str(player['matches_played']),
            str(player['wins']),
            f"{win_rate:.2f}%"
        )
    console.print(table)

# --- Main CLI Commands ---

@app.command(name="refresh-static", help="Force refresh static data (heroes, teams, leagues) from the API.")
def refresh_static_data_command():
    logger.info("Attempting to refresh all static data...")
    console.print(Panel("[bold yellow]Refreshing all static data... This will clear existing static data tables.[/bold yellow]"))
    if not typer.confirm("Are you sure you want to proceed? This action cannot be undone."):
        logger.info("Static data refresh cancelled by user.")
        console.print("[yellow]Operation cancelled by user.[/yellow]")
        raise typer.Abort()
    
    data_service.get_hero_map(force_refresh=True)
    data_service.get_team_map(force_refresh=True)
    data_service.get_all_leagues(force_refresh=True)
    logger.info("Static data refreshed successfully.")
    console.print(Panel("[bold green]Static data refreshed successfully![/bold green]"))

@app.command(name="fetch-league", help="Fetch and store all matches for a given league ID.")
def populate_league_matches_command(league_id: int = typer.Argument(..., help="The ID of the league to populate.")):
    logger.info(f"Fetching match summaries for league {league_id}...")
    matches = api_client.fetch_league_matches_summary(league_id)
    if not matches:
        logger.error(f"Could not fetch matches for league {league_id}.")
        console.print(f"[red]Could not fetch matches for league {league_id}.[/red]")
        raise typer.Exit(code=1)
    
    logger.info(f"Found {len(matches)} matches for league {league_id}. Processing details...")
    for match_summary in matches:
        match_id = match_summary['match_id']
        logger.debug(f"Processing Match ID: {match_id}")
        data_service.get_match_details(match_id, force_refresh=True)
    logger.info(f"Finished processing all matches for league {league_id}.")

@app.command(name="get-draft", help="Display the draft for a specific match ID.")
def get_draft_details_command(match_id: int = typer.Argument(..., help="The Match ID.")):
    logger.info(f"Fetching draft details for match ID: {match_id}")
    hero_map = data_service.get_hero_map()
    match_data = data_service.get_match_details(match_id)
    if match_data:
        display_draft_info(match_data, hero_map)
    else:
        logger.error(f"Could not retrieve details for match {match_id}.")
        console.print(f"[red]Could not retrieve details for match {match_id}.[/red]")

@app.command(name="analyze-lanes", help="Perform and display lane analysis for a specific match ID.")
def analyze_lanes_command(
    match_id: int = typer.Argument(..., help="The Match ID to analyze."),
    output_format: str = typer.Option("table", "--format", help="Output format: 'table' or 'json'.")
):
    logger.info(f"Performing lane analysis for match ID: {match_id}, format: {output_format}")
    hero_map = data_service.get_hero_map()
    match_data = data_service.get_match_details(match_id)
    if not match_data:
        logger.error(f"Could not get match data for {match_id} to perform analysis.")
        console.print(f"[red]Could not get match data for {match_id}.[/red]")
        raise typer.Exit(code=1)
    
    analysis_results = perform_core_lane_analysis_for_match(match_data, hero_map)
    
    if analysis_results.get("error"):
        logger.error(f"Analysis Error for match {match_id}: {analysis_results['error']}")
        console.print(f"[red]Analysis Error: {analysis_results['error']}[/red]")
        raise typer.Exit(code=1)

    if output_format.lower() == 'json':
        logger.debug(f"Outputting analysis for match {match_id} in JSON format.")
        console.print(json.dumps(analysis_results, indent=2))
    elif output_format.lower() == 'table':
        logger.debug(f"Outputting analysis for match {match_id} in table format.")
        table = Table(title=f"Laning Phase Analysis for Match {match_id}", show_header=True, header_style="bold blue")
        table.add_column("Lane")
        table.add_column("Radiant Score", justify="right")
        table.add_column("Dire Score", justify="right")
        table.add_column("Details")

        for lane, data in analysis_results["lanes"].items():
            r_score = data['radiant'].get('score', 0)
            d_score = data['dire'].get('score', 0)
            details = f"Radiant: {data['radiant'].get('heroes', [])} | Dire: {data['dire'].get('heroes', [])}"
            table.add_row(lane.capitalize(), str(r_score), str(d_score), details)

        console.print(table)
        console.print(f"[bold]Draft String:[/] {analysis_results['draft_order']}")
    else:
        logger.warning(f"Invalid output format specified: '{output_format}'. Defaulting to table or erroring.")
        console.print(f"[red]Invalid output format: '{output_format}'. Choose 'table' or 'json'.[/red]")

@app.command(name="export-analysis", help="Analyze all matches in a league and export to a CSV file.")
def export_analysis_command(
    league_id: int = typer.Argument(..., help="The ID of the league to analyze."),
    output_file: str = typer.Option(CONFIG['csv_output_path'], "--out", "-o", help="Output CSV file name.")
):
    logger.info(f"Exporting analysis for league {league_id} to {output_file}")
    if os.path.exists(output_file):
        logger.warning(f"Output file '{output_file}' already exists.")
        if not typer.confirm(f"File '{output_file}' already exists. Overwrite?"):
            logger.info("Export operation cancelled by user due to existing file.")
            console.print("[yellow]Operation cancelled by user.[/yellow]")
            raise typer.Abort()

    matches = db_manager.get_all_matches_from_league(league_id)
    if not matches:
        logger.warning(f"No matches found in local DB for league {league_id}. Use 'fetch-league' first.")
        console.print("[yellow]No matches found in the local DB for this league. Use 'fetch-league' first.[/yellow]")
        raise typer.Exit(code=1)

    hero_map = data_service.get_hero_map()
    header = ['match_id', 'draft_order', 'top_radiant_score', 'top_dire_score', 'mid_radiant_score', 'mid_dire_score', 'bot_radiant_score', 'bot_dire_score', 'analysis_error_message']

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for match_data_item in matches: # Renamed to avoid conflict with outer scope 'match_data'
                analysis = perform_core_lane_analysis_for_match(match_data_item, hero_map)
                row = {
                    'match_id': analysis.get('match_id'),
                    'draft_order': analysis.get('draft_order'),
                    'analysis_error_message': analysis.get('analysis_error_message')
                }
                for lane in ['top', 'mid', 'bot']:
                    row[f'{lane}_radiant_score'] = analysis['lanes'][lane]['radiant'].get('score')
                    row[f'{lane}_dire_score'] = analysis['lanes'][lane]['dire'].get('score')
                writer.writerow(row)
        logger.info(f"Analysis exported successfully to {os.path.abspath(output_file)}")
        console.print(f"[green]Analysis exported to {os.path.abspath(output_file)}[/green]")
    except IOError as e:
        logger.error(f"Failed to write CSV file to {output_file}: {e}")
        console.print(f"[red]Error writing CSV file: {e}[/red]")
        raise typer.Exit(code=1)

# --- NN CLI Commands ---

@nn_app.command(name="train", help="Train the draft prediction model.")
def nn_train_command(
    csv_file: str = typer.Option(CONFIG['csv_output_path'], help="Path to the training data CSV."),
    epochs: int = typer.Option(CONFIG['nn_training_defaults']['epochs'], help="Number of training epochs."),
    batch_size: int = typer.Option(CONFIG['nn_training_defaults']['batch_size'], help="Batch size for training."),
    learning_rate: float = typer.Option(CONFIG['nn_training_defaults']['learning_rate'], help="Learning rate for the optimizer."),
    model_file: str = typer.Option(CONFIG['model_weights_path'], "--model-file", help="Path to save/load model weights. Overrides config.")
):
    logger.info(f"Starting NN training with CSV: {csv_file}, Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")
    logger.info(f"Model weights will be handled via: {model_file if model_file else CONFIG['model_weights_path']}")
    # Setup model
    train_loader, val_loader, _, num_heroes_in_map = load_and_preprocess_data(csv_file, batch_size)
    
    nn_params = CONFIG['nn_training_defaults']
    model = DraftLanePredictor(
        num_heroes=num_heroes_in_map,
        hidden_layer_size1=nn_params['hidden_layer_size1'],
        hidden_layer_size2=nn_params['hidden_layer_size2'],
        hidden_layer_size3=nn_params['hidden_layer_size3'],
        hidden_layer_size4=nn_params['hidden_layer_size4'],
        output_layer_size=nn_params['output_layer_size']
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    trained_model, train_hist, val_hist = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
    
    # Evaluate
    avg_mse, mae, actual_targets, predicted_outputs = evaluate_model(trained_model, val_loader, criterion)
    logger.info(f"Validation Set - MSE: {avg_mse:.4f}, MAE: {mae:.4f}")
    console.print(f"Validation Set - MSE: {avg_mse:.4f}, MAE: {mae:.4f}") # Keep console for direct user feedback

    # Save artifacts
    save_model_weights(trained_model, file_path_override=model_file)
    plot_training_loss(train_hist, val_hist)
    score_names = ["Top Radiant", "Top Dire", "Mid Radiant", "Mid Dire", "Bot Radiant", "Bot Dire"]
    plot_evaluation_results(actual_targets, predicted_outputs, score_names)
    logger.info("Training complete. Model and plots saved.")
    console.print("[green]Training complete. Model and plots saved.[/green]")

@nn_app.command(name="predict", help="Predict lane scores for a given draft string.")
def nn_predict_command(
    draft_string: str = typer.Argument(..., help='Semicolon-separated draft string. E.g., "Radiant Pick: Axe; Dire Pick: Juggernaut;..."'),
    model_file: str = typer.Option(CONFIG['model_weights_path'], "--model-file", help="Path to load model weights from. Overrides config.")
):
    logger.info(f"Attempting prediction for draft: '{draft_string[:50]}...' using model: {model_file if model_file else CONFIG['model_weights_path']}")
    # Load model and hero map
    try:
        _, _, hero_map, num_heroes_in_map = load_and_preprocess_data(CONFIG['csv_output_path'], CONFIG['nn_training_defaults']['batch_size'])
    except SystemExit: # Raised by load_and_preprocess_data if CSV not found or no data
        logger.error("Failed to load hero map for prediction due to CSV data issues. Ensure 'lanes.csv' (or configured CSV) exists and is valid.")
        # console.print is handled by load_and_preprocess_data
        raise typer.Exit(code=1)
    
    nn_params = CONFIG['nn_training_defaults']
    model = DraftLanePredictor(
        num_heroes=num_heroes_in_map,
        hidden_layer_size1=nn_params['hidden_layer_size1'],
        hidden_layer_size2=nn_params['hidden_layer_size2'],
        hidden_layer_size3=nn_params['hidden_layer_size3'],
        hidden_layer_size4=nn_params['hidden_layer_size4'],
        output_layer_size=nn_params['output_layer_size']
    )
    loaded_model = load_model_weights(model, file_path_override=model_file)
    if not loaded_model:
        # load_model_weights logs the error
        raise typer.Exit(code=1)

    # Predict
    scores = predict_draft(loaded_model, draft_string, hero_map, num_heroes_in_map)
    if scores:
        logger.info(f"Prediction successful. Scores: {scores}")
        table = Table(title="Predicted Lane Scores")
        table.add_column("Lane")
        table.add_column("Score", justify="right")
        score_names = ["Top Radiant", "Top Dire", "Mid Radiant", "Mid Dire", "Bot Radiant", "Bot Dire"]
        for name, score_val in zip(score_names, scores): # Renamed score to score_val
            table.add_row(name, f"{score_val:.2f}")
        console.print(table)
    else:
        logger.warning("Prediction did not return scores.")
        # predict_draft logs if parsing fails

if __name__ == "__main__":
    logger.info("Dota2Draft CLI application started.")
    app()
    logger.info("Dota2Draft CLI application finished.")
