import questionary
import typer
from typer.testing import CliRunner
from typing import List

from dota2draft.db import DBManager

# Assuming your main Typer app in dota2draft_cli.py is named 'app'
# Adjust the import if your app object or file name is different
from dota2draft_cli import app as cli_app

db_manager = DBManager()
runner = CliRunner()

def _run_command(command_args: List[str]):
    """Helper to run a command via CliRunner and handle its output gracefully."""
    typer.echo(f"\n> Running command: `{' '.join(command_args)}`")
    result = runner.invoke(cli_app, command_args)

    if result.output:
        typer.echo(result.output.rstrip())

    if result.exit_code != 0 and not isinstance(result.exception, typer.Abort):
        typer.secho(f"--- Command exited with an unexpected error (code: {result.exit_code}) ---", fg=typer.colors.RED, err=True)


# Command Dictionaries

PLAYER_COMMANDS = {
    "Get Player Stats": ["players", "stats"],
    "Set Player Nickname": ["players", "set-nickname"],
    "Remove Player Nickname": ["players", "remove-nickname"],
    "List Player Nicknames": ["players", "list-nicknames"],
    "Back to Main Menu": None
}

HERO_COMMANDS = {
    "Get Hero Stats": ["heroes", "stats"],
    "Set Hero Nickname": ["heroes", "set-nickname"],
    "Remove Hero Nickname": ["heroes", "remove-nickname"],
    "List Hero Nicknames": ["heroes", "list-nicknames"],
    "Back to Main Menu": None,
}

MATCHES_COMMANDS = {
    "Export Match Data": ["matches", "export"],
    "Back to Main Menu": None,
}

LEAGUE_COMMANDS = {
    "List All Leagues": ["leagues", "list"],
    "Search Leagues": ["leagues", "search"],
    "List Downloaded Leagues": ["leagues", "downloaded"],
    "Fetch League Matches": ["leagues", "fetch"],
    "Retry Failed Downloads": ["leagues", "retry-failed"],
    "List Failed Downloads": ["leagues", "list-failed"],
    "Delete League Data": ["leagues", "delete"],
    "Back to Main Menu": None
}

# Menu Handlers

def handle_players_menu():
    """Handles the interactive menu for player commands."""
    while True:
        choice = questionary.select("Players Menu:", choices=list(PLAYER_COMMANDS.keys())).ask()
        if choice is None or choice == "Back to Main Menu": break

        command_args = PLAYER_COMMANDS[choice][:]
        if choice == "Get Player Stats":
            if not (identifier := questionary.text("Enter Player ID or Nickname:").ask()):
                typer.echo("Identifier cannot be empty."); continue
            command_args.append(identifier)
            if league_id := questionary.text("Filter by League ID (optional):").ask(): command_args.extend(["--league-id", league_id])
            if after_date := questionary.text("Filter by date (YYYY-MM-DD, optional):").ask(): command_args.extend(["--after-date", after_date])
        elif choice in ["Set Player Nickname", "Remove Player Nickname"]:
            if not (account_id := questionary.text("Enter Player Account ID:").ask()):
                typer.echo("Account ID cannot be empty."); continue
            if not (nickname := questionary.text(f"Enter nickname to {choice.split(' ')[0].lower()}:").ask()):
                typer.echo("Nickname cannot be empty."); continue
            command_args.extend([account_id, nickname])
        elif choice == "List Player Nicknames":
            if not (identifier := questionary.text("Enter Player ID or Nickname:").ask()):
                typer.echo("Identifier cannot be empty."); continue
            command_args.append(identifier)
        _run_command(command_args)

def handle_heroes_menu():
    """Handles the interactive menu for hero commands."""
    while True:
        choice = questionary.select("Heroes Menu:", choices=list(HERO_COMMANDS.keys())).ask()
        if choice is None or choice == "Back to Main Menu": break

        command_args = HERO_COMMANDS[choice][:]
        if choice == "Get Hero Stats":
            if league_id := questionary.text("Filter by League ID (optional):").ask(): command_args.extend(["--league-id", league_id])
            if after_date := questionary.text("Filter by date (YYYY-MM-DD, optional):").ask(): command_args.extend(["--after-date", after_date])
        elif choice in ["Set Hero Nickname", "Remove Hero Nickname"]:
            if not (identifier := questionary.text("Enter Hero ID, name, or nickname:").ask()):
                typer.echo("Hero identifier cannot be empty."); continue
            if not (nickname := questionary.text(f"Enter nickname to {choice.split(' ')[0].lower()}:").ask()):
                typer.echo("Nickname cannot be empty."); continue
            command_args.extend([identifier, nickname])
        elif choice == "List Hero Nicknames":
            if not (identifier := questionary.text("Enter Hero ID, name, or nickname:").ask()):
                typer.echo("Identifier cannot be empty."); continue
            command_args.append(identifier)
        _run_command(command_args)

def handle_leagues_menu():
    """Handles the interactive menu for league commands."""
    while True:
        choice = questionary.select("Leagues Menu:", choices=list(LEAGUE_COMMANDS.keys())).ask()
        if choice is None or choice == "Back to Main Menu": break

        # Special handling for delete, as it has a custom flow and needs to break out
        if choice == "Delete League Data":
            league_id_str = questionary.text("Enter League ID:").ask()
            if not league_id_str:
                print("Operation cancelled.")
                continue
            
            try:
                league_id = int(league_id_str)
                league_name = db_manager.get_league_name_by_id(league_id)
                if not league_name:
                    print(f"Error: League with ID {league_id} not found in the database.")
                    continue

                confirm_delete = questionary.confirm(
                    f"Are you sure you want to delete all data for league '{league_name}' ({league_id})?"
                ).ask()

                if confirm_delete:
                    _run_command(["leagues", "delete", str(league_id), "--yes"])
                else:
                    print("Deletion cancelled.")
            except ValueError:
                print("Error: Please enter a valid integer for the League ID.")
            
            continue # Restart the menu loop after this special action

        # Default handling for other commands that follow a simple pattern
        command_args = LEAGUE_COMMANDS[choice][:]
        if choice == "Search Leagues":
            if not (keyword := questionary.text("Enter keyword to search for:").ask()):
                typer.echo("Search keyword cannot be empty."); continue
            command_args.append(keyword)
        elif choice in ["Fetch League Matches", "Retry Failed Downloads", "List Failed Downloads"]:
            if not (league_id := questionary.text("Enter League ID:").ask()):
                typer.echo("League ID cannot be empty."); continue
            command_args.append(league_id)
            if choice == "Fetch League Matches":
                if questionary.confirm("Force refresh? (re-download existing matches)").ask():
                    command_args.append("--force-refresh")
        
        _run_command(command_args)

def handle_matches_menu():
    """Handles the interactive menu for match commands."""
    while True:
        choice = questionary.select("Matches Menu:", choices=list(MATCHES_COMMANDS.keys())).ask()
        if choice is None or choice == "Back to Main Menu": break

        command_args = MATCHES_COMMANDS[choice][:]
        if choice == "Export Match Data":
            match_id = questionary.text("Enter the Match ID to export:").ask()
            if not match_id or not match_id.isdigit():
                typer.echo("[red]Invalid Match ID. Please enter a numeric ID.[/red]")
                continue

            output_file = questionary.text("Enter output file path (optional, press Enter to skip):").ask()

            command_args.extend([match_id])
            if output_file:
                command_args.extend(["--out", output_file])

        _run_command(command_args)

def main_interactive_loop():
    """Main loop for the interactive CLI."""
    typer.echo("Welcome to the Dota2Draft Interactive CLI!")
    typer.echo("Use Ctrl+C to exit at any time.")

    main_menu_choices = {
        "Leagues": handle_leagues_menu,
        "Players": handle_players_menu,
        "Heroes": handle_heroes_menu,
        "Matches": handle_matches_menu,
        "Refresh Static Data": lambda: _run_command(["refresh-static"]) if questionary.confirm("Refresh all static data (heroes, leagues, teams)?").ask() else None,
        "Get Match Draft": lambda: _run_command(["get-draft", mid]) if (mid := questionary.text("Enter Match ID:").ask()) else None,
        "Analyze Lanes for Match": lambda: _run_command(["analyze-lanes", mid, "--format", fmt]) if (mid := questionary.text("Enter Match ID:").ask()) and (fmt := questionary.select("Select output format:", choices=["table", "json"]).ask()) else None,
        "Export League Analysis to CSV": lambda: _run_command(["export-analysis", lid] + (["--out", ofile] if (ofile := questionary.text("Output file path (optional):").ask()) else [])) if (lid := questionary.text("Enter League ID:").ask()) else None,
        "Custom SQL Query": lambda: handle_sql_query(),
        "Exit": None
    }

    while True:
        choice = questionary.select(
            "Main Menu - What would you like to do?",
            choices=list(main_menu_choices.keys())
        ).ask()

        if choice is None or choice == "Exit":
            typer.echo("Exiting interactive mode.")
            break
        
        action = main_menu_choices.get(choice)
        if action:
            action()

def handle_sql_query():
    """Handles custom SQL query input and execution."""
    typer.echo("\n[bold cyan]Custom SQL Query Interface[/bold cyan]")
    typer.echo("Execute SELECT or WITH queries safely on the database.")
    typer.echo("Examples:")
    typer.echo("  SELECT COUNT(*) FROM matches")
    typer.echo("  SELECT league_id, COUNT(*) as match_count FROM matches GROUP BY league_id")
    
    while True:
        query = questionary.text("Enter your SQL query (or 'back' to return to main menu):").ask()
        if not query or query.lower() in ['back', 'exit']:
            break
            
        format_choice = questionary.select(
            "Output format:",
            choices=["table", "json", "csv"]
        ).ask()
        
        limit = questionary.text("Row limit (default 100):").ask()
        limit_arg = ["--limit", limit] if limit and limit.isdigit() else []
        
        export_file = questionary.text("Export to file (optional):").ask()
        export_arg = ["--export", export_file] if export_file else []
        
        command_args = ["sql", query, "--format", format_choice] + limit_arg + export_arg
        _run_command(command_args)

if __name__ == "__main__":
    main_interactive_loop()
