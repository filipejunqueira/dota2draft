# db_viewer.py
# Description: A CLI tool to view data stored in the OpenDota SQLite database
# created by league_info.py. This script ONLY interacts with the local database.
# It now includes functionality to set/view custom player names, view raw JSON,
# get player/hero summaries (with win rates), and DB stats.

import sqlite3   # For interacting with the SQLite database
import json      # For parsing JSON data stored in the database
from datetime import datetime # For potentially formatting timestamps
from typing import Optional, Dict, Any, List # For type hinting
from collections import Counter # For counting hero/player occurrences

import typer # For creating the command-line interface
from rich.console import Console # For pretty output in the terminal
from rich.table import Table     # For displaying data in tables
from rich.panel import Panel     # For displaying text in bordered panels
from rich.text import Text       # For styled text
from rich.syntax import Syntax   # For pretty-printing JSON

# Initialize Typer app and Rich Console
app = typer.Typer(help="CLI tool to view data from the OpenDota SQLite database (DB interactions only).")
console = Console()

# Define the name of the SQLite database file (should be the same as in league_info.py)
DB_NAME = "opendota_league_info.db"

def _init_custom_player_names_table(conn: sqlite3.Connection):
    """
    Ensures the 'player_custom_names' table exists in the database.
    Args:
        conn: An active SQLite connection.
    """
    cursor = conn.cursor()
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_custom_names (
            account_id INTEGER PRIMARY KEY,
            custom_name TEXT NOT NULL UNIQUE,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error creating 'player_custom_names' table: {e}[/bold red]")


def get_db_connection() -> sqlite3.Connection:
    """
    Establishes and returns a connection to the SQLite database.
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row 
        _init_custom_player_names_table(conn) 
        return conn
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite connection error: {e}[/bold red]")
        raise typer.Exit(code=1) 

def get_hero_id_from_name_db(hero_name_input: str, conn: sqlite3.Connection) -> Optional[int]:
    """
    Queries the local 'heroes' table to find a hero_id for a given hero name (case-insensitive).
    """
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT hero_id, name FROM heroes WHERE LOWER(name) = LOWER(?)", (hero_name_input,))
        row = cursor.fetchone()
        if row:
            return int(row["hero_id"])
        cursor.execute("SELECT hero_id, name FROM heroes WHERE name LIKE ?", (f'%{hero_name_input}%',))
        possible_matches = cursor.fetchall()
        if len(possible_matches) == 1:
            console.print(f"[dim]Found unique partial match for hero: '{possible_matches[0]['name']}' (ID: {possible_matches[0]['hero_id']})[/dim]")
            return int(possible_matches[0]["hero_id"])
        elif len(possible_matches) > 1:
            console.print(f"[yellow]Multiple heroes found for '{hero_name_input}'. Please be more specific:[/yellow]")
            for match in possible_matches:
                console.print(f"  - {match['name']} (ID: {match['hero_id']})")
            return None 
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error searching for hero '{hero_name_input}': {e}[/bold red]")
    return None

def _get_all_heroes_map_from_db(conn: sqlite3.Connection) -> Dict[int, str]:
    """
    Helper function to load all hero IDs and names from the DB into a dictionary.
    """
    cursor = conn.cursor()
    hero_map: Dict[int, str] = {}
    try:
        cursor.execute("SELECT hero_id, name FROM heroes")
        rows = cursor.fetchall()
        for row in rows:
            hero_map[row["hero_id"]] = row["name"]
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error loading all heroes: {e}[/bold red]")
    if not hero_map:
        console.print("[yellow]Warning: Hero map from DB is empty. Hero names may not be displayed.[/yellow]")
    return hero_map

def _get_all_player_custom_names_map_from_db(conn: sqlite3.Connection) -> Dict[int, str]:
    """
    Loads all custom player names from the 'player_custom_names' table.
    """
    cursor = conn.cursor()
    custom_names_map: Dict[int, str] = {}
    try:
        cursor.execute("SELECT account_id, custom_name FROM player_custom_names")
        rows = cursor.fetchall()
        for row in rows:
            custom_names_map[row["account_id"]] = row["custom_name"]
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error loading custom player names: {e}[/bold red]")
    return custom_names_map

# --- New Commands & Updated Commands ---

@app.command(name="set-player-name", help="Assign a custom name to a player's Account ID.")
def set_player_name(
    account_id: int = typer.Argument(..., help="The player's unique Account ID."),
    name: str = typer.Argument(..., help="The custom name to assign (e.g., 'CCnC', 'Miracle-').")
):
    console.print(Panel(f"[bold blue]Setting Custom Name for Account ID {account_id}[/bold blue]", expand=False))
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO player_custom_names (account_id, custom_name, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (account_id, name)
        )
        conn.commit()
        console.print(f"[green]Successfully set custom name for Account ID {account_id} to '{name}'.[/green]")
    except sqlite3.IntegrityError: 
        console.print(f"[bold red]Error: The custom name '{name}' might already be assigned to another Account ID or another error occurred.[/bold red]")
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error setting custom name for Account ID {account_id}: {e}[/bold red]")
    finally:
        conn.close()

@app.command(name="list-custom-names", help="Lists all custom player names stored in the database.")
def list_custom_names():
    console.print(Panel("[bold blue]Listing All Custom Player Names[/bold blue]", expand=False))
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT account_id, custom_name, updated_at FROM player_custom_names ORDER BY custom_name ASC")
        names_rows = cursor.fetchall()
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite query error for custom names: {e}[/bold red]")
        conn.close(); raise typer.Exit(code=1)
    finally:
        conn.close()
    
    if not names_rows:
        console.print("[yellow]No custom player names found in the database.[/yellow]"); return

    table = Table(title="Stored Custom Player Names", show_header=True, header_style="bold magenta")
    table.add_column("Account ID", style="dim", width=15, justify="center")
    table.add_column("Custom Name", min_width=20)
    table.add_column("Last Updated (UTC)", style="dim", min_width=20, justify="center")
    for row in names_rows:
        table.add_row(str(row["account_id"]), row["custom_name"], row["updated_at"])
    console.print(table)
    console.print(f"\nTotal custom names in DB: {len(names_rows)}")

@app.command(name="delete-player-name", help="Deletes a custom player name mapping.")
def delete_player_name(
    identifier: str = typer.Argument(..., help="The Account ID or Custom Name to delete.")
):
    console.print(Panel(f"[bold blue]Deleting Custom Name for Identifier '{identifier}'[/bold blue]", expand=False))
    conn = get_db_connection()
    cursor = conn.cursor()
    account_id_to_delete: Optional[int] = None

    try: 
        account_id_to_delete = int(identifier)
    except ValueError: 
        try:
            cursor.execute("SELECT account_id FROM player_custom_names WHERE LOWER(custom_name) = LOWER(?)", (identifier,))
            row = cursor.fetchone()
            if row:
                account_id_to_delete = row["account_id"]
            else:
                console.print(f"[yellow]No custom name '{identifier}' found.[/yellow]")
                conn.close(); return
        except sqlite3.Error as e:
            console.print(f"[bold red]SQLite error finding custom name '{identifier}': {e}[/bold red]")
            conn.close(); return
            
    if account_id_to_delete is None: 
        console.print(f"[yellow]Could not resolve '{identifier}' to an Account ID for deletion.[/yellow]")
        conn.close(); return

    try:
        cursor.execute("DELETE FROM player_custom_names WHERE account_id = ?", (account_id_to_delete,))
        conn.commit()
        if cursor.rowcount > 0:
            console.print(f"[green]Successfully deleted custom name mapping for Account ID {account_id_to_delete}.[/green]")
        else:
            console.print(f"[yellow]No custom name found for Account ID {account_id_to_delete} to delete (it might have been a direct ID input without a custom name set).[/yellow]")
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error deleting custom name for Account ID {account_id_to_delete}: {e}[/bold red]")
    finally:
        conn.close()


@app.command(name="list-heroes", help="Lists all heroes stored in the database.")
def list_heroes():
    console.print(Panel("[bold blue]Listing All Stored Heroes[/bold blue]", expand=False))
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT hero_id, name, fetched_at FROM heroes ORDER BY hero_id ASC")
        heroes_rows = cursor.fetchall() 
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite query error for heroes: {e}[/bold red]")
        conn.close(); raise typer.Exit(code=1)
    finally:
        conn.close() 
    if not heroes_rows:
        console.print("[yellow]No heroes found in the database. (Hint: Populate using league_info.py)[/yellow]"); return
    table = Table(title="Stored Heroes", show_header=True, header_style="bold magenta")
    table.add_column("Hero ID", style="dim", width=10, justify="center")
    table.add_column("Name", min_width=20)
    table.add_column("Fetched At (UTC)", style="dim", min_width=20, justify="center")
    for hero in heroes_rows:
        table.add_row(str(hero["hero_id"]), hero["name"], hero["fetched_at"])
    console.print(table)
    console.print(f"\nTotal heroes in DB: {len(heroes_rows)}")

@app.command(name="list-matches", help="Lists all matches stored in the database with summary details and players (Custom Name/Account ID & Hero).")
def list_matches(
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit the number of matches displayed (most recent first)."),
    search_team: Optional[str] = typer.Option(None, "--team", "-t", help="Filter matches by a team name (case-insensitive search in Radiant or Dire).")
):
    console.print(Panel("[bold blue]Listing All Stored Matches with Player Details[/bold blue]", expand=False))
    conn = get_db_connection()
    cursor = conn.cursor()
    hero_map = _get_all_heroes_map_from_db(conn)
    player_custom_names_map = _get_all_player_custom_names_map_from_db(conn)

    query = "SELECT match_id, data, fetched_at FROM matches ORDER BY fetched_at DESC"
    params = ()
    if search_team:
        query = "SELECT match_id, data, fetched_at FROM matches WHERE data LIKE ? ORDER BY fetched_at DESC"
        params = (f'%"{search_team}"%',) 
    if limit is not None and limit > 0:
        query += f" LIMIT ?"
        params += (limit,)
    try:
        cursor.execute(query, params)
        match_rows = cursor.fetchall() 
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite query error for matches: {e}[/bold red]")
        conn.close(); raise typer.Exit(code=1)
    finally:
        conn.close() 
    if not match_rows:
        if search_team: console.print(f"[yellow]No matches found in DB potentially matching team '{search_team}' via LIKE search.[/yellow]")
        else: console.print("[yellow]No matches found in the database. (Hint: Populate using league_info.py)[/yellow]")
        return
    table = Table(title="Stored Matches with Player Details", show_header=True, header_style="bold magenta")
    table.add_column("Match ID", style="dim", width=12, justify="center")
    table.add_column("Radiant Team", style="cyan", min_width=20)
    table.add_column("Dire Team", style="orange3", min_width=20)
    table.add_column("Score (R-D)", justify="center", width=12)
    table.add_column("Winner", min_width=20, justify="center")
    table.add_column("Radiant Players (Name/ID, Hero)", style="cyan", min_width=45, overflow="fold") 
    table.add_column("Dire Players (Name/ID, Hero)", style="orange3", min_width=45, overflow="fold")
    table.add_column("Fetched At (UTC)", style="dim", min_width=20, justify="center")

    matches_displayed_count = 0
    for row in match_rows:
        match_id = row["match_id"]; fetched_at = row["fetched_at"]
        try:
            match_data: Dict[str, Any] = json.loads(row["data"])
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: Could not parse JSON for match ID {match_id} from DB. Skipping.[/yellow]"); continue
        r_team_name = (match_data.get('radiant_name') or (isinstance(match_data.get('radiant_team'), dict) and (match_data['radiant_team'].get('name') or match_data['radiant_team'].get('tag'))) or "Radiant")
        d_team_name = (match_data.get('dire_name') or (isinstance(match_data.get('dire_team'), dict) and (match_data['dire_team'].get('name') or match_data['dire_team'].get('tag'))) or "Dire")
        if search_team: 
            if not (search_team.lower() in r_team_name.lower() or search_team.lower() in d_team_name.lower()):
                continue
        r_score = str(match_data.get('radiant_score', '-')); d_score = str(match_data.get('dire_score', '-'))
        winner = "N/A"
        if match_data.get('radiant_win') is True: winner = f"[bold cyan]{r_team_name}[/bold cyan]"
        elif match_data.get('radiant_win') is False: winner = f"[bold orange3]{d_team_name}[/bold orange3]"
        radiant_players_details_list = []
        dire_players_details_list = []
        players_data: List[Dict[str, Any]] = match_data.get("players", [])
        for p_info in players_data:
            account_id_val = p_info.get("account_id") 
            player_identifier = player_custom_names_map.get(account_id_val, str(account_id_val) if account_id_val is not None else p_info.get("personaname", "N/A"))
            hero_id = p_info.get("hero_id")
            hero_name = hero_map.get(hero_id, f"ID:{hero_id}") if hero_id else "N/A"
            player_detail_str = f"{player_identifier} ({hero_name})"
            if p_info.get("isRadiant", p_info.get("player_slot", -1) < 5 if p_info.get("player_slot") is not None else False):
                radiant_players_details_list.append(player_detail_str)
            else:
                dire_players_details_list.append(player_detail_str)
        radiant_players_display_str = "\n".join(radiant_players_details_list) if radiant_players_details_list else "N/A"
        dire_players_display_str = "\n".join(dire_players_details_list) if dire_players_details_list else "N/A"
        table.add_row(str(match_id), r_team_name, d_team_name, f"{r_score}-{d_score}", winner, radiant_players_display_str, dire_players_display_str, fetched_at)
        matches_displayed_count +=1
    if matches_displayed_count > 0:
        console.print(table)
        console.print(f"\nTotal matches displayed: {matches_displayed_count}")
        if limit and matches_displayed_count >= limit : console.print(f"Showing up to {limit} matches. Use --limit to change.")
        elif search_team: console.print(f"Showing matches filtered by team: '{search_team}'")
    elif search_team: 
        console.print(f"[yellow]No matches found where Radiant or Dire team name contains '{search_team}' after parsing JSON.[/yellow]")


@app.command(name="find-player-hero", help="Finds matches where a player played a specific hero (or all heroes if hero name is omitted).")
def find_player_hero_matches(
    player_identifier: str = typer.Argument(..., help="Player's Account ID, assigned Custom Name, or current Persona Name."),
    hero_name_input: Optional[str] = typer.Argument(None, help="Optional: The name of the hero. If omitted, lists all matches for the player."),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit the number of matches displayed.")
):
    conn = get_db_connection()
    target_hero_id: Optional[int] = None
    hero_search_active = False
    hero_map = _get_all_heroes_map_from_db(conn) 
    player_custom_names_map = _get_all_player_custom_names_map_from_db(conn)

    if hero_name_input: 
        hero_search_active = True
        target_hero_id = get_hero_id_from_name_db(hero_name_input, conn) 
        if target_hero_id is None:
            console.print(f"[bold red]Hero '{hero_name_input}' not found. Try 'list-heroes'.[/bold red]")
            conn.close(); raise typer.Exit(code=1) 
        console.print(f"[info]Targeting Hero ID: {target_hero_id} for '{hero_name_input}'[/info]")
        panel_title = f"[bold blue]Player '{player_identifier}' on Hero '{hero_name_input}'[/bold blue]"
        table_title_suffix = f" on Hero '{hero_name_input}' (ID: {target_hero_id})"
    else:
        panel_title = f"[bold blue]All matches for Player '{player_identifier}'[/bold blue]"
        table_title_suffix = "" 
        console.print(f"[info]Listing all matches for player identifier '{player_identifier}'[/info]")
    console.print(Panel(panel_title, expand=False))
    
    search_target_account_id: Optional[int] = None 
    search_by_personaname_fallback = False
    display_search_term = player_identifier 
    try:
        search_target_account_id = int(player_identifier)
        custom_name_for_id = player_custom_names_map.get(search_target_account_id)
        display_search_term = f"{custom_name_for_id} (ID: {search_target_account_id})" if custom_name_for_id else f"ID: {search_target_account_id}"
        console.print(f"[info]Searching by Account ID: {search_target_account_id}[/info]")
    except ValueError:
        found_in_custom = False
        for acc_id, cust_name in player_custom_names_map.items():
            if player_identifier.lower() == cust_name.lower():
                search_target_account_id = acc_id
                display_search_term = f"{cust_name} (resolved to ID: {acc_id})"
                console.print(f"[info]Searching by Custom Name '{player_identifier}', resolved to Account ID: {search_target_account_id}[/info]")
                found_in_custom = True; break
        if not found_in_custom:
            search_by_personaname_fallback = True
            display_search_term = f"Persona Name containing '{player_identifier}'"
            console.print(f"[info]Searching by Persona Name containing: '{player_identifier}'[/info]")

    query = "SELECT match_id, data, fetched_at FROM matches WHERE data LIKE ? ORDER BY fetched_at DESC"
    params = (f'%{player_identifier}%',) 
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        match_rows = cursor.fetchall() 
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite query error: {e}[/bold red]"); conn.close(); raise typer.Exit(code=1) 
    if not match_rows:
        console.print(f"[yellow]No matches in DB potentially involving '{player_identifier}'.[/yellow]"); conn.close(); return

    found_matches_details = []
    for row in match_rows:
        try:
            match_data: Dict[str, Any] = json.loads(row["data"]) 
        except json.JSONDecodeError: continue 
        players_list: List[Dict[str, Any]] = match_data.get("players", [])
        player_found_in_match = False 
        hero_played_by_target_player_id: Optional[int] = None
        for p_info in players_list:
            player_matches_identifier = False
            if search_target_account_id is not None: 
                if p_info.get("account_id") == search_target_account_id: player_matches_identifier = True
            elif search_by_personaname_fallback: 
                current_player_name = p_info.get("personaname")
                if current_player_name and isinstance(current_player_name, str) and player_identifier.lower() in current_player_name.lower():
                    player_matches_identifier = True
            if player_matches_identifier:
                player_found_in_match = True 
                hero_played_by_target_player_id = p_info.get("hero_id")
                if hero_search_active: 
                    if hero_played_by_target_player_id == target_hero_id: break 
                else: break 
        if player_found_in_match and (not hero_search_active or hero_played_by_target_player_id == target_hero_id):
            r_team = (match_data.get('radiant_name') or (isinstance(match_data.get('radiant_team'), dict) and (match_data['radiant_team'].get('name') or match_data['radiant_team'].get('tag'))) or "Radiant")
            d_team = (match_data.get('dire_name') or (isinstance(match_data.get('dire_team'), dict) and (match_data['dire_team'].get('name') or match_data['dire_team'].get('tag'))) or "Dire")
            r_score = str(match_data.get('radiant_score', '-')); d_score = str(match_data.get('dire_score', '-'))
            winner = "N/A"
            if match_data.get('radiant_win') is True: winner = f"[bold cyan]{r_team}[/bold cyan]"
            elif match_data.get('radiant_win') is False: winner = f"[bold orange3]{d_team}[/bold orange3]"
            player_hero_name_display = hero_map.get(hero_played_by_target_player_id, f"ID:{hero_played_by_target_player_id}") if hero_played_by_target_player_id else "N/A"
            found_matches_details.append({"match_id": str(row["match_id"]), "radiant_name": r_team, "dire_name": d_team, "score": f"{r_score}-{d_score}", "winner": winner, "fetched_at": row["fetched_at"], "player_hero": player_hero_name_display})
            if limit and len(found_matches_details) >= limit: break 
    conn.close() 
    if not found_matches_details:
        if hero_search_active: console.print(f"[yellow]No matches found where player '{display_search_term}' played hero '{hero_name_input}' (ID: {target_hero_id}).[/yellow]")
        else: console.print(f"[yellow]No matches found for player '{display_search_term}'.[/yellow]")
        return
    table_title = f"Matches: Player '{display_search_term}'{table_title_suffix}"
    table = Table(title=table_title, show_header=True, header_style="bold magenta")
    table.add_column("Match ID", style="dim", width=12, justify="center")
    table.add_column("Player's Hero", min_width=20) 
    table.add_column("Radiant Team", style="cyan", min_width=20)
    table.add_column("Dire Team", style="orange3", min_width=20)
    table.add_column("Score (R-D)", justify="center", width=12)
    table.add_column("Winner", min_width=20, justify="center")
    table.add_column("Fetched At (UTC)", style="dim", min_width=20, justify="center")
    for detail in found_matches_details:
        table.add_row(detail["match_id"], detail["player_hero"], detail["radiant_name"], detail["dire_name"], detail["score"], detail["winner"], detail["fetched_at"])
    console.print(table)
    console.print(f"\nFound {len(found_matches_details)} match(es) meeting the criteria.")
    if limit and len(found_matches_details) >= limit: console.print(f"Showing up to {limit} matches. Use --limit to change.")

@app.command(name="show-match-json", help="Displays the full stored JSON for a specific match ID.")
def show_match_json(match_id: int = typer.Argument(..., help="The Match ID.")):
    console.print(Panel(f"[bold blue]Raw JSON for Match ID {match_id}[/bold blue]", expand=False))
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT data FROM matches WHERE match_id = ?", (match_id,))
        row = cursor.fetchone()
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite query error for match JSON: {e}[/bold red]")
        conn.close(); raise typer.Exit(code=1)
    finally:
        conn.close()
    
    if not row or not row["data"]:
        console.print(f"[yellow]No data found in DB for match ID {match_id}. (Hint: Populate using league_info.py)[/yellow]"); return
    
    try:
        match_json_data = json.loads(row["data"]) 
        syntax = Syntax(json.dumps(match_json_data, indent=2), "json", theme="material", line_numbers=True)
        console.print(syntax)
    except json.JSONDecodeError:
        console.print(f"[bold red]Error: Stored data for match ID {match_id} is not valid JSON.[/bold red]")
        console.print("Raw data from DB:")
        console.print(row["data"]) 

@app.command(name="db-stats", help="Shows basic statistics about the database content.")
def db_stats():
    console.print(Panel("[bold blue]Database Statistics[/bold blue]", expand=False))
    conn = get_db_connection()
    cursor = conn.cursor()
    stats = {}
    try:
        cursor.execute("SELECT COUNT(*) FROM matches")
        stats["Total Matches"] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM heroes")
        stats["Total Heroes"] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM player_custom_names")
        stats["Total Custom Player Names"] = cursor.fetchone()[0]
        cursor.execute("SELECT MIN(fetched_at), MAX(fetched_at) FROM matches WHERE fetched_at IS NOT NULL")
        match_dates = cursor.fetchone()
        stats["Matches Fetched Between"] = f"{match_dates[0]} and {match_dates[1]}" if match_dates and match_dates[0] else "N/A"
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite query error for DB stats: {e}[/bold red]")
        conn.close(); raise typer.Exit(code=1)
    finally:
        conn.close()

    table = Table(title="Database Overview", show_lines=True)
    table.add_column("Statistic", style="bold magenta")
    table.add_column("Value", style="green")
    for key, value in stats.items():
        table.add_row(key, str(value))
    console.print(table)

@app.command(name="player-summary", help="Shows a summary for a player (matches, top heroes, win rate).")
def player_summary(
    player_identifier: str = typer.Argument(..., help="Player's Account ID, Custom Name, or Persona Name.")
):
    console.print(Panel(f"[bold blue]Summary for Player '{player_identifier}'[/bold blue]", expand=False))
    conn = get_db_connection()
    player_custom_names_map = _get_all_player_custom_names_map_from_db(conn)
    hero_map = _get_all_heroes_map_from_db(conn)

    target_account_id: Optional[int] = None
    search_by_personaname_fallback = False
    display_player_term = player_identifier
    # Corrected SQL LIKE term logic
    sql_like_term_for_query = f'%{player_identifier}%' # Broadest default for personaname

    try:
        target_account_id = int(player_identifier)
        custom_name = player_custom_names_map.get(target_account_id)
        display_player_term = f"{custom_name} (ID: {target_account_id})" if custom_name else f"ID: {target_account_id}"
        # If it's an ID, search for the ID string. This is more reliable than specific JSON key matching.
        sql_like_term_for_query = f'%{str(target_account_id)}%' 
        console.print(f"[info]Interpreted player identifier as Account ID: {target_account_id}[/info]")
    except ValueError: # Input is not a number, so it's a name
        resolved_by_custom_name = False
        for acc_id, cust_name in player_custom_names_map.items():
            if player_identifier.lower() == cust_name.lower():
                target_account_id = acc_id
                display_player_term = f"{cust_name} (resolved to ID: {acc_id})"
                sql_like_term_for_query = f'%{str(target_account_id)}%' # Use resolved account_id string for LIKE
                console.print(f"[info]Resolved Custom Name '{player_identifier}' to Account ID: {target_account_id}[/info]")
                resolved_by_custom_name = True
                break
        if not resolved_by_custom_name:
            search_by_personaname_fallback = True
            # sql_like_term_for_query remains f'%{player_identifier}%' (the input personaname)
            console.print(f"[info]Searching by Persona Name containing: '{player_identifier}'[/info]")

    query = "SELECT data FROM matches WHERE data LIKE ? ORDER BY fetched_at DESC"
    params = (sql_like_term_for_query,) 
    
    matches_played = 0
    wins = 0
    hero_performance: Dict[int, Dict[str, int]] = {} 

    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        match_rows = cursor.fetchall()
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error fetching matches for player summary: {e}[/bold red]")
        conn.close(); return
    
    if not match_rows:
        console.print(f"[yellow]No matches found in DB potentially involving '{display_player_term}' based on initial search.[/yellow]"); conn.close(); return

    for row in match_rows:
        try:
            match_data: Dict[str, Any] = json.loads(row["data"])
        except json.JSONDecodeError: continue

        player_in_this_match_info = None 
        for p_info in match_data.get("players", []):
            current_account_id = p_info.get("account_id")
            current_personaname = p_info.get("personaname")

            player_matches_criteria = False
            if target_account_id is not None: 
                if current_account_id == target_account_id:
                    player_matches_criteria = True
            elif search_by_personaname_fallback: 
                if current_personaname and isinstance(current_personaname, str) and \
                   player_identifier.lower() in current_personaname.lower():
                    player_matches_criteria = True
            
            if player_matches_criteria:
                player_in_this_match_info = p_info
                break 
        
        if player_in_this_match_info:
            matches_played += 1
            hero_id = player_in_this_match_info.get("hero_id")
            
            player_is_radiant = player_in_this_match_info.get("isRadiant", 
                                                              player_in_this_match_info.get("player_slot", -1) < 5 if player_in_this_match_info.get("player_slot") is not None else False)
            radiant_won = match_data.get("radiant_win")
            player_won_this_match = False
            if radiant_won is not None:
                if radiant_won is True and player_is_radiant: 
                    wins += 1
                    player_won_this_match = True
                elif radiant_won is False and not player_is_radiant: 
                    wins += 1
                    player_won_this_match = True
            
            if hero_id: 
                if hero_id not in hero_performance:
                    hero_performance[hero_id] = {"picks": 0, "wins": 0}
                hero_performance[hero_id]["picks"] += 1
                if player_won_this_match:
                    hero_performance[hero_id]["wins"] += 1
    conn.close()

    if matches_played == 0:
        console.print(f"[yellow]No confirmed matches found for player '{display_player_term}' after detailed check.[/yellow]"); return

    win_rate = (wins / matches_played * 100) if matches_played > 0 else 0
    
    console.print(f"\n--- Summary for Player: {display_player_term} ---")
    console.print(f"Total Matches in DB: {matches_played}")
    console.print(f"Wins: {wins}")
    console.print(f"Win Rate: {win_rate:.2f}%")

    if hero_performance:
        console.print("\n[bold]Most Played Heroes:[/bold]")
        sorted_hero_performance = sorted(hero_performance.items(), key=lambda item: item[1]["picks"], reverse=True)
        
        top_heroes_table = Table(show_header=True, header_style="bold cyan") 
        top_heroes_table.add_column("Hero", style="green")
        top_heroes_table.add_column("Picks", style="magenta", justify="center")
        top_heroes_table.add_column("Win Rate (%)", style="blue", justify="center") 

        for hero_id, stats in sorted_hero_performance[:5]: 
            hero_display_name = hero_map.get(hero_id, f"ID: {hero_id}")
            hero_win_rate = (stats['wins'] / stats['picks'] * 100) if stats['picks'] > 0 else 0
            top_heroes_table.add_row(hero_display_name, str(stats['picks']), f"{hero_win_rate:.2f}%")
        console.print(top_heroes_table)
    else:
        console.print("No hero pick data available for this player in the stored matches.")


@app.command(name="hero-summary", help="Shows a summary for a hero (pick count, win rate, top players with win rates).")
def hero_summary(
    hero_identifier: str = typer.Argument(..., help="The name or ID of the hero.") 
):
    console.print(Panel(f"[bold blue]Summary for Hero '{hero_identifier}'[/bold blue]", expand=False))
    conn = get_db_connection()
    
    target_hero_id: Optional[int] = None
    try: 
        target_hero_id = int(hero_identifier)
        console.print(f"[info]Interpreted hero identifier as Hero ID: {target_hero_id}[/info]")
    except ValueError: 
        target_hero_id = get_hero_id_from_name_db(hero_identifier, conn)
        if target_hero_id is None:
            console.print(f"[bold red]Hero name '{hero_identifier}' not found or ambiguous.[/bold red]"); conn.close(); return
        console.print(f"[info]Resolved hero name '{hero_identifier}' to ID: {target_hero_id}[/info]")

    hero_map = _get_all_heroes_map_from_db(conn) 
    player_custom_names_map = _get_all_player_custom_names_map_from_db(conn)
    actual_hero_name = hero_map.get(target_hero_id, f"ID: {target_hero_id}") 

    query = "SELECT data FROM matches WHERE data LIKE ? ORDER BY fetched_at DESC"
    # Corrected SQL LIKE term for hero_id
    params = (f'%{str(target_hero_id)}%',) 

    total_hero_picks = 0
    total_hero_wins = 0
    player_performance_with_hero: Dict[int, Dict[str, int]] = {}

    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        match_rows = cursor.fetchall()
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error fetching matches for hero summary: {e}[/bold red]"); conn.close(); return
    
    if not match_rows:
        console.print(f"[yellow]No matches found in DB potentially featuring hero '{actual_hero_name}' (ID: {target_hero_id}) based on initial search.[/yellow]"); conn.close(); return

    for row in match_rows:
        try:
            match_data: Dict[str, Any] = json.loads(row["data"])
        except json.JSONDecodeError: continue

        player_who_picked_hero_info = None 

        for p_info in match_data.get("players", []):
            if p_info.get("hero_id") == target_hero_id:
                player_who_picked_hero_info = p_info
                break 
        
        if player_who_picked_hero_info:
            total_hero_picks += 1
            account_id = player_who_picked_hero_info.get("account_id")
            
            player_is_radiant = player_who_picked_hero_info.get("isRadiant", 
                                                                player_who_picked_hero_info.get("player_slot", -1) < 5 if player_who_picked_hero_info.get("player_slot") is not None else False)
            radiant_won = match_data.get("radiant_win")
            player_won_this_match_with_hero = False

            if radiant_won is not None:
                if radiant_won is True and player_is_radiant: 
                    total_hero_wins += 1
                    player_won_this_match_with_hero = True
                elif radiant_won is False and not player_is_radiant: 
                    total_hero_wins += 1
                    player_won_this_match_with_hero = True
            
            if account_id: 
                if account_id not in player_performance_with_hero:
                    player_performance_with_hero[account_id] = {"picks": 0, "wins": 0}
                player_performance_with_hero[account_id]["picks"] += 1
                if player_won_this_match_with_hero:
                    player_performance_with_hero[account_id]["wins"] += 1
    conn.close()

    if total_hero_picks == 0:
        console.print(f"[yellow]Hero '{actual_hero_name}' (ID: {target_hero_id}) was not confirmed picked in any stored matches after detailed check.[/yellow]"); return

    overall_hero_win_rate = (total_hero_wins / total_hero_picks * 100) if total_hero_picks > 0 else 0

    console.print(f"\n--- Summary for Hero: {actual_hero_name} (ID: {target_hero_id}) ---")
    console.print(f"Total Picks in DB: {total_hero_picks}")
    console.print(f"Wins (when hero was played): {total_hero_wins}")
    console.print(f"Win Rate (when hero was played): {overall_hero_win_rate:.2f}%")

    if player_performance_with_hero:
        console.print("\n[bold]Most Frequent Players:[/bold]")
        sorted_player_performance = sorted(player_performance_with_hero.items(), key=lambda item: item[1]["picks"], reverse=True)

        top_players_table = Table(show_header=True, header_style="bold cyan") 
        top_players_table.add_column("Player (Name/ID)", style="green")
        top_players_table.add_column("Picks", style="magenta", justify="center")
        top_players_table.add_column("Win Rate (%) with Hero", style="blue", justify="center") 

        for acc_id, stats in sorted_player_performance[:5]: 
            player_display_name = player_custom_names_map.get(acc_id, f"ID: {acc_id}")
            player_hero_win_rate = (stats['wins'] / stats['picks'] * 100) if stats['picks'] > 0 else 0
            top_players_table.add_row(player_display_name, str(stats['picks']), f"{player_hero_win_rate:.2f}%")
        console.print(top_players_table)


if __name__ == "__main__":
    console.print(Panel("[bold green]OpenDota Database Viewer CLI (db_viewer.py)[/bold green]", expand=False))
    app()

