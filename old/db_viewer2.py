# db_viewer.py
# Description: A CLI tool to view data stored in the OpenDota SQLite database
# created by league_info.py. This script ONLY interacts with the local database.
# It now includes functionality to set/view custom player names, view raw JSON
# (with an option for interactive visualization and saving to file), 
# get player/hero summaries, DB stats, and Dota 2 lane analysis.

import sqlite3   # For interacting with the SQLite database
import json      # For parsing JSON data stored in the database
from datetime import datetime # For potentially formatting timestamps
from typing import Optional, Dict, Any, List, Tuple # For type hinting
from collections import Counter, defaultdict # For counting hero/player occurrences

import os # For path operations, useful for webbrowser

import typer # For creating the command-line interface
from rich.console import Console # For pretty output in the terminal
from rich.table import Table     # For displaying data in tables
from rich.panel import Panel     # For displaying text in bordered panels
from rich.text import Text       # For styled text
from rich.syntax import Syntax   # For pretty-printing JSON

# --- Visualization imports ---
try:
    import gravis as gv
    import networkx as nx
    import webbrowser 
    VISUALIZATION_LIBRARIES_AVAILABLE = True
except ImportError:
    VISUALIZATION_LIBRARIES_AVAILABLE = False
    gv = None 
    nx = None
    webbrowser = None
# --- End of visualization imports ---

# Initialize Typer app and Rich Console
app = typer.Typer(help="CLI tool to view data from the OpenDota SQLite database (DB interactions only).")
console = Console()

# Define the name of the SQLite database file
DB_NAME = "opendota_league_info.db"

# --- Constants for Lane Analysis ---
LANE_PHASE_END_TIME_SECONDS_SHORT = 8 * 60  # 8 minutes
LANE_PHASE_END_TIME_SECONDS_LONG = 10 * 60 # 10 minutes
LANE_PHASE_TOWER_KILL_TIME_LIMIT = 12 * 60 # 12 minutes for early tower consideration

# Manually created map for hero_id to a simplified name string (used for parsing kill logs)
# Based on the provided gamejson.json. For wider use, this should be more comprehensive.
HERO_ID_TO_NAME_MAP = {
    109: "terrorblade", 87: "disruptor", 11: "nevermore", 123: "hoodwink", 16: "sand_king",
    10: "morphling", 99: "bristleback", 106: "ember_spirit", 93: "slark", 68: "ancient_apparition",
    # Add more hero_ids and their string names as needed from your data
    228: "monkey_king", 92: "weaver", 38: "beastmaster", 128: "snapfire", 198: "bristleback", # from draft
    218: "terrorblade", 276: "arc_warden", 74: "invoker", 240: "lycan", 246: "hoodwink",
    136: "ancient_apparition", 186: "slark", 32: "sand_king", 174: "disruptor", 20: "morphling",
    138: "pangolier", 26: "lion", 34: "tinker", 58: "enchantress", 212: "ember_spirit", 22: "shadow_fiend"
}
# --- End Constants for Lane Analysis ---


def _init_custom_player_names_table(conn: sqlite3.Connection):
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
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row 
        _init_custom_player_names_table(conn) 
        return conn
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite connection error: {e}[/bold red]")
        raise typer.Exit(code=1) 

def get_hero_id_from_name_db(hero_name_input: str, conn: sqlite3.Connection) -> Optional[int]:
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
            for match_info in possible_matches: 
                console.print(f"  - {match_info['name']} (ID: {match_info['hero_id']})")
            return None 
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error searching for hero '{hero_name_input}': {e}[/bold red]")
    return None

def _get_all_heroes_map_from_db(conn: sqlite3.Connection) -> Dict[int, str]:
    cursor = conn.cursor()
    hero_map: Dict[int, str] = {}
    try:
        cursor.execute("SELECT hero_id, localized_name FROM heroes") # Prefer localized_name if available
        rows = cursor.fetchall()
        for row in rows:
            hero_map[row["hero_id"]] = row["localized_name"]
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error loading all heroes: {e}[/bold red]")
        console.print("[yellow]Falling back to internal HERO_ID_TO_NAME_MAP for hero names.[/yellow]")
        return HERO_ID_TO_NAME_MAP # Fallback to manual map
    if not hero_map:
        console.print("[yellow]Warning: Hero map from DB is empty. Hero names may not be displayed correctly.[/yellow]")
        console.print("[yellow]Falling back to internal HERO_ID_TO_NAME_MAP for hero names.[/yellow]")
        return HERO_ID_TO_NAME_MAP # Fallback to manual map
    return hero_map


def _get_all_player_custom_names_map_from_db(conn: sqlite3.Connection) -> Dict[int, str]:
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

# --- Visualization Helper Function (from previous context, ensure it's Gravis 0.1.0 compatible) ---
def _visualize_json_structure_gravis(json_data: Dict[str, Any], item_id: Any, item_type_name: str = "Match") -> bool:
    if not VISUALIZATION_LIBRARIES_AVAILABLE:
        console.print("[bold yellow]Visualization libraries (gravis, networkx) are not installed. Skipping visualization.[/bold yellow]")
        return False
    if not json_data or not isinstance(json_data, (dict, list)):
        console.print("[yellow]Invalid or empty JSON data provided for visualization.[/yellow]")
        return False

    console.print(f"[info]Generating interactive graph for {item_type_name} ID {item_id}...[/info]")
    if gv is not None:
         console.print(f"[info]Using Gravis version: {gv.__version__}[/info]")
    
    graph = nx.DiGraph() 

    def add_nodes_edges_recursive(data_node: Any, parent_node_name: Optional[str] = None):
        default_shape = 'ellipse' 
        if isinstance(data_node, dict):
            for key, value in data_node.items():
                current_node_name = f"{parent_node_name}.{key}" if parent_node_name else str(key)
                value_snippet = str(value); 
                if len(value_snippet) > 150: value_snippet = value_snippet[:147] + "..."
                graph.add_node(current_node_name, title=value_snippet, label=str(key), color='lightblue', size=10, shape=default_shape)
                if parent_node_name: graph.add_edge(parent_node_name, current_node_name)
                if isinstance(value, (dict, list)): add_nodes_edges_recursive(value, current_node_name)
        elif isinstance(data_node, list):
            if not data_node and parent_node_name:
                empty_list_node_name = f"{parent_node_name}.[]"
                graph.add_node(empty_list_node_name, title="Empty List", label="[]", color='lightgrey', size=8, shape='rectangle')
                graph.add_edge(parent_node_name, empty_list_node_name)
                return
            for i, item in enumerate(data_node):
                current_node_name = f"{parent_node_name}[{i}]"
                value_snippet = str(item); 
                if len(value_snippet) > 150: value_snippet = value_snippet[:147] + "..."
                graph.add_node(current_node_name, title=value_snippet, label=f"[{i}]", color='lightgreen', size=10, shape='box')
                if parent_node_name: graph.add_edge(parent_node_name, current_node_name)
                if isinstance(item, (dict, list)): add_nodes_edges_recursive(item, current_node_name)

    root_node_label = f"{item_type_name} ID: {item_id}"
    graph.add_node(root_node_label, label=root_node_label, color='salmon', size=15, shape='diamond') 
    add_nodes_edges_recursive(json_data, root_node_label)

    if not graph.nodes() or (len(graph.nodes()) == 1 and root_node_label in graph):
         console.print("[yellow]Warning: The graph is empty or contains only the root node.[/yellow]")
         if root_node_label in graph: graph.nodes[root_node_label]['title'] = json.dumps(json_data, indent=2)
         else: graph.add_node(root_node_label, label=root_node_label, title=json.dumps(json_data, indent=2), color='salmon', size=15, shape='diamond')

    output_filename_abs = os.path.abspath(f"{item_type_name.lower()}_{item_id}_visualization.html")
    
    try: # --- Attempt 1: Gravis 0.1.0 compatible call ---
        console.print("[info]Attempting Gravis 0.1.0 compatible visualization...[/info]")
        fig = gv.d3(graph, graph_height=800, node_label_data_source='label', 
                    show_menu=True, zoom_factor=0.7, details_min_height=150, details_max_height=300)
        fig.export_html(output_filename_abs, overwrite=True)
        console.print(f"[green]Successfully generated interactive visualization: {output_filename_abs}[/green]")
        if webbrowser: webbrowser.open(f"file://{output_filename_abs}") 
        return True
    except TypeError as te:
        console.print(f"[bold red]Gravis TypeError (Initial Attempt for 0.1.0): {te}[/bold red]")
        console.print("[info]Attempting an extremely simplified visualization (fallback)...[/info]")
        try: # --- Attempt 2: Absolute Bare Minimum Fallback ---
            fig_simple = gv.d3(graph, graph_height=800, node_label_data_source='label')
            fig_simple.export_html(output_filename_abs, overwrite=True)
            console.print(f"[green]Successfully generated extremely simplified visualization: {output_filename_abs}[/green]")
            if webbrowser: webbrowser.open(f"file://{output_filename_abs}")
            return True
        except Exception as e_simple_gravis:
            console.print(f"[bold red]Error during extremely simplified Gravis (fallback) attempt: {e_simple_gravis}[/bold red]")
            return False
    except Exception as e_gravis: 
        console.print(f"[bold red]General error during Gravis visualization: {e_gravis}[/bold red]")
        return False
# --- End of Visualization Helper ---

# --- Lane Analysis Helper Functions ---
def _parse_hero_name_from_log_key(npc_hero_key: Optional[str]) -> str:
    if not npc_hero_key: return "Unknown"
    if npc_hero_key.startswith("npc_dota_hero_"):
        return npc_hero_key.replace("npc_dota_hero_", "")
    return npc_hero_key # Should not happen for hero keys but good fallback

def _get_hero_name_from_id(hero_id: Optional[int], hero_map: Dict[int, str]) -> str:
    if hero_id is None: return "N/A"
    return hero_map.get(hero_id, HERO_ID_TO_NAME_MAP.get(hero_id, f"ID:{hero_id}"))


def _identify_laners(players_data: List[Dict[str, Any]], hero_map: Dict[int, str]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    lanes_assignment = {
        "top": {"radiant": [], "dire": []},
        "mid": {"radiant": [], "dire": []},
        "bot": {"radiant": [], "dire": []},
        "unknown": {"radiant": [], "dire": []} 
    }
    for player in players_data:
        team = "radiant" if player.get("isRadiant", player.get("player_slot", 0) < 5) else "dire"
        assigned_lane_id = player.get("lane") 
        
        lane_name = "unknown"
        if team == "radiant":
            if assigned_lane_id == 1: lane_name = "bot" 
            elif assigned_lane_id == 2: lane_name = "mid"
            elif assigned_lane_id == 3: lane_name = "top" 
        else: # Dire
            if assigned_lane_id == 1: lane_name = "top" 
            elif assigned_lane_id == 2: lane_name = "mid"
            elif assigned_lane_id == 3: lane_name = "bot" 
        
        player['hero_name_parsed'] = _get_hero_name_from_id(player.get('hero_id'), hero_map)
        player['team_str'] = team # Add team string for convenience
        lanes_assignment[lane_name][team].append(player)
    return lanes_assignment

def _get_player_kpis_at_minute(player_data: Dict[str, Any], minute_mark: int, hero_map: Dict[int, str]) -> Dict[str, Any]:
    kpis = {"lh": 0, "dn": 0, "gold": 0, "xp": 0, "gpm": 0, "xpm": 0, "level": 1, "hero": "N/A", "name": "N/A"}
    
    kpis["hero"] = _get_hero_name_from_id(player_data.get('hero_id'), hero_map)
    kpis["name"] = player_data.get("personaname", f"Slot {player_data.get('player_slot', '?')}")

    # Ensure time index is valid. The _t arrays are per-minute, so index = minute.
    time_index = minute_mark 
    
    # Check if times array exists and its length
    times_array = player_data.get("times")
    if not times_array or time_index >= len(times_array):
        # If times_array is missing or too short, use the last available values for _t arrays
        # This is a fallback, ideally data should be present up to the minute_mark
        if player_data.get("lh_t"): kpis["lh"] = player_data["lh_t"][-1]
        if player_data.get("dn_t"): kpis["dn"] = player_data["dn_t"][-1]
        if player_data.get("gold_t"): kpis["gold"] = player_data["gold_t"][-1]
        if player_data.get("xp_t"): kpis["xp"] = player_data["xp_t"][-1]
        # GPM/XPM might be less accurate if we don't have data at exact minute_mark
        if kpis["gold"] > 0 and minute_mark > 0: kpis["gpm"] = round(kpis["gold"] / minute_mark)
        if kpis["xp"] > 0 and minute_mark > 0: kpis["xpm"] = round(kpis["xp"] / minute_mark)
    else:
        # Standard extraction if data is present
        if player_data.get("lh_t") and len(player_data["lh_t"]) > time_index:
            kpis["lh"] = player_data["lh_t"][time_index]
        if player_data.get("dn_t") and len(player_data["dn_t"]) > time_index:
            kpis["dn"] = player_data["dn_t"][time_index]
        if player_data.get("gold_t") and len(player_data["gold_t"]) > time_index:
            kpis["gold"] = player_data["gold_t"][time_index]
            if minute_mark > 0: kpis["gpm"] = round(kpis["gold"] / minute_mark)
        if player_data.get("xp_t") and len(player_data["xp_t"]) > time_index:
            kpis["xp"] = player_data["xp_t"][time_index]
            if minute_mark > 0: kpis["xpm"] = round(kpis["xp"] / minute_mark)

    # Approximate level
    xp_thresholds = [0, 240, 600, 1080, 1680, 2400, 3240, 4200, 5280, 6480, 7800, 9000, 10200, 11400, 12600] # Up to L15
    level = 1
    for i, threshold in enumerate(xp_thresholds):
        if kpis["xp"] >= threshold: level = i + 1
        else: break
    kpis["level"] = level
    return kpis

def _analyze_lane_kill_death_events(
    laner_player_obj: Dict[str, Any], 
    all_players_data: List[Dict[str, Any]], 
    opposing_laner_hero_names: List[str], 
    time_limit_seconds: int
) -> Tuple[int, int]:
    kills_on_opp_laners = 0
    deaths_to_opp_laners = 0
    laner_hero_name = laner_player_obj.get('hero_name_parsed', 'UnknownHero')

    # Kills by this laner on opposing laners
    if laner_player_obj.get("kills_log"):
        for kill_event in laner_player_obj["kills_log"]:
            if kill_event["time"] <= time_limit_seconds:
                victim_hero_name = _parse_hero_name_from_log_key(kill_event.get("key"))
                if victim_hero_name in opposing_laner_hero_names:
                    kills_on_opp_laners += 1
    
    # Deaths of this laner to opposing laners
    # Iterate through all players to find who might have killed this laner
    for potential_killer_obj in all_players_data:
        # Skip if it's the same player or same team
        if potential_killer_obj["player_slot"] == laner_player_obj["player_slot"]:
            continue
        if potential_killer_obj.get("isRadiant") == laner_player_obj.get("isRadiant"):
            continue

        killer_hero_name = potential_killer_obj.get('hero_name_parsed', 'UnknownHero')
        if killer_hero_name in opposing_laner_hero_names: # Killer must be an opposing laner
            if potential_killer_obj.get("kills_log"):
                for kill_event in potential_killer_obj["kills_log"]:
                    if kill_event["time"] <= time_limit_seconds:
                        victim_hero_name = _parse_hero_name_from_log_key(kill_event.get("key"))
                        if victim_hero_name == laner_hero_name:
                            deaths_to_opp_laners += 1
    return kills_on_opp_laners, deaths_to_opp_laners

def _check_early_tower_status(objectives_data: List[Dict[str, Any]], time_limit_seconds: int) -> Dict[str, Dict[str, bool]]:
    tower_status = {
        "top": {"radiant_t1_fallen": False, "dire_t1_fallen": False},
        "mid": {"radiant_t1_fallen": False, "dire_t1_fallen": False},
        "bot": {"radiant_t1_fallen": False, "dire_t1_fallen": False},
    }
    t1_tower_keys = {
        "npc_dota_goodguys_tower1_top": ("top", "radiant_t1_fallen"),
        "npc_dota_badguys_tower1_top": ("top", "dire_t1_fallen"),
        "npc_dota_goodguys_tower1_mid": ("mid", "radiant_t1_fallen"),
        "npc_dota_badguys_tower1_mid": ("mid", "dire_t1_fallen"),
        "npc_dota_goodguys_tower1_bot": ("bot", "radiant_t1_fallen"),
        "npc_dota_badguys_tower1_bot": ("bot", "dire_t1_fallen"),
    }
    if objectives_data:
        for obj_event in objectives_data:
            if obj_event.get("type") == "building_kill" and obj_event.get("time", float('inf')) <= time_limit_seconds:
                building_key = obj_event.get("key")
                if building_key in t1_tower_keys:
                    lane, status_key = t1_tower_keys[building_key]
                    tower_status[lane][status_key] = True
    return tower_status
# --- End Lane Analysis Helper Functions ---


# --- Typer Commands (existing ones omitted for brevity, add new command below) ---
# ... (set-player-name, list-custom-names, etc. - keep them as they are) ...

@app.command(name="show-match-json", help="Displays or visualizes the JSON for a specific match ID. Can also save to file.")
def show_match_json(
    match_id: int = typer.Argument(..., help="The Match ID."),
    visualize: bool = typer.Option( 
        False, 
        "--visualize", "-v", 
        help="Generate an interactive visual map of the JSON structure (opens in browser). Requires gravis and networkx.",
        is_flag=True 
    ),
    output_file: Optional[str] = typer.Option( 
        None,
        "--output-file", "-o",
        help="Path to save the raw JSON data (e.g., match_data.json).",
        show_default=False 
    )
):
    """
    Displays the full stored JSON for a specific match ID.
    Can optionally generate an interactive HTML visualization and/or save the JSON to a file.
    """
    console.print(Panel(f"[bold blue]JSON Data for Match ID {match_id}[/bold blue]", expand=False))
    conn = get_db_connection()
    cursor = conn.cursor()
    row_data: Optional[sqlite3.Row] = None
    try:
        cursor.execute("SELECT data FROM matches WHERE match_id = ?", (match_id,))
        row_data = cursor.fetchone()
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite query error for match JSON: {e}[/bold red]")
        if conn: conn.close()
        raise typer.Exit(code=1)

    if not row_data or not row_data["data"]:
        console.print(f"[yellow]No data found in DB for match ID {match_id}. (Hint: Populate using league_info.py)[/yellow]")
        if conn: conn.close()
        return
    
    json_string_data = row_data["data"]
    match_json_parsed_data: Optional[Dict[str, Any]] = None
    
    try:
        match_json_parsed_data = json.loads(json_string_data)
        
        if output_file:
            try:
                output_file_abs = os.path.abspath(output_file)
                with open(output_file_abs, 'w') as f:
                    json.dump(match_json_parsed_data, f, indent=2)
                console.print(f"[green]Successfully saved JSON data for match {match_id} to: {output_file_abs}[/green]")
            except IOError as e:
                console.print(f"[bold red]Error saving JSON to file '{output_file}': {e}[/bold red]")
        
        if visualize:
            console.print("[info]Visualization requested. Attempting to generate graph...[/info]")
            if not VISUALIZATION_LIBRARIES_AVAILABLE:
                console.print("[bold yellow]Visualization libraries (gravis, networkx) are not installed. Cannot visualize.[/bold yellow]")
                console.print("To install them, run: [code]pip install gravis networkx[/code]")
            else:
                visualization_successful = _visualize_json_structure_gravis(match_json_parsed_data, match_id, "Match")
                if not visualization_successful:
                    console.print("[yellow]Visualization generation failed or was skipped.[/yellow]")
        
        elif not output_file: # Only print to terminal if not visualizing AND not saving
             console.print("\n[info]Displaying JSON in terminal:[/info]")
             syntax = Syntax(json.dumps(match_json_parsed_data, indent=2), "json", theme="material", line_numbers=True)
             console.print(syntax)

    except json.JSONDecodeError:
        console.print(f"[bold red]Error: Stored data for match ID {match_id} is not valid JSON.[/bold red]")
        if output_file: 
            try:
                output_file_abs = os.path.abspath(output_file)
                with open(output_file_abs, 'w') as f:
                    f.write(json_string_data) 
                console.print(f"[yellow]Saved raw (invalid JSON) data for match {match_id} to: {output_file_abs}[/yellow]")
            except IOError as e:
                console.print(f"[bold red]Error saving raw JSON to file '{output_file}': {e}[/bold red]")
        console.print("[info]The raw data was not valid JSON. If you saved it to a file, you can inspect it there.[/info]")

    finally:
        if conn: 
            conn.close()

# --- New Lane Analysis Command ---
@app.command(name="analyze-lanes", help="Analyzes and compares laning phase performance for a specific match ID.")
def analyze_lanes_command(match_id: int = typer.Argument(..., help="The Match ID to analyze.")):
    """
    Fetches match data from the DB and performs a laning phase analysis,
    comparing players in each lane based on KPIs at 8 and 10 minutes.
    """
    console.print(Panel(f"[bold blue]Laning Phase Analysis for Match ID {match_id}[/bold blue]", expand=False))
    conn = get_db_connection()
    cursor = conn.cursor()
    hero_map_from_db = _get_all_heroes_map_from_db(conn) # Use DB heroes map if available
    player_custom_names_map = _get_all_player_custom_names_map_from_db(conn)

    row_data: Optional[sqlite3.Row] = None
    try:
        cursor.execute("SELECT data FROM matches WHERE match_id = ?", (match_id,))
        row_data = cursor.fetchone()
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite query error fetching match data: {e}[/bold red]")
        if conn: conn.close()
        raise typer.Exit(code=1)

    if not row_data or not row_data["data"]:
        console.print(f"[yellow]No data found in DB for match ID {match_id}.[/yellow]")
        if conn: conn.close()
        return

    try:
        match_data = json.loads(row_data["data"])
    except json.JSONDecodeError:
        console.print(f"[bold red]Error: Stored data for match ID {match_id} is not valid JSON.[/bold red]")
        if conn: conn.close()
        return
    
    if conn: conn.close() # Close DB connection after fetching data

    players_data = match_data.get("players", [])
    if not players_data:
        console.print("[yellow]No player data found in the match JSON.[/yellow]")
        return

    # Enrich player data with custom names
    for p in players_data:
        p_acc_id = p.get("account_id")
        if p_acc_id and p_acc_id in player_custom_names_map:
            p["display_name"] = player_custom_names_map[p_acc_id]
        else:
            p["display_name"] = p.get("personaname", f"Slot {p.get('player_slot', '?')}")


    lanes_assignment = _identify_laners(players_data, hero_map_from_db)
    objectives_data = match_data.get("objectives", [])
    early_tower_status = _check_early_tower_status(objectives_data, LANE_PHASE_TOWER_KILL_TIME_LIMIT)

    for lane_name_key, teams_in_lane in lanes_assignment.items():
        if lane_name_key == "unknown": continue # Skip players not clearly in a lane for this analysis

        lane_panel_title = f"Lane Analysis: {lane_name_key.upper()}"
        lane_content = ""
        
        lane_summary_data = {"radiant_total_gold_10m": 0, "dire_total_gold_10m": 0,
                             "radiant_total_xp_10m": 0, "dire_total_xp_10m": 0,
                             "radiant_lane_kills": 0, "dire_lane_kills": 0}

        for team_name, players_in_lane_team in teams_in_lane.items():
            if not players_in_lane_team: continue

            team_color = "cyan" if team_name == "radiant" else "orange3"
            table = Table(title=f"{team_name.capitalize()} - {lane_name_key.upper()} @ 8min / 10min", title_style=f"bold {team_color}", show_header=True, header_style="bold magenta")
            table.add_column("Player", style="dim", min_width=15, overflow="fold")
            table.add_column("Hero", min_width=15)
            table.add_column("Lvl", justify="center")
            table.add_column("LH", justify="center")
            table.add_column("DN", justify="center")
            table.add_column("Gold", justify="center")
            table.add_column("GPM", justify="center")
            table.add_column("XP", justify="center")
            table.add_column("XPM", justify="center")
            table.add_column("Kills (vs Lane)", justify="center", header_style="bold green") # Kills on opposing laners by 10min
            table.add_column("Deaths (to Lane)", justify="center", header_style="bold red")  # Deaths to opposing laners by 10min
            
            opposing_team_str = "dire" if team_name == "radiant" else "radiant"
            opposing_laners_in_this_lane_objs = lanes_assignment[lane_name_key][opposing_team_str]
            opposing_laner_hero_names = [p.get('hero_name_parsed', '') for p in opposing_laners_in_this_lane_objs]


            for player_obj in players_in_lane_team:
                kpis_8m = _get_player_kpis_at_minute(player_obj, 8, hero_map_from_db)
                kpis_10m = _get_player_kpis_at_minute(player_obj, 10, hero_map_from_db)

                kills, deaths = _analyze_lane_kill_death_events(player_obj, players_data, opposing_laner_hero_names, LANE_PHASE_END_TIME_SECONDS_LONG)
                
                if team_name == "radiant":
                    lane_summary_data["radiant_total_gold_10m"] += kpis_10m["gold"]
                    lane_summary_data["radiant_total_xp_10m"] += kpis_10m["xp"]
                    lane_summary_data["radiant_lane_kills"] += kills
                else:
                    lane_summary_data["dire_total_gold_10m"] += kpis_10m["gold"]
                    lane_summary_data["dire_total_xp_10m"] += kpis_10m["xp"]
                    lane_summary_data["dire_lane_kills"] += kills


                table.add_row(
                    player_obj.get("display_name", player_obj.get("personaname", f"Slot {player_obj['player_slot']}")),
                    kpis_10m["hero"], # Show hero once
                    f"{kpis_8m['level']} / {kpis_10m['level']}",
                    f"{kpis_8m['lh']} / {kpis_10m['lh']}",
                    f"{kpis_8m['dn']} / {kpis_10m['dn']}",
                    f"{kpis_8m['gold']} / {kpis_10m['gold']}",
                    f"{kpis_8m['gpm']} / {kpis_10m['gpm']}",
                    f"{kpis_8m['xp']} / {kpis_10m['xp']}",
                    f"{kpis_8m['xpm']} / {kpis_10m['xpm']}",
                    str(kills),
                    str(deaths)
                )
            console.print(table)
        
        # Lane Summary & Heuristic
        gold_diff = lane_summary_data["radiant_total_gold_10m"] - lane_summary_data["dire_total_gold_10m"]
        xp_diff = lane_summary_data["radiant_total_xp_10m"] - lane_summary_data["dire_total_xp_10m"]
        kill_diff = lane_summary_data["radiant_lane_kills"] - lane_summary_data["dire_lane_kills"] # Net kills for Radiant in lane

        lane_verdict = "Even Lane"
        radiant_score = 0
        dire_score = 0

        if gold_diff > 750 * len(teams_in_lane.get("radiant",[])): radiant_score += 2
        elif gold_diff < -750 * len(teams_in_lane.get("dire",[])): dire_score += 2
        elif abs(gold_diff) > 300 * max(1, len(teams_in_lane.get("radiant",[]))): # Minor lead
            if gold_diff > 0: radiant_score +=1
            else: dire_score +=1
            
        if xp_diff > 1000 * len(teams_in_lane.get("radiant",[])): radiant_score += 2
        elif xp_diff < -1000 * len(teams_in_lane.get("dire",[])): dire_score += 2
        elif abs(xp_diff) > 500 * max(1, len(teams_in_lane.get("radiant",[]))): # Minor lead
            if xp_diff > 0: radiant_score +=1
            else: dire_score +=1

        if kill_diff >= 2: radiant_score += 2
        elif kill_diff <= -2: dire_score += 2
        elif kill_diff == 1 : radiant_score +=1
        elif kill_diff == -1 : dire_score +=1
            
        if early_tower_status[lane_name_key]["dire_t1_fallen"]: radiant_score += 3
        if early_tower_status[lane_name_key]["radiant_t1_fallen"]: dire_score += 3

        if radiant_score > dire_score + 1: lane_verdict = f"[bold cyan]Radiant Appears Ahead ({radiant_score} vs {dire_score})[/bold cyan]"
        elif dire_score > radiant_score + 1: lane_verdict = f"[bold orange3]Dire Appears Ahead ({dire_score} vs {radiant_score})[/bold orange3]"
        else: lane_verdict = f"Even Lane ({radiant_score} vs {dire_score})"

        summary_text = Text(f"\nGold Adv @10m: {gold_diff:+} G | XP Adv @10m: {xp_diff:+} XP | Net Kills (R-D): {kill_diff:+}\n"
                            f"Radiant T1 Fallen by {LANE_PHASE_TOWER_KILL_TIME_LIMIT//60}min: {early_tower_status[lane_name_key]['radiant_t1_fallen']} | "
                            f"Dire T1 Fallen by {LANE_PHASE_TOWER_KILL_TIME_LIMIT//60}min: {early_tower_status[lane_name_key]['dire_t1_fallen']}\n"
                            f"Verdict: {lane_verdict}\n")
        console.print(summary_text)
        console.print("-" * 80)

# --- End of New Lane Analysis Command ---


# ... (Other existing Typer commands: db-stats, player-summary, hero-summary) ...
# Ensure these are still present in your final script.
# For brevity, I'm omitting them here but they should be included from your db_viewer2.py.
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
        
        cursor.execute("SELECT MIN(fetched_at), MAX(fetched_at) FROM matches WHERE fetched_at IS NOT NULL AND fetched_at != ''")
        match_dates_row = cursor.fetchone()
        if match_dates_row and match_dates_row[0] is not None and match_dates_row[1] is not None:
            stats["Matches Fetched Between"] = f"{match_dates_row[0]} and {match_dates_row[1]}"
        else:
            stats["Matches Fetched Between"] = "N/A (No valid dates or no matches)"
            
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite query error for DB stats: {e}[/bold red]")
        if conn: conn.close(); 
        raise typer.Exit(code=1)
    finally:
        if conn: conn.close()

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
    sql_like_term_for_query: str

    try:
        target_account_id = int(player_identifier)
        custom_name = player_custom_names_map.get(target_account_id)
        display_player_term = f"{custom_name} (ID: {target_account_id})" if custom_name else f"ID: {target_account_id}"
        sql_like_term_for_query = f'%"account_id": {target_account_id}%'
        console.print(f"[info]Interpreted player identifier as Account ID: {target_account_id}[/info]")
    except ValueError: 
        resolved_by_custom_name = False
        for acc_id, cust_name in player_custom_names_map.items():
            if player_identifier.lower() == cust_name.lower():
                target_account_id = acc_id
                display_player_term = f"{cust_name} (resolved to ID: {acc_id})"
                sql_like_term_for_query = f'%"account_id": {target_account_id}%'
                console.print(f"[info]Resolved Custom Name '{player_identifier}' to Account ID: {target_account_id}[/info]")
                resolved_by_custom_name = True
                break
        if not resolved_by_custom_name:
            search_by_personaname_fallback = True
            sql_like_term_for_query = f'%"personaname": "%{player_identifier}%"%'
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
        if conn: conn.close(); 
        return
    
    if not match_rows:
        console.print(f"[yellow]No matches found in DB potentially involving '{display_player_term}' based on initial search.[/yellow]"); 
        if conn: conn.close(); 
        return

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
            
            player_is_radiant_val = player_in_this_match_info.get("isRadiant")
            if player_is_radiant_val is None and "player_slot" in player_in_this_match_info:
                player_is_radiant_val = player_in_this_match_info["player_slot"] < 100

            radiant_won = match_data.get("radiant_win")
            player_won_this_match = False
            if radiant_won is not None and player_is_radiant_val is not None: 
                if radiant_won is True and player_is_radiant_val: 
                    wins += 1
                    player_won_this_match = True
                elif radiant_won is False and not player_is_radiant_val: 
                    wins += 1
                    player_won_this_match = True
            
            if hero_id: 
                if hero_id not in hero_performance:
                    hero_performance[hero_id] = {"picks": 0, "wins": 0}
                hero_performance[hero_id]["picks"] += 1
                if player_won_this_match:
                    hero_performance[hero_id]["wins"] += 1
    if conn: conn.close()

    if matches_played == 0:
        console.print(f"[yellow]No confirmed matches found for player '{display_player_term}' after detailed check.[/yellow]"); return

    win_rate = (wins / matches_played * 100) if matches_played > 0 else 0
    
    console.print(f"\n--- Summary for Player: {display_player_term} ---")
    console.print(f"Total Matches in DB where player was found: {matches_played}")
    console.print(f"Wins (in those matches): {wins}")
    console.print(f"Win Rate: {win_rate:.2f}%")

    if hero_performance:
        console.print("\n[bold]Most Played Heroes (Top 5):[/bold]")
        sorted_hero_performance = sorted(hero_performance.items(), key=lambda item: item[1]["picks"], reverse=True)
        
        top_heroes_table = Table(show_header=True, header_style="bold cyan") 
        top_heroes_table.add_column("Hero", style="green")
        top_heroes_table.add_column("Picks", style="magenta", justify="center")
        top_heroes_table.add_column("Win Rate (%)", style="blue", justify="center") 

        for hero_id_val, stats in sorted_hero_performance[:5]: 
            hero_display_name = hero_map.get(hero_id_val, f"ID: {hero_id_val}")
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
            console.print(f"[bold red]Hero name '{hero_identifier}' not found or ambiguous.[/bold red]"); 
            if conn: conn.close(); 
            return
        console.print(f"[info]Resolved hero name '{hero_identifier}' to ID: {target_hero_id}[/info]")

    hero_map = _get_all_heroes_map_from_db(conn) 
    player_custom_names_map = _get_all_player_custom_names_map_from_db(conn)
    actual_hero_name = hero_map.get(target_hero_id, f"ID: {target_hero_id}") 

    query = "SELECT data FROM matches WHERE data LIKE ? ORDER BY fetched_at DESC"
    params = (f'%"hero_id": {target_hero_id}%',) 

    total_hero_picks = 0
    total_hero_wins = 0
    player_performance_with_hero: Dict[int, Dict[str, int]] = {} 

    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        match_rows = cursor.fetchall()
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error fetching matches for hero summary: {e}[/bold red]"); 
        if conn: conn.close(); 
        return
    
    if not match_rows:
        console.print(f"[yellow]No matches found in DB potentially featuring hero '{actual_hero_name}' (ID: {target_hero_id}) based on initial search.[/yellow]"); 
        if conn: conn.close(); 
        return

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
            
            player_is_radiant_val = player_who_picked_hero_info.get("isRadiant")
            if player_is_radiant_val is None and "player_slot" in player_who_picked_hero_info:
                 player_is_radiant_val = player_who_picked_hero_info["player_slot"] < 100
            
            radiant_won = match_data.get("radiant_win")
            player_won_this_match_with_hero = False

            if radiant_won is not None and player_is_radiant_val is not None:
                if radiant_won is True and player_is_radiant_val: 
                    total_hero_wins += 1
                    player_won_this_match_with_hero = True
                elif radiant_won is False and not player_is_radiant_val: 
                    total_hero_wins += 1
                    player_won_this_match_with_hero = True
            
            if account_id: 
                if account_id not in player_performance_with_hero:
                    player_performance_with_hero[account_id] = {"picks": 0, "wins": 0}
                player_performance_with_hero[account_id]["picks"] += 1
                if player_won_this_match_with_hero:
                    player_performance_with_hero[account_id]["wins"] += 1
    if conn: conn.close()

    if total_hero_picks == 0:
        console.print(f"[yellow]Hero '{actual_hero_name}' (ID: {target_hero_id}) was not confirmed picked in any stored matches after detailed check.[/yellow]"); return

    overall_hero_win_rate = (total_hero_wins / total_hero_picks * 100) if total_hero_picks > 0 else 0

    console.print(f"\n--- Summary for Hero: {actual_hero_name} (ID: {target_hero_id}) ---")
    console.print(f"Total Picks in DB: {total_hero_picks}")
    console.print(f"Wins (when hero was played): {total_hero_wins}")
    console.print(f"Win Rate (when hero was played): {overall_hero_win_rate:.2f}%")

    if player_performance_with_hero:
        console.print("\n[bold]Most Frequent Players (Top 5):[/bold]")
        sorted_player_performance = sorted(
            player_performance_with_hero.items(), 
            key=lambda item: (item[1]["picks"], (item[1]['wins'] / item[1]['picks'] * 100) if item[1]['picks'] > 0 else 0), 
            reverse=True
        )

        top_players_table = Table(show_header=True, header_style="bold cyan") 
        top_players_table.add_column("Player (Name/ID)", style="green")
        top_players_table.add_column("Picks", style="magenta", justify="center")
        top_players_table.add_column("Win Rate (%) with Hero", style="blue", justify="center") 

        for acc_id_val, stats in sorted_player_performance[:5]: 
            player_display_name = player_custom_names_map.get(acc_id_val, f"ID: {acc_id_val}")
            player_hero_win_rate = (stats['wins'] / stats['picks'] * 100) if stats['picks'] > 0 else 0
            top_players_table.add_row(player_display_name, str(stats['picks']), f"{player_hero_win_rate:.2f}%")
        console.print(top_players_table)
    else:
        console.print("No specific player performance data available for this hero (likely due to missing account IDs in match data).")


if __name__ == "__main__":
    # This check ensures that the main app() call is only made when the script is executed directly.
    # It prevents app() from being called if this script is imported as a module elsewhere.
    console.print(Panel("[bold green]OpenDota Database Viewer CLI (db_viewer.py)[/bold green]", expand=False))
    app()

