# merged_db_viewer.py
# Description: A comprehensive CLI tool to view and analyze data stored in the
# OpenDota SQLite database created by league_info.py. This script ONLY interacts
# with the local database.
# Features include:
# - Setting/viewing custom player names.
# - Viewing raw match JSON with optional interactive visualization and file output.
# - Detailed player and hero summaries with win rates.
# - Database statistics.
# - Dota 2 laning phase analysis for specific matches.
# - Listing heroes, matches, and finding player-hero specific matches.
# - Exporting all match IDs to CSV.
# - Batch lane analysis from a list of match IDs, outputting to CSV.

import sqlite3   # For interacting with the SQLite database
import json      # For parsing JSON data stored in the database
import os        # For path operations, useful for webbrowser and file output
import csv       # For CSV file operations
from datetime import datetime # For potentially formatting timestamps
from typing import Optional, Dict, Any, List, Tuple # For type hinting
from collections import Counter, defaultdict # For counting hero/player occurrences

import typer # For creating the command-line interface
from rich.console import Console # For pretty output in the terminal
from rich.table import Table     # For displaying data in tables
from rich.panel import Panel     # For displaying text in bordered panels
from rich.text import Text       # For styled text
from rich.syntax import Syntax   # For pretty-printing JSON
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn # For progress bars

# --- Visualization imports (optional) ---
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
app = typer.Typer(
    help="CLI tool to view and analyze data from the OpenDota SQLite database. Interacts with the local DB only."
)
console = Console()

# Define the name of the SQLite database file
DB_NAME = "opendota_league_info.db"

# --- Constants for Lane Analysis ---
LANE_PHASE_END_TIME_SECONDS_SHORT = 8 * 60  # 8 minutes
LANE_PHASE_END_TIME_SECONDS_LONG = 10 * 60 # 10 minutes
LANE_PHASE_TOWER_KILL_TIME_LIMIT = 12 * 60 # 12 minutes for early tower consideration
# --- End Constants for Lane Analysis ---

# --- Database and General Helper Functions ---
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
    Initializes player_custom_names table if not present.
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
    Searches the 'name' column (internal Dota 2 name, e.g., 'npc_dota_hero_antimage').
    Attempts exact match first, then partial match.
    """
    cursor = conn.cursor()
    try:
        # Try exact match on 'name'
        cursor.execute("SELECT hero_id, name FROM heroes WHERE LOWER(name) = LOWER(?)", (hero_name_input,))
        row = cursor.fetchone()
        if row:
            return int(row["hero_id"])

        processed_hero_name_input = hero_name_input.replace("npc_dota_hero_", "") 
        
        cursor.execute("SELECT hero_id, name FROM heroes WHERE name LIKE ? OR name LIKE ?",
                       (f'%{processed_hero_name_input}%', f'%npc_dota_hero_{processed_hero_name_input}%'))
        possible_matches = cursor.fetchall()
        
        if len(possible_matches) == 1:
            matched_name = possible_matches[0]['name']
            console.print(f"[dim]Found unique partial match for hero: '{matched_name}' (ID: {possible_matches[0]['hero_id']})[/dim]")
            return int(possible_matches[0]["hero_id"])
        elif len(possible_matches) > 1:
            console.print(f"[yellow]Multiple heroes found for '{hero_name_input}'. Please be more specific. Matches found based on internal name:[/yellow]")
            for match_info in possible_matches:
                console.print(f"  - {match_info['name']} (ID: {match_info['hero_id']})")
            return None
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error searching for hero '{hero_name_input}': {e}[/bold red]")
    return None


def _get_all_heroes_map_from_db(conn: sqlite3.Connection) -> Dict[int, str]:
    """
    Helper function to load all hero IDs and their 'name' (internal game name)
    from the 'heroes' table in the DB into a dictionary.
    Example: {1: "npc_dota_hero_antimage"}
    """
    cursor = conn.cursor()
    hero_map: Dict[int, str] = {}
    try:
        cursor.execute("SELECT hero_id, name FROM heroes")
        rows = cursor.fetchall()
        for row in rows:
            hero_map[row["hero_id"]] = row["name"] 
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error loading hero data from 'heroes' table: {e}[/bold red]")
        console.print("[yellow]Hero names may not be displayed correctly. Ensure 'heroes' table exists and is populated.[/yellow]")
        return {} 

    if not hero_map:
        console.print("[yellow]Warning: Hero map from DB is empty. Hero names may not be displayed correctly.[/yellow]")
        console.print("[dim]This might happen if the 'heroes' table in the database is empty.[/dim]")
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

# --- Visualization Helper Function ---
def _visualize_json_structure_gravis(json_data: Dict[str, Any], item_id: Any, item_type_name: str = "Match") -> bool:
    """
    Generates an interactive HTML graph of the JSON structure using Gravis and NetworkX.
    Saves the graph to an HTML file and attempts to open it in a web browser.
    """
    if not VISUALIZATION_LIBRARIES_AVAILABLE:
        console.print("[bold yellow]Visualization libraries (gravis, networkx) are not installed. Skipping visualization.[/bold yellow]")
        console.print("To install them, run: [code]pip install gravis networkx[/code]")
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
                value_snippet = str(value)
                if len(value_snippet) > 150: value_snippet = value_snippet[:147] + "..."
                graph.add_node(current_node_name, title=value_snippet, label=str(key), color='lightblue', size=10, shape=default_shape)
                if parent_node_name: graph.add_edge(parent_node_name, current_node_name)
                if isinstance(value, (dict, list)): add_nodes_edges_recursive(value, current_node_name)
        elif isinstance(data_node, list):
            if not data_node and parent_node_name: 
                empty_list_node_name = f"{parent_node_name}.[]"
                graph.add_node(empty_list_node_name, title="Empty List", label="[]", color='lightgrey', size=8, shape='rectangle')
                if parent_node_name: graph.add_edge(parent_node_name, empty_list_node_name)
                return
            for i, item in enumerate(data_node):
                current_node_name = f"{parent_node_name}[{i}]"
                value_snippet = str(item)
                if len(value_snippet) > 150: value_snippet = value_snippet[:147] + "..."
                graph.add_node(current_node_name, title=value_snippet, label=f"[{i}]", color='lightgreen', size=10, shape='box')
                if parent_node_name: graph.add_edge(parent_node_name, current_node_name)
                if isinstance(item, (dict, list)): add_nodes_edges_recursive(item, current_node_name)

    root_node_label = f"{item_type_name} ID: {item_id}"
    graph.add_node(root_node_label, label=root_node_label, color='salmon', size=15, shape='diamond')
    add_nodes_edges_recursive(json_data, root_node_label)

    if not graph.nodes() or (len(graph.nodes()) == 1 and root_node_label in graph.nodes()):
         console.print("[yellow]Warning: The graph is empty or contains only the root node. Visualization might not be useful.[/yellow]")
         if root_node_label in graph.nodes():
             graph.nodes[root_node_label]['title'] = json.dumps(json_data, indent=2)
         else: 
             graph.add_node(root_node_label, label=root_node_label, title=json.dumps(json_data, indent=2), color='salmon', size=15, shape='diamond')

    output_filename_abs = os.path.abspath(f"{item_type_name.lower()}_{item_id}_visualization.html")

    try:
        fig = gv.d3(graph, graph_height=800, node_label_data_source='label',
                    show_menu=True, zoom_factor=0.7, details_min_height=150, details_max_height=300,
                    use_edge_size_normalization=True, edge_size_data_source='weight', 
                    use_node_size_normalization=True, node_size_data_source='size')
        fig.export_html(output_filename_abs, overwrite=True)
        console.print(f"[green]Successfully generated interactive visualization: {output_filename_abs}[/green]")
        if webbrowser:
            try:
                webbrowser.open(f"file://{output_filename_abs}")
            except webbrowser.Error as wb_error:
                console.print(f"[yellow]Could not open visualization in browser: {wb_error}. Please open the file manually.[/yellow]")
        return True
    except TypeError as te: 
        console.print(f"[bold red]Gravis TypeError (likely version incompatibility): {te}[/bold red]")
        console.print("[info]Attempting a more basic Gravis visualization call...[/info]")
        try:
            fig_simple = gv.d3(graph, graph_height=800, node_label_data_source='label')
            fig_simple.export_html(output_filename_abs, overwrite=True)
            console.print(f"[green]Successfully generated basic interactive visualization: {output_filename_abs}[/green]")
            if webbrowser:
                try:
                    webbrowser.open(f"file://{output_filename_abs}")
                except webbrowser.Error as wb_error:
                    console.print(f"[yellow]Could not open visualization in browser: {wb_error}. Please open the file manually.[/yellow]")
            return True
        except Exception as e_simple_gravis:
            console.print(f"[bold red]Error during basic Gravis (fallback) attempt: {e_simple_gravis}[/bold red]")
            return False
    except Exception as e_gravis:
        console.print(f"[bold red]General error during Gravis visualization: {e_gravis}[/bold red]")
        return False

# --- Lane Analysis Core Logic and Helper Functions ---
def _parse_hero_name_from_log_key(npc_hero_key: Optional[str]) -> str:
    """
    Extracts simplified hero name from an NPC string like 'npc_dota_hero_bloodseeker',
    returning 'bloodseeker'.
    """
    if not npc_hero_key: return "Unknown"
    if npc_hero_key.startswith("npc_dota_hero_"):
        return npc_hero_key.replace("npc_dota_hero_", "")
    return npc_hero_key 

def _get_hero_display_name_from_id(hero_id: Optional[int], hero_map: Dict[int, str]) -> str:
    """
    Gets a displayable hero name from ID using the provided hero_map.
    The hero_map contains internal names (e.g., "npc_dota_hero_antimage").
    This function will attempt to simplify it (e.g., "antimage").
    """
    if hero_id is None: return "N/A"
    internal_name = hero_map.get(hero_id)
    if internal_name:
        return _parse_hero_name_from_log_key(internal_name) 
    return f"ID:{hero_id}" 


def _identify_laners(players_data: List[Dict[str, Any]], hero_map: Dict[int, str]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Identifies players in each lane based on 'lane' field in player data.
    Enriches player objects with 'hero_name_parsed' (simplified name) and 'team_str'.
    """
    lanes_assignment: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "top": {"radiant": [], "dire": []},
        "mid": {"radiant": [], "dire": []},
        "bot": {"radiant": [], "dire": []},
        "unknown": {"radiant": [], "dire": []} 
    }
    for player in players_data:
        is_radiant = player.get("isRadiant")
        if is_radiant is None: 
            player_slot = player.get("player_slot")
            if player_slot is not None:
                is_radiant = player_slot < 100 
            else: 
                is_radiant = True 
                # console.print(f"[yellow]Warning: Could not determine team for player {player.get('personaname', 'Unknown')}. Defaulting to Radiant for lane assignment.[/yellow]") # Keep console prints for interactive commands

        team = "radiant" if is_radiant else "dire"
        assigned_lane_id = player.get("lane") 

        lane_name = "unknown"
        if assigned_lane_id is not None:
            if team == "radiant":
                if assigned_lane_id == 1: lane_name = "bot"
                elif assigned_lane_id == 2: lane_name = "mid"
                elif assigned_lane_id == 3: lane_name = "top"
            else: # Dire
                if assigned_lane_id == 1: lane_name = "top"
                elif assigned_lane_id == 2: lane_name = "mid"
                elif assigned_lane_id == 3: lane_name = "bot"

        player['hero_name_parsed'] = _get_hero_display_name_from_id(player.get('hero_id'), hero_map)
        player['team_str'] = team
        lanes_assignment[lane_name][team].append(player)
    return lanes_assignment

def _get_player_kpis_at_minute(player_data: Dict[str, Any], minute_mark: int, hero_map: Dict[int, str]) -> Dict[str, Any]:
    """
    Extracts Key Performance Indicators (KPIs) for a player at a specific minute mark.
    """
    kpis = {"lh": 0, "dn": 0, "gold": 0, "xp": 0, "gpm": 0, "xpm": 0, "level": 1, "hero": "N/A", "name": "N/A"}

    kpis["hero"] = _get_hero_display_name_from_id(player_data.get('hero_id'), hero_map)
    kpis["name"] = player_data.get("display_name", player_data.get("personaname", f"Slot {player_data.get('player_slot', '?')}"))

    time_index = minute_mark 

    def get_t_value(arr_name: str, default_val: int = 0) -> int:
        arr = player_data.get(arr_name)
        if arr and isinstance(arr, list):
            if time_index < len(arr):
                return arr[time_index]
            elif arr: 
                return arr[-1]
        return default_val

    kpis["lh"] = get_t_value("lh_t")
    kpis["dn"] = get_t_value("dn_t")
    kpis["gold"] = get_t_value("gold_t")
    kpis["xp"] = get_t_value("xp_t")

    if minute_mark > 0:
        kpis["gpm"] = round(kpis["gold"] / minute_mark) if kpis["gold"] else 0
        kpis["xpm"] = round(kpis["xp"] / minute_mark) if kpis["xp"] else 0
    
    level_t_array = player_data.get("level_t")
    if level_t_array and isinstance(level_t_array, list) and time_index < len(level_t_array):
        kpis["level"] = level_t_array[time_index]
    else: 
        xp_thresholds = [0, 240, 600, 1080, 1680, 2400, 3240, 4200, 5280, 6480, 7800, 9000, 10200, 11400, 12600] 
        level = 1
        current_xp = kpis["xp"]
        for i, threshold in enumerate(xp_thresholds): 
            if current_xp >= threshold:
                level = i + 1 
            else:
                break
        kpis["level"] = level
    return kpis

def _analyze_lane_kill_death_events(
    laner_player_obj: Dict[str, Any],
    all_players_data: List[Dict[str, Any]],
    opposing_laner_hero_names: List[str],
    time_limit_seconds: int
) -> Tuple[int, int]:
    """
    Analyzes kills by and deaths of the laner against opposing laners.
    """
    kills_on_opp_laners = 0
    deaths_to_opp_laners = 0
    laner_hero_name = laner_player_obj.get('hero_name_parsed', 'UnknownHero')
    if laner_hero_name == 'UnknownHero' or not laner_hero_name: 
        return 0,0

    if laner_player_obj.get("kills_log"):
        for kill_event in laner_player_obj["kills_log"]:
            if kill_event.get("time", float('inf')) <= time_limit_seconds:
                victim_hero_name_key = kill_event.get("key") 
                victim_hero_name_simplified = _parse_hero_name_from_log_key(victim_hero_name_key)
                if victim_hero_name_simplified in opposing_laner_hero_names:
                    kills_on_opp_laners += 1

    for potential_killer_obj in all_players_data:
        if potential_killer_obj.get("player_slot") == laner_player_obj.get("player_slot"):
            continue
        if potential_killer_obj.get("isRadiant") == laner_player_obj.get("isRadiant"):
            continue

        killer_hero_name_parsed = potential_killer_obj.get('hero_name_parsed', 'UnknownHero') 
        if killer_hero_name_parsed in opposing_laner_hero_names: 
            if potential_killer_obj.get("kills_log"):
                for kill_event in potential_killer_obj["kills_log"]:
                    if kill_event.get("time", float('inf')) <= time_limit_seconds:
                        victim_hero_name_key = kill_event.get("key") 
                        victim_hero_name_simplified = _parse_hero_name_from_log_key(victim_hero_name_key)
                        if victim_hero_name_simplified == laner_hero_name: 
                            deaths_to_opp_laners += 1
    return kills_on_opp_laners, deaths_to_opp_laners

def _check_early_tower_status(objectives_data: Optional[List[Dict[str, Any]]], time_limit_seconds: int) -> Dict[str, Dict[str, bool]]:
    """
    Checks T1 tower status by a time limit.
    """
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
                    lane, status_key_part = t1_tower_keys[building_key]
                    tower_status[lane][status_key_part] = True
    return tower_status

def _extract_draft_order_str(match_data: Dict[str, Any], hero_map: Dict[int, str]) -> str:
    """
    Extracts and formats the pick/ban draft order from match data into a string.
    """
    picks_bans = match_data.get("picks_bans")
    if not picks_bans or not isinstance(picks_bans, list):
        return "N/A"

    draft_entries = []
    # Sort by order if not already sorted (OpenDota usually provides it sorted)
    # sorted_picks_bans = sorted(picks_bans, key=lambda x: x.get("order", float('inf')))
    
    for entry in picks_bans:
        is_pick = entry.get("is_pick")
        hero_id = entry.get("hero_id")
        team_val = entry.get("team") # 0 for Radiant, 1 for Dire
        # order = entry.get("order") # For ensuring sequence if needed

        hero_name = _get_hero_display_name_from_id(hero_id, hero_map)
        action = "Pick" if is_pick else "Ban"
        team_str = "Radiant" if team_val == 0 else "Dire"
        
        draft_entries.append(f"{team_str} {action}: {hero_name}")
        
    return "; ".join(draft_entries) if draft_entries else "N/A"


def _perform_core_lane_analysis(match_id: int, conn: sqlite3.Connection, hero_map: Dict[int, str], player_custom_names_map: Dict[int, str]) -> Optional[Dict[str, Any]]:
    """
    Performs core laning phase analysis for a match and returns structured data.
    This function is designed to be called by both interactive and batch commands.
    It does not print to console directly but returns data for the caller to handle.
    """
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT data FROM matches WHERE match_id = ?", (match_id,))
        row_data = cursor.fetchone()
    except sqlite3.Error as e:
        return {"error": f"SQLite query error for match {match_id}: {e}"}

    if not row_data or not row_data["data"]:
        return {"error": f"No data found in DB for match ID {match_id}."}

    try:
        match_data = json.loads(row_data["data"])
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON for match ID {match_id}."}

    players_data = match_data.get("players", [])
    if not players_data:
        return {"error": f"No player data in JSON for match ID {match_id}."}

    # Enrich player data with display names and ensure isRadiant
    for p_idx, p_info in enumerate(players_data):
        p_acc_id = p_info.get("account_id")
        display_name = p_info.get("personaname", f"Slot {p_info.get('player_slot', p_idx)}")
        if p_acc_id and p_acc_id in player_custom_names_map:
            display_name = player_custom_names_map[p_acc_id]
        players_data[p_idx]["display_name"] = display_name
        if "isRadiant" not in p_info and "player_slot" in p_info:
             players_data[p_idx]["isRadiant"] = p_info["player_slot"] < 100
    
    draft_order_str = _extract_draft_order_str(match_data, hero_map)
    lanes_assignment = _identify_laners(players_data, hero_map)
    objectives_data = match_data.get("objectives")
    early_tower_status = _check_early_tower_status(objectives_data, LANE_PHASE_TOWER_KILL_TIME_LIMIT)

    analysis_results = {
        "match_id": match_id,
        "draft_order": draft_order_str,
        "lanes": {}, # To store "top", "mid", "bot" results
        "error": None
    }

    for lane_name_key in ["top", "mid", "bot"]: # Focus on main lanes
        teams_in_lane = lanes_assignment.get(lane_name_key)
        if not teams_in_lane or (not teams_in_lane["radiant"] and not teams_in_lane["dire"]):
            analysis_results["lanes"][lane_name_key] = {
                "radiant_score": 0, "dire_score": 0, 
                "verdict_text": "No players in lane",
                "summary": {} # No detailed summary if no players
            }
            continue

        lane_summary_data = {
            "radiant_total_gold_10m": 0, "dire_total_gold_10m": 0,
            "radiant_total_xp_10m": 0, "dire_total_xp_10m": 0,
            "radiant_lane_kills": 0, "dire_lane_kills": 0
        }
        
        # Calculate KPIs and K/D for each player in the lane
        for team_name_str, players_in_lane_for_team in teams_in_lane.items():
            if not players_in_lane_for_team: continue

            opposing_team_str_val = "dire" if team_name_str == "radiant" else "radiant"
            opposing_laners_in_this_lane_objs = lanes_assignment[lane_name_key].get(opposing_team_str_val, [])
            opposing_laner_hero_names_list = [p.get('hero_name_parsed', '') for p in opposing_laners_in_this_lane_objs if p.get('hero_name_parsed')]

            for player_obj_data in players_in_lane_for_team:
                kpis_10m = _get_player_kpis_at_minute(player_obj_data, 10, hero_map) # Focus on 10 min for summary
                kills, _ = _analyze_lane_kill_death_events(player_obj_data, players_data, opposing_laner_hero_names_list, LANE_PHASE_END_TIME_SECONDS_LONG)

                if team_name_str == "radiant":
                    lane_summary_data["radiant_total_gold_10m"] += kpis_10m["gold"]
                    lane_summary_data["radiant_total_xp_10m"] += kpis_10m["xp"]
                    lane_summary_data["radiant_lane_kills"] += kills
                else: # dire
                    lane_summary_data["dire_total_gold_10m"] += kpis_10m["gold"]
                    lane_summary_data["dire_total_xp_10m"] += kpis_10m["xp"]
                    lane_summary_data["dire_lane_kills"] += kills
        
        # Calculate lane verdict scores
        num_radiant_laners = len(teams_in_lane.get("radiant",[]))
        num_dire_laners = len(teams_in_lane.get("dire",[]))
        avg_gold_threshold = 750; avg_xp_threshold = 1000
        minor_gold_lead_threshold = 300; minor_xp_lead_threshold = 500

        gold_diff = lane_summary_data["radiant_total_gold_10m"] - lane_summary_data["dire_total_gold_10m"]
        xp_diff = lane_summary_data["radiant_total_xp_10m"] - lane_summary_data["dire_total_xp_10m"]
        kill_diff = lane_summary_data["radiant_lane_kills"] - lane_summary_data["dire_lane_kills"]

        radiant_score = 0; dire_score = 0
        if num_radiant_laners > 0 and gold_diff > avg_gold_threshold * num_radiant_laners : radiant_score += 2
        elif num_dire_laners > 0 and gold_diff < -avg_gold_threshold * num_dire_laners : dire_score += 2
        elif num_radiant_laners > 0 and gold_diff > minor_gold_lead_threshold * num_radiant_laners : radiant_score +=1
        elif num_dire_laners > 0 and gold_diff < -minor_gold_lead_threshold * num_dire_laners : dire_score +=1

        if num_radiant_laners > 0 and xp_diff > avg_xp_threshold * num_radiant_laners : radiant_score += 2
        elif num_dire_laners > 0 and xp_diff < -avg_xp_threshold * num_dire_laners : dire_score += 2
        elif num_radiant_laners > 0 and xp_diff > minor_xp_lead_threshold * num_radiant_laners : radiant_score +=1
        elif num_dire_laners > 0 and xp_diff < -minor_xp_lead_threshold * num_dire_laners : dire_score +=1

        if kill_diff >= 2 : radiant_score += 2 
        elif kill_diff <= -2 : dire_score += 2 
        elif kill_diff == 1 : radiant_score +=1
        elif kill_diff == -1 : dire_score +=1
        
        current_lane_tower_status = early_tower_status.get(lane_name_key, {"radiant_t1_fallen": False, "dire_t1_fallen": False})
        if current_lane_tower_status["dire_t1_fallen"]: radiant_score += 3 
        if current_lane_tower_status["radiant_t1_fallen"]: dire_score += 3 

        lane_verdict_text_raw = ""
        if radiant_score > dire_score + 1: lane_verdict_text_raw = f"Radiant Ahead ({radiant_score} vs {dire_score})"
        elif dire_score > radiant_score + 1: lane_verdict_text_raw = f"Dire Ahead ({dire_score} vs {radiant_score})"
        else: lane_verdict_text_raw = f"Even Lane ({radiant_score} vs {dire_score})"
        
        analysis_results["lanes"][lane_name_key] = {
            "radiant_score": radiant_score,
            "dire_score": dire_score,
            "verdict_text": lane_verdict_text_raw,
            "summary_details": { # Store details for potential use by caller
                "gold_diff": gold_diff, "xp_diff": xp_diff, "kill_diff": kill_diff,
                "radiant_t1_fallen": current_lane_tower_status["radiant_t1_fallen"],
                "dire_t1_fallen": current_lane_tower_status["dire_t1_fallen"]
            }
        }
    return analysis_results
# --- End Lane Analysis Core Logic ---


# --- Typer Commands ---

@app.command(name="set-player-name", help="Assign or update a custom name for a player's Account ID.")
def set_player_name(
    account_id: int = typer.Argument(..., help="The player's unique Account ID."),
    name: str = typer.Argument(..., help="The custom name to assign (e.g., 'SumaiL', 'Miracle-').")
):
    """Stores or updates a custom name for a given player account ID in the database."""
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
        console.print(f"[bold red]Error: The custom name '{name}' might already be assigned to another Account ID, or another integrity constraint failed.[/bold red]")
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error setting custom name for Account ID {account_id}: {e}[/bold red]")
    finally:
        conn.close()

@app.command(name="list-custom-names", help="Lists all custom player names stored in the database.")
def list_custom_names():
    """Displays all stored custom player names and their associated account IDs."""
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

@app.command(name="delete-player-name", help="Deletes a custom player name by Account ID or the Custom Name itself.")
def delete_player_name(
    identifier: str = typer.Argument(..., help="The Account ID or Custom Name to delete.")
):
    """Deletes a custom player name from the database using either the account ID or the name itself."""
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
            console.print(f"[yellow]No custom name found for Account ID {account_id_to_delete} to delete.[/yellow]")
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error deleting custom name for Account ID {account_id_to_delete}: {e}[/bold red]")
    finally:
        conn.close()

@app.command(name="list-heroes", help="Lists all heroes from the 'heroes' table in the database.")
def list_heroes():
    """Displays a table of all heroes found in the 'heroes' table, showing their ID and internal name."""
    console.print(Panel("[bold blue]Listing All Stored Heroes[/bold blue]", expand=False))
    conn = get_db_connection()
    cursor = conn.cursor()
    heroes_rows: Optional[List[sqlite3.Row]] = None
    try:
        cursor.execute("SELECT hero_id, name, fetched_at FROM heroes ORDER BY hero_id ASC")
        heroes_rows = cursor.fetchall()
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite query error for heroes: {e}[/bold red]")
        console.print("[dim]Ensure the 'heroes' table exists and contains 'hero_id', 'name', and 'fetched_at' columns.[/dim]")
        conn.close(); raise typer.Exit(code=1)
    finally:
        conn.close()

    if not heroes_rows:
        console.print("[yellow]No heroes found in the database. (Hint: Populate using the main data fetching script, e.g., league_info.py)[/yellow]"); return

    table = Table(title="Stored Heroes (from Database)", show_header=True, header_style="bold magenta")
    table.add_column("Hero ID", style="dim", width=10, justify="center")
    table.add_column("Internal Name (e.g., npc_dota_hero_...)", min_width=30) 
    table.add_column("Fetched At (UTC)", style="dim", min_width=20, justify="center")
    for hero in heroes_rows:
        table.add_row(str(hero["hero_id"]), hero["name"], hero["fetched_at"] or "N/A")
    console.print(table)
    console.print(f"\nTotal heroes in DB: {len(heroes_rows)}")


@app.command(name="list-matches", help="Lists stored matches with summary details. Use -l for limit, -t to filter by team.")
def list_matches(
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit the number of matches displayed (most recent first)."),
    search_team: Optional[str] = typer.Option(None, "--team", "-t", help="Filter matches by a team name (case-insensitive search in Radiant or Dire team names/tags).")
):
    """
    Displays a list of matches from the database, including team names, scores, winner,
    and player details (Custom Name/Account ID & Hero).
    """
    console.print(Panel("[bold blue]Listing Stored Matches with Player Details[/bold blue]", expand=False))
    conn = get_db_connection()
    hero_map = _get_all_heroes_map_from_db(conn) 
    player_custom_names_map = _get_all_player_custom_names_map_from_db(conn)

    query = "SELECT match_id, data, fetched_at FROM matches "
    params_list: list = [] 

    where_clauses = []
    if search_team:
        where_clauses.append("(data LIKE ? OR data LIKE ? OR data LIKE ? OR data LIKE ?)") 
        like_pattern = f'%"{search_team}"%' 
        params_list.extend([like_pattern, like_pattern, like_pattern, like_pattern]) 

    if where_clauses:
        query += "WHERE " + " AND ".join(where_clauses)

    query += " ORDER BY fetched_at DESC" 

    if limit is not None and limit > 0:
        query += " LIMIT ?"
        params_list.append(limit)

    params = tuple(params_list)

    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        match_rows = cursor.fetchall()
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite query error for matches: {e}[/bold red]")
        conn.close(); raise typer.Exit(code=1)

    if not match_rows:
        if search_team:
            console.print(f"[yellow]No matches found in DB potentially matching team '{search_team}' via JSON LIKE search.[/yellow]")
        else:
            console.print("[yellow]No matches found in the database. (Hint: Populate using the main data fetching script)[/yellow]")
        conn.close()
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

        r_team_data = match_data.get('radiant_team', {})
        d_team_data = match_data.get('dire_team', {})
        r_team_name = match_data.get('radiant_name') or (r_team_data.get('name') if isinstance(r_team_data, dict) else None) or (r_team_data.get('tag') if isinstance(r_team_data, dict) else None) or "Radiant"
        d_team_name = match_data.get('dire_name') or (d_team_data.get('name') if isinstance(d_team_data, dict) else None) or (d_team_data.get('tag') if isinstance(d_team_data, dict) else None) or "Dire"

        if search_team:
            team_name_lower = search_team.lower()
            if not (team_name_lower in r_team_name.lower() or \
                    team_name_lower in d_team_name.lower() or \
                    (isinstance(r_team_data, dict) and team_name_lower in str(r_team_data.get('tag','')).lower()) or \
                    (isinstance(d_team_data, dict) and team_name_lower in str(d_team_data.get('tag','')).lower())):
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
            hero_name_display = _get_hero_display_name_from_id(hero_id, hero_map) if hero_id else "N/A"
            player_detail_str = f"{player_identifier} ({hero_name_display})"

            is_radiant_player = p_info.get("isRadiant")
            if is_radiant_player is None: 
                player_slot = p_info.get("player_slot")
                if player_slot is not None: is_radiant_player = player_slot < 100
                else: is_radiant_player = False 

            if is_radiant_player:
                radiant_players_details_list.append(player_detail_str)
            else:
                dire_players_details_list.append(player_detail_str)

        radiant_players_display_str = "\n".join(radiant_players_details_list) if radiant_players_details_list else "N/A"
        dire_players_display_str = "\n".join(dire_players_details_list) if dire_players_details_list else "N/A"
        table.add_row(str(match_id), r_team_name, d_team_name, f"{r_score}-{d_score}", winner, radiant_players_display_str, dire_players_display_str, fetched_at)
        matches_displayed_count +=1

    conn.close() 

    if matches_displayed_count > 0:
        console.print(table)
        console.print(f"\nTotal matches displayed: {matches_displayed_count}")
        if limit and matches_displayed_count >= limit : console.print(f"Showing up to {limit} matches. Use --limit to change.")
        elif search_team: console.print(f"Showing matches filtered by team: '{search_team}'")
    elif search_team:
        console.print(f"[yellow]No matches found where Radiant or Dire team name/tag contains '{search_team}' after parsing JSON.[/yellow]")


@app.command(name="find-player-hero", help="Finds matches for a player, optionally on a specific hero. Use -l for limit.")
def find_player_hero_matches(
    player_identifier: str = typer.Argument(..., help="Player's Account ID, assigned Custom Name, or current Persona Name."),
    hero_name_input: Optional[str] = typer.Argument(None, help="Optional: The name of the hero (e.g., 'antimage', 'npc_dota_hero_juggernaut'). If omitted, lists all matches for the player."),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit the number of matches displayed.")
):
    """
    Searches for matches based on a player identifier and optionally a hero name (internal or simplified).
    """
    conn = get_db_connection()
    target_hero_id: Optional[int] = None
    hero_search_active = False
    hero_map = _get_all_heroes_map_from_db(conn) 
    player_custom_names_map = _get_all_player_custom_names_map_from_db(conn)
    actual_hero_name_for_display = ""

    if hero_name_input:
        hero_search_active = True
        target_hero_id = get_hero_id_from_name_db(hero_name_input, conn) 
        if target_hero_id is None:
            console.print(f"[bold red]Hero '{hero_name_input}' not found or ambiguous. Try 'list-heroes' to see available internal names.[/bold red]")
            conn.close(); raise typer.Exit(code=1)
        actual_hero_name_for_display = _get_hero_display_name_from_id(target_hero_id, hero_map)
        console.print(f"[info]Targeting Hero ID: {target_hero_id} ('{actual_hero_name_for_display}')[/info]")
        panel_title = f"[bold blue]Player '{player_identifier}' on Hero '{actual_hero_name_for_display}'[/bold blue]"
        table_title_suffix = f" on Hero '{actual_hero_name_for_display}' (ID: {target_hero_id})"
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
    sql_search_term = str(search_target_account_id) if search_target_account_id is not None else player_identifier
    params = (f'%{sql_search_term}%',)

    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        match_rows = cursor.fetchall()
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite query error: {e}[/bold red]"); conn.close(); raise typer.Exit(code=1)

    if not match_rows:
        console.print(f"[yellow]No matches in DB potentially involving '{player_identifier}' based on initial SQL filter.[/yellow]"); conn.close(); return

    found_matches_details = []
    for row in match_rows:
        try:
            match_data: Dict[str, Any] = json.loads(row["data"])
        except json.JSONDecodeError: continue

        players_list: List[Dict[str, Any]] = match_data.get("players", [])
        player_found_in_match_and_met_criteria = False
        hero_played_by_target_player_id_in_this_match: Optional[int] = None

        for p_info in players_list:
            player_matches_identifier_criteria = False
            if search_target_account_id is not None: 
                if p_info.get("account_id") == search_target_account_id:
                    player_matches_identifier_criteria = True
            elif search_by_personaname_fallback: 
                current_player_name = p_info.get("personaname")
                if current_player_name and isinstance(current_player_name, str) and player_identifier.lower() in current_player_name.lower():
                    player_matches_identifier_criteria = True

            if player_matches_identifier_criteria:
                hero_played_by_target_player_id_in_this_match = p_info.get("hero_id")
                if hero_search_active: 
                    if hero_played_by_target_player_id_in_this_match == target_hero_id:
                        player_found_in_match_and_met_criteria = True; break 
                else: 
                    player_found_in_match_and_met_criteria = True; break 

        if player_found_in_match_and_met_criteria:
            r_team_data = match_data.get('radiant_team', {})
            d_team_data = match_data.get('dire_team', {})
            r_team_name = match_data.get('radiant_name') or (r_team_data.get('name') if isinstance(r_team_data, dict) else None) or (r_team_data.get('tag') if isinstance(r_team_data, dict) else None) or "Radiant"
            d_team_name = match_data.get('dire_name') or (d_team_data.get('name') if isinstance(d_team_data, dict) else None) or (d_team_data.get('tag') if isinstance(d_team_data, dict) else None) or "Dire"

            r_score = str(match_data.get('radiant_score', '-')); d_score = str(match_data.get('dire_score', '-'))
            winner = "N/A"
            if match_data.get('radiant_win') is True: winner = f"[bold cyan]{r_team_name}[/bold cyan]"
            elif match_data.get('radiant_win') is False: winner = f"[bold orange3]{d_team_name}[/bold orange3]"

            player_hero_display_name = _get_hero_display_name_from_id(hero_played_by_target_player_id_in_this_match, hero_map) if hero_played_by_target_player_id_in_this_match else "N/A"
            found_matches_details.append({
                "match_id": str(row["match_id"]),
                "radiant_name": r_team_name,
                "dire_name": d_team_name,
                "score": f"{r_score}-{d_score}",
                "winner": winner,
                "fetched_at": row["fetched_at"],
                "player_hero": player_hero_display_name
            })
            if limit and len(found_matches_details) >= limit: break
    conn.close()

    if not found_matches_details:
        if hero_search_active and target_hero_id is not None:
             console.print(f"[yellow]No matches found where player '{display_search_term}' played hero '{actual_hero_name_for_display}'.[/yellow]")
        else:
            console.print(f"[yellow]No matches found for player '{display_search_term}'.[/yellow]")
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


@app.command(name="show-match-json", help="Shows JSON for a match. Use -v to visualize, -o to save to file.")
def show_match_json(
    match_id: int = typer.Argument(..., help="The Match ID."),
    visualize: bool = typer.Option(
        False, "--visualize", "-v",
        help="Generate an interactive visual map of the JSON structure (opens in browser). Requires gravis and networkx.",
        is_flag=True
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output-file", "-o",
        help="Path to save the raw JSON data (e.g., match_data.json).",
        show_default=False, dir_okay=False, writable=True 
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
    output_file_path_abs: Optional[str] = None # To store absolute path if output_file is used

    try:
        cursor.execute("SELECT data FROM matches WHERE match_id = ?", (match_id,))
        row_data = cursor.fetchone()
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite query error for match JSON: {e}[/bold red]")
        if conn: conn.close(); raise typer.Exit(code=1)

    if not row_data or not row_data["data"]:
        console.print(f"[yellow]No data found in DB for match ID {match_id}.[/yellow]")
        if conn: conn.close(); return

    json_string_data = row_data["data"]
    match_json_parsed_data: Optional[Dict[str, Any]] = None
    visualization_successful = False # Track visualization status

    try:
        match_json_parsed_data = json.loads(json_string_data)

        if output_file:
            try:
                output_file_path_abs = os.path.abspath(output_file)
                output_dir = os.path.dirname(output_file_path_abs)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir) 
                    console.print(f"[info]Created directory: {output_dir}[/info]")

                with open(output_file_path_abs, 'w') as f:
                    json.dump(match_json_parsed_data, f, indent=2)
                console.print(f"[green]Successfully saved JSON data for match {match_id} to: {output_file_path_abs}[/green]")
            except IOError as e:
                console.print(f"[bold red]Error saving JSON to file '{output_file}': {e}[/bold red]")
            except Exception as e_fs: 
                console.print(f"[bold red]An unexpected error occurred while preparing to save to file '{output_file}': {e_fs}[/bold red]")

        if visualize:
            console.print("[info]Visualization requested. Attempting to generate graph...[/info]")
            if not VISUALIZATION_LIBRARIES_AVAILABLE: 
                console.print("[bold yellow]Visualization libraries (gravis, networkx) are not installed. Cannot visualize.[/bold yellow]")
                console.print("To install them, run: [code]pip install gravis networkx[/code]")
            elif match_json_parsed_data is not None: 
                visualization_successful = _visualize_json_structure_gravis(match_json_parsed_data, match_id, "Match")
                if not visualization_successful:
                    console.print("[yellow]Visualization generation failed or was skipped.[/yellow]")
            else: 
                console.print("[yellow]Cannot visualize as JSON data was not parsed successfully.[/yellow]")

        # Print to terminal if no other action was taken or if actions failed
        should_print_to_console = True
        if output_file and output_file_path_abs and os.path.exists(output_file_path_abs):
            should_print_to_console = False # Successfully saved to file
        if visualize and visualization_successful:
            should_print_to_console = False # Successfully visualized (implies browser open)
        
        if should_print_to_console:
            if not visualize and not output_file : # Explicitly no other option chosen
                 console.print("\n[info]Displaying JSON in terminal:[/info]")
            elif (visualize and not visualization_successful) or \
                 (output_file and (not output_file_path_abs or not os.path.exists(output_file_path_abs))):
                # If visualization or saving was attempted but failed, offer to print
                if typer.confirm("Visualization/Saving failed or was not performed. Print JSON to console instead?", default=False):
                    console.print("\n[info]Displaying JSON in terminal as fallback:[/info]")
                else: # User chose not to print as fallback
                    should_print_to_console = False 
            
            if should_print_to_console and match_json_parsed_data: # Check again if we should print
                syntax = Syntax(json.dumps(match_json_parsed_data, indent=2), "json", theme="material", line_numbers=True)
                console.print(syntax)


    except json.JSONDecodeError:
        console.print(f"[bold red]Error: Stored data for match ID {match_id} is not valid JSON.[/bold red]")
        if output_file: 
            try:
                output_file_path_abs = os.path.abspath(output_file) # Ensure path is defined
                output_dir = os.path.dirname(output_file_path_abs)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(output_file_path_abs, 'w') as f:
                    f.write(json_string_data)
                console.print(f"[yellow]Saved raw (invalid JSON) data for match {match_id} to: {output_file_path_abs}[/yellow]")
            except IOError as e:
                console.print(f"[bold red]Error saving raw invalid JSON to file '{output_file}': {e}[/bold red]")
        else:
            console.print("[info]Raw data from DB (which is not valid JSON):")
            console.print(json_string_data[:1000] + "..." if len(json_string_data) > 1000 else json_string_data)
    finally:
        if conn: conn.close()

@app.command(name="db-stats", help="Shows basic statistics about the database (match counts, hero counts, etc.).")
def db_stats():
    """Displays an overview of the database, including counts of matches, heroes, and custom names."""
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
            stats["Matches Fetched Between (UTC)"] = f"{match_dates_row[0]} and {match_dates_row[1]}"
        else:
            stats["Matches Fetched Between (UTC)"] = "N/A (No valid dates or no matches with dates)"

    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite query error for DB stats: {e}[/bold red]")
        if conn: conn.close(); raise typer.Exit(code=1)
    finally:
        if conn: conn.close()

    table = Table(title="Database Overview", show_lines=True)
    table.add_column("Statistic", style="bold magenta")
    table.add_column("Value", style="green")
    for key, value in stats.items():
        table.add_row(key, str(value))
    console.print(table)


@app.command(name="player-summary", help="Summary for a player (matches, top heroes, win rate).")
def player_summary(
    player_identifier: str = typer.Argument(..., help="Player's Account ID, Custom Name, or current Persona Name.")
):
    """
    Provides a performance summary for a player, including total matches, wins, win rate,
    and top played heroes with their win rates.
    """
    console.print(Panel(f"[bold blue]Summary for Player '{player_identifier}'[/bold blue]", expand=False))
    conn = get_db_connection()
    player_custom_names_map = _get_all_player_custom_names_map_from_db(conn)
    hero_map = _get_all_heroes_map_from_db(conn) 

    target_account_id: Optional[int] = None
    search_by_personaname_fallback = False
    display_player_term = player_identifier
    sql_like_prefilter_term: str

    try:
        target_account_id = int(player_identifier)
        custom_name = player_custom_names_map.get(target_account_id)
        display_player_term = f"{custom_name} (ID: {target_account_id})" if custom_name else f"ID: {target_account_id}"
        sql_like_prefilter_term = f'%"account_id": {target_account_id}%' 
        console.print(f"[info]Interpreted player identifier as Account ID: {target_account_id}[/info]")
    except ValueError: 
        resolved_by_custom_name = False
        for acc_id, cust_name in player_custom_names_map.items():
            if player_identifier.lower() == cust_name.lower():
                target_account_id = acc_id
                display_player_term = f"{cust_name} (resolved to ID: {acc_id})"
                sql_like_prefilter_term = f'%"account_id": {target_account_id}%'
                console.print(f"[info]Resolved Custom Name '{player_identifier}' to Account ID: {target_account_id}[/info]")
                resolved_by_custom_name = True
                break
        if not resolved_by_custom_name:
            search_by_personaname_fallback = True
            escaped_player_identifier = player_identifier.replace('"', '""')
            sql_like_prefilter_term = f'%"personaname": "%{escaped_player_identifier}%"%'
            console.print(f"[info]Searching by Persona Name containing: '{player_identifier}'[/info]")

    query = "SELECT data FROM matches WHERE data LIKE ? ORDER BY fetched_at DESC"
    params = (sql_like_prefilter_term,)

    matches_played = 0
    wins = 0
    hero_performance: Dict[int, Dict[str, int]] = defaultdict(lambda: {"picks": 0, "wins": 0})

    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        match_rows = cursor.fetchall()
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error fetching matches for player summary: {e}[/bold red]")
        if conn: conn.close(); return

    if not match_rows:
        console.print(f"[yellow]No matches found in DB potentially involving '{display_player_term}' based on initial SQL filter.[/yellow]")
        if conn: conn.close(); return

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
        sorted_hero_performance = sorted(
            hero_performance.items(),
            key=lambda item: (item[1]["picks"], (item[1]['wins'] / item[1]['picks'] * 100) if item[1]['picks'] > 0 else 0),
            reverse=True
        )

        top_heroes_table = Table(show_header=True, header_style="bold cyan")
        top_heroes_table.add_column("Hero", style="green")
        top_heroes_table.add_column("Picks", style="magenta", justify="center")
        top_heroes_table.add_column("Win Rate (%)", style="blue", justify="center")

        for hero_id_val, stats_dict in sorted_hero_performance[:5]:
            hero_display_name = _get_hero_display_name_from_id(hero_id_val, hero_map) 
            hero_win_rate = (stats_dict['wins'] / stats_dict['picks'] * 100) if stats_dict['picks'] > 0 else 0
            top_heroes_table.add_row(hero_display_name, str(stats_dict['picks']), f"{hero_win_rate:.2f}%")
        console.print(top_heroes_table)
    else:
        console.print("No hero pick data available for this player in the stored matches.")


@app.command(name="hero-summary", help="Summary for a hero (pick count, win rate, top players).")
def hero_summary(
    hero_identifier: str = typer.Argument(..., help="The name (e.g. 'antimage') or ID of the hero.")
):
    """
    Provides a summary for a specific hero, including total picks, win rate,
    and top players who played this hero with their respective win rates.
    """
    console.print(Panel(f"[bold blue]Summary for Hero '{hero_identifier}'[/bold blue]", expand=False))
    conn = get_db_connection()
    hero_map = _get_all_heroes_map_from_db(conn) 
    player_custom_names_map = _get_all_player_custom_names_map_from_db(conn)
    
    target_hero_id: Optional[int] = None
    try:
        target_hero_id = int(hero_identifier)
        console.print(f"[info]Interpreted hero identifier as Hero ID: {target_hero_id}[/info]")
    except ValueError:
        target_hero_id = get_hero_id_from_name_db(hero_identifier, conn) 
        if target_hero_id is None:
            console.print(f"[bold red]Hero name '{hero_identifier}' not found or ambiguous. Try 'list-heroes'.[/bold red]")
            if conn: conn.close(); return
        console.print(f"[info]Resolved hero name '{hero_identifier}' to ID: {target_hero_id}[/info]")

    actual_hero_name_for_display = _get_hero_display_name_from_id(target_hero_id, hero_map)

    sql_like_prefilter_term = f'%"hero_id": {target_hero_id}%' 

    query = "SELECT data FROM matches WHERE data LIKE ? ORDER BY fetched_at DESC"
    params = (sql_like_prefilter_term,)

    total_hero_picks = 0
    total_hero_wins = 0 
    player_performance_with_hero: Dict[int, Dict[str, int]] = defaultdict(lambda: {"picks": 0, "wins": 0})

    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        match_rows = cursor.fetchall()
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error fetching matches for hero summary: {e}[/bold red]")
        if conn: conn.close(); return

    if not match_rows:
        console.print(f"[yellow]No matches found in DB potentially featuring hero '{actual_hero_name_for_display}' (ID: {target_hero_id}) based on initial SQL filter.[/yellow]")
        if conn: conn.close(); return

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
            account_id_of_picker = player_who_picked_hero_info.get("account_id")

            picker_is_radiant_val = player_who_picked_hero_info.get("isRadiant")
            if picker_is_radiant_val is None and "player_slot" in player_who_picked_hero_info:
                picker_is_radiant_val = player_who_picked_hero_info["player_slot"] < 100

            radiant_won = match_data.get("radiant_win")
            picker_won_this_match = False

            if radiant_won is not None and picker_is_radiant_val is not None:
                if radiant_won is True and picker_is_radiant_val:
                    total_hero_wins += 1 
                    picker_won_this_match = True
                elif radiant_won is False and not picker_is_radiant_val:
                    total_hero_wins += 1 
                    picker_won_this_match = True

            if account_id_of_picker: 
                player_performance_with_hero[account_id_of_picker]["picks"] += 1
                if picker_won_this_match:
                    player_performance_with_hero[account_id_of_picker]["wins"] += 1
    if conn: conn.close()

    if total_hero_picks == 0:
        console.print(f"[yellow]Hero '{actual_hero_name_for_display}' (ID: {target_hero_id}) was not confirmed picked in any stored matches after detailed check.[/yellow]"); return

    overall_hero_win_rate = (total_hero_wins / total_hero_picks * 100) if total_hero_picks > 0 else 0

    console.print(f"\n--- Summary for Hero: {actual_hero_name_for_display} (ID: {target_hero_id}) ---")
    console.print(f"Total Picks in DB: {total_hero_picks}")
    console.print(f"Overall Wins (when hero was played): {total_hero_wins}")
    console.print(f"Overall Win Rate (when hero was played): {overall_hero_win_rate:.2f}%")

    if player_performance_with_hero:
        console.print("\n[bold]Most Frequent Players (Top 5, sorted by picks then win rate):[/bold]")
        sorted_player_performance = sorted(
            player_performance_with_hero.items(),
            key=lambda item: (item[1]["picks"], (item[1]['wins'] / item[1]['picks'] * 100) if item[1]['picks'] > 0 else 0),
            reverse=True
        )

        top_players_table = Table(show_header=True, header_style="bold cyan")
        top_players_table.add_column("Player (Name/ID)", style="green")
        top_players_table.add_column("Picks of Hero", style="magenta", justify="center")
        top_players_table.add_column("Win Rate (%) with Hero", style="blue", justify="center")

        for acc_id_val, stats_dict in sorted_player_performance[:5]:
            player_display_name = player_custom_names_map.get(acc_id_val, f"ID: {acc_id_val}")
            player_hero_win_rate = (stats_dict['wins'] / stats_dict['picks'] * 100) if stats_dict['picks'] > 0 else 0
            top_players_table.add_row(player_display_name, str(stats_dict['picks']), f"{player_hero_win_rate:.2f}%")
        console.print(top_players_table)
    else:
        console.print("No specific player performance data available for this hero (likely due to missing account IDs in match data or hero not picked by identifiable players).")


@app.command(name="analyze-lanes", help="Analyzes laning phase for a match ID and prints to console.")
def analyze_lanes_command(match_id: int = typer.Argument(..., help="The Match ID to analyze.")):
    """
    Fetches match data from the DB, performs a laning phase analysis,
    and prints detailed results to the console.
    """
    console.print(Panel(f"[bold blue]Laning Phase Analysis for Match ID {match_id}[/bold blue]", expand=False))
    conn = get_db_connection()
    hero_map = _get_all_heroes_map_from_db(conn)
    player_custom_names_map = _get_all_player_custom_names_map_from_db(conn)

    # Perform the core analysis
    analysis_data = _perform_core_lane_analysis(match_id, conn, hero_map, player_custom_names_map)

    if not analysis_data or analysis_data.get("error"):
        console.print(f"[bold red]Could not analyze lanes for match {match_id}: {analysis_data.get('error', 'Unknown error')}[/bold red]")
        if conn: conn.close()
        return

    # Now, use analysis_data to print to console in the original format
    # This requires fetching the full match_data again to get player details for tables,
    # or modifying _perform_core_lane_analysis to return more detailed player kpis if needed for console.
    # For simplicity, let's re-fetch match_data here for detailed player tables.
    
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT data FROM matches WHERE match_id = ?", (match_id,))
        row_data = cursor.fetchone()
        match_data = json.loads(row_data["data"]) if row_data and row_data["data"] else None
    except (sqlite3.Error, json.JSONDecodeError) as e:
        console.print(f"[bold red]Error re-fetching match data for console display: {e}[/bold red]")
        if conn: conn.close()
        return
    finally:
        if conn: conn.close() # Close connection after this specific command is done

    if not match_data:
        console.print(f"[bold red]Could not re-fetch match data for console display for match {match_id}.[/bold red]")
        return

    players_data_for_tables = match_data.get("players", [])
    # Enrich again, as _perform_core_lane_analysis might not have modified the original list in place
    # or we might want a fresh copy.
    for p_idx, p_info in enumerate(players_data_for_tables):
        p_acc_id = p_info.get("account_id")
        display_name = p_info.get("personaname", f"Slot {p_info.get('player_slot', p_idx)}")
        if p_acc_id and p_acc_id in player_custom_names_map:
            display_name = player_custom_names_map[p_acc_id]
        players_data_for_tables[p_idx]["display_name"] = display_name
        if "isRadiant" not in p_info and "player_slot" in p_info:
             players_data_for_tables[p_idx]["isRadiant"] = p_info["player_slot"] < 100
    
    lanes_assignment_for_tables = _identify_laners(players_data_for_tables, hero_map)
    
    console.print(f"[info]Draft Order: {analysis_data.get('draft_order', 'N/A')}[/info]\n")

    for lane_name_key, lane_info in analysis_data.get("lanes", {}).items():
        if lane_name_key == "unknown": continue # Already handled if necessary by core logic

        console.print(Panel(f"Lane Analysis: {lane_name_key.upper()}", style="bold yellow", expand=False))
        
        teams_in_lane_for_table = lanes_assignment_for_tables.get(lane_name_key, {"radiant": [], "dire": []})

        for team_name_str, players_in_lane_for_team in teams_in_lane_for_table.items():
            if not players_in_lane_for_team:
                console.print(f"[dim]No {team_name_str} players in {lane_name_key.upper()} lane for detailed table.[/dim]")
                continue
            
            team_color = "cyan" if team_name_str == "radiant" else "orange3"
            table = Table(
                title=f"{team_name_str.capitalize()} - {lane_name_key.upper()} @ 8min / 10min",
                title_style=f"bold {team_color}", show_header=True, header_style="bold magenta"
            )
            table.add_column("Player", style="dim", min_width=15, overflow="fold")
            table.add_column("Hero", min_width=15)
            table.add_column("Lvl", justify="center")
            table.add_column("LH", justify="center")
            table.add_column("DN", justify="center")
            table.add_column("Gold", justify="center")
            table.add_column("GPM", justify="center")
            table.add_column("XP", justify="center")
            table.add_column("XPM", justify="center")
            table.add_column("Kills (vs Lane)", justify="center", header_style="bold green")
            table.add_column("Deaths (to Lane)", justify="center", header_style="bold red")

            opposing_team_str_val = "dire" if team_name_str == "radiant" else "radiant"
            opposing_laners_in_this_lane_objs = lanes_assignment_for_tables[lane_name_key].get(opposing_team_str_val,[])
            opposing_laner_hero_names_list = [p.get('hero_name_parsed', '') for p in opposing_laners_in_this_lane_objs if p.get('hero_name_parsed')]

            for player_obj_data in players_in_lane_for_team:
                kpis_8m = _get_player_kpis_at_minute(player_obj_data, 8, hero_map)
                kpis_10m = _get_player_kpis_at_minute(player_obj_data, 10, hero_map)
                kills, deaths = _analyze_lane_kill_death_events(player_obj_data, players_data_for_tables, opposing_laner_hero_names_list, LANE_PHASE_END_TIME_SECONDS_LONG)
                table.add_row(
                    player_obj_data.get("display_name"), 
                    kpis_10m["hero"], 
                    f"{kpis_8m['level']} / {kpis_10m['level']}",
                    f"{kpis_8m['lh']} / {kpis_10m['lh']}",
                    f"{kpis_8m['dn']} / {kpis_10m['dn']}",
                    f"{kpis_8m['gold']} / {kpis_10m['gold']}",
                    f"{kpis_8m['gpm']} / {kpis_10m['gpm']}",
                    f"{kpis_8m['xp']} / {kpis_10m['xp']}",
                    f"{kpis_8m['xpm']} / {kpis_10m['xpm']}",
                    str(kills), str(deaths)
                )
            console.print(table)

        # Display overall lane summary from analysis_data
        lane_summary_panel_content = Text()
        details = lane_info.get("summary_details", {})
        lane_summary_panel_content.append(f"Total Gold Adv @10m (Radiant - Dire): {details.get('gold_diff', 0):+G}\n", style="default")
        lane_summary_panel_content.append(f"Total XP Adv @10m (Radiant - Dire): {details.get('xp_diff', 0):+G}\n", style="default")
        lane_summary_panel_content.append(f"Net Kills in Lane @10m (Radiant - Dire): {details.get('kill_diff', 0):+}\n", style="default")
        lane_summary_panel_content.append(f"Radiant T1 Fallen by {LANE_PHASE_TOWER_KILL_TIME_LIMIT//60}min: {details.get('radiant_t1_fallen', False)}\n", style="default")
        lane_summary_panel_content.append(f"Dire T1 Fallen by {LANE_PHASE_TOWER_KILL_TIME_LIMIT//60}min: {details.get('dire_t1_fallen', False)}\n", style="default")
        
        verdict_text_for_console = lane_info.get('verdict_text', 'N/A')
        # Apply Rich formatting for console
        if "Radiant Ahead" in verdict_text_for_console: verdict_text_for_console = f"[bold cyan]{verdict_text_for_console}[/bold cyan]"
        elif "Dire Ahead" in verdict_text_for_console: verdict_text_for_console = f"[bold orange3]{verdict_text_for_console}[/bold orange3]"
        
        lane_summary_panel_content.append(f"Verdict: ", style="bold")
        lane_summary_panel_content.append(Text.from_markup(verdict_text_for_console))

        console.print(Panel(lane_summary_panel_content, title=f"Overall {lane_name_key.upper()} Lane Summary (@10min)", expand=False, border_style="green"))
        console.print("-" * console.width)


@app.command(name="export-match-ids", help="Exports all match IDs from the database to a CSV file.")
def export_match_ids(
    output_file: str = typer.Option(..., "--output-file", "-o", help="Path to save the CSV file (e.g., all_match_ids.csv).", dir_okay=False, writable=True)
):
    """Exports all match IDs found in the 'matches' table to the specified CSV file."""
    console.print(Panel(f"[bold blue]Exporting All Match IDs to '{output_file}'[/bold blue]", expand=False))
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT match_id FROM matches ORDER BY match_id ASC")
        match_id_rows = cursor.fetchall()
    except sqlite3.Error as e:
        console.print(f"[bold red]SQLite error fetching match IDs: {e}[/bold red]")
        conn.close(); raise typer.Exit(code=1)
    finally:
        conn.close()

    if not match_id_rows:
        console.print("[yellow]No match IDs found in the database to export.[/yellow]")
        return

    try:
        output_file_path_abs = os.path.abspath(output_file)
        output_dir = os.path.dirname(output_file_path_abs)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            console.print(f"[info]Created directory: {output_dir}[/info]")

        with open(output_file_path_abs, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["match_id"]) # Header
            for row in match_id_rows:
                writer.writerow([row["match_id"]])
        console.print(f"[green]Successfully exported {len(match_id_rows)} match IDs to: {output_file_path_abs}[/green]")
    except IOError as e:
        console.print(f"[bold red]Error writing to CSV file '{output_file}': {e}[/bold red]")
    except Exception as e_fs:
        console.print(f"[bold red]An unexpected error occurred while preparing to save to file '{output_file}': {e_fs}[/bold red]")


@app.command(name="batch-analyze-lanes", help="Batch analyzes lanes for matches in a CSV file, outputs results to another CSV.")
def batch_analyze_lanes(
    input_file: str = typer.Argument(..., help="Path to the input CSV file containing match IDs (must have a 'match_id' header).", exists=True, file_okay=True, dir_okay=False, readable=True),
    output_file: str = typer.Option(..., "--output-file", "-o", help="Path to save the batch analysis results CSV (e.g., batch_lane_analysis.csv).", dir_okay=False, writable=True)
):
    """
    Reads match IDs from an input CSV, performs lane analysis for each,
    and writes the draft order and lane scores to an output CSV file.
    """
    console.print(Panel(f"[bold blue]Batch Lane Analysis from '{input_file}' to '{output_file}'[/bold blue]", expand=False))
    
    match_ids_to_process = []
    try:
        with open(input_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            if "match_id" not in reader.fieldnames:
                console.print(f"[bold red]Error: Input CSV '{input_file}' must contain a 'match_id' column header.[/bold red]")
                raise typer.Exit(code=1)
            for row in reader:
                try:
                    match_ids_to_process.append(int(row["match_id"]))
                except ValueError:
                    console.print(f"[yellow]Warning: Skipping invalid match_id '{row['match_id']}' in '{input_file}'.[/yellow]")
    except IOError as e:
        console.print(f"[bold red]Error reading input CSV file '{input_file}': {e}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e_csv_read:
        console.print(f"[bold red]An unexpected error occurred while reading '{input_file}': {e_csv_read}[/bold red]")
        raise typer.Exit(code=1)

    if not match_ids_to_process:
        console.print(f"[yellow]No valid match IDs found in '{input_file}' to process.[/yellow]")
        return

    conn = get_db_connection()
    hero_map = _get_all_heroes_map_from_db(conn) # Load once
    player_custom_names_map = _get_all_player_custom_names_map_from_db(conn) # Load once

    # Prepare output CSV
    output_file_path_abs = os.path.abspath(output_file)
    output_dir = os.path.dirname(output_file_path_abs)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            console.print(f"[info]Created directory: {output_dir}[/info]")
        except OSError as e:
            console.print(f"[bold red]Error creating output directory '{output_dir}': {e}[/bold red]")
            conn.close(); raise typer.Exit(code=1)
            
    csv_fieldnames = [
        "match_id", "draft_order",
        "top_radiant_score", "top_dire_score", "top_verdict",
        "mid_radiant_score", "mid_dire_score", "mid_verdict",
        "bot_radiant_score", "bot_dire_score", "bot_verdict",
        "analysis_error"
    ]

    processed_count = 0
    error_count = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console, # Use the Rich console for progress output
        transient=True # Clear progress bar on completion
    ) as progress:
        task = progress.add_task("[cyan]Analyzing matches...", total=len(match_ids_to_process))

        try:
            with open(output_file_path_abs, 'w', newline='') as csvoutfile:
                writer = csv.DictWriter(csvoutfile, fieldnames=csv_fieldnames)
                writer.writeheader()

                for match_id in match_ids_to_process:
                    analysis_result = _perform_core_lane_analysis(match_id, conn, hero_map, player_custom_names_map)
                    
                    row_to_write = {"match_id": match_id, "analysis_error": None}
                    
                    if analysis_result and not analysis_result.get("error"):
                        row_to_write["draft_order"] = analysis_result.get("draft_order", "N/A")
                        for lane in ["top", "mid", "bot"]:
                            lane_data = analysis_result.get("lanes", {}).get(lane, {})
                            row_to_write[f"{lane}_radiant_score"] = lane_data.get("radiant_score", 0)
                            row_to_write[f"{lane}_dire_score"] = lane_data.get("dire_score", 0)
                            row_to_write[f"{lane}_verdict"] = lane_data.get("verdict_text", "N/A")
                        processed_count +=1
                    else:
                        row_to_write["analysis_error"] = analysis_result.get("error", "Unknown analysis error")
                        error_count +=1
                    
                    writer.writerow(row_to_write)
                    progress.update(task, advance=1)
        
        except IOError as e:
            console.print(f"[bold red]Error writing to output CSV file '{output_file_path_abs}': {e}[/bold red]")
            conn.close(); raise typer.Exit(code=1)
        except Exception as e_batch:
            console.print(f"[bold red]An unexpected error occurred during batch processing: {e_batch}[/bold red]")
            conn.close(); raise typer.Exit(code=1)

    conn.close()
    console.print(f"\n[green]Batch lane analysis complete.[/green]")
    console.print(f"Successfully processed and wrote {processed_count} matches to '{output_file_path_abs}'.")
    if error_count > 0:
        console.print(f"[yellow]Encountered errors for {error_count} matches (details in CSV).[/yellow]")


if __name__ == "__main__":
    console.print(Panel("[bold green]OpenDota Database Viewer & Analyzer CLI[/bold green]", expand=False))
    app()

