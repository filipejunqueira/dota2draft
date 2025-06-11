# merged_db_viewer_enhanced.py
# CLI tool to view and analyze OpenDota SQLite DB. Local DB interaction only.

import sqlite3
import json
import os
import csv
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter, defaultdict

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.padding import Padding
from rich.style import Style

# Optional Visualization
try:
    import gravis as gv
    import networkx as nx
    import webbrowser
    VISUALIZATION_LIBRARIES_AVAILABLE = True
except ImportError:
    VISUALIZATION_LIBRARIES_AVAILABLE = False
    gv = nx = webbrowser = None

app = typer.Typer(
    help=Text.from_markup(
        "[bold cyan]OpenDota Database Viewer & Analyzer :mag_right:[/bold cyan]\n"
        "A CLI tool to interact with your local OpenDota league information database."
    ),
    rich_markup_mode="markdown"
)
console = Console(theme=None) # Using a theme can be added later if desired

DB_NAME = "opendota_league_info.db"
CONFIG_FILE_NAME = "lane_analysis_kpi_config.json"

# --- KPI Configuration Default Values ---
DEFAULT_KPI_PARAMETERS = {
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
}

# --- KPI Configuration Helper Functions ---
def _save_kpi_parameters(parameters: Dict[str, Any]):
    """Saves the KPI parameters to the JSON configuration file."""
    try:
        with open(CONFIG_FILE_NAME, 'w') as config_file_writer:
            json.dump(parameters, config_file_writer, indent=4)
    except IOError as error:
        console.print(Panel(f"[bold red]Error saving KPI configuration to {CONFIG_FILE_NAME}:[/bold red]\n{error}", title="[bold red]Save Error[/bold red]", border_style="red"))

def _load_kpi_parameters() -> Dict[str, Any]:
    """Loads KPI parameters from JSON, using defaults if file is missing/invalid."""
    if os.path.exists(CONFIG_FILE_NAME):
        try:
            with open(CONFIG_FILE_NAME, 'r') as config_file_reader:
                loaded_params = json.load(config_file_reader)
            
            # Start with defaults and update with loaded params to ensure all keys are present
            current_parameters = DEFAULT_KPI_PARAMETERS.copy()
            current_parameters.update(loaded_params) # Overwrites defaults with loaded values
            
            # Specifically handle nested score_weights to ensure all sub-keys are present
            if isinstance(loaded_params.get("score_weights"), dict):
                 current_parameters["score_weights"] = DEFAULT_KPI_PARAMETERS["score_weights"].copy()
                 current_parameters["score_weights"].update(loaded_params["score_weights"])
            else:
                 current_parameters["score_weights"] = DEFAULT_KPI_PARAMETERS["score_weights"].copy()
                 if "score_weights" in loaded_params : # If it existed but was invalid type
                    console.print(f"[yellow]Warning: 'score_weights' in {CONFIG_FILE_NAME} was invalid. Using defaults for this section.[/yellow]")

            # If the loaded structure was missing keys that defaults provided, save the merged version
            if loaded_params != current_parameters:
                _save_kpi_parameters(current_parameters)
            return current_parameters
        except json.JSONDecodeError:
            console.print(Panel(f"[bold red]Error decoding JSON from {CONFIG_FILE_NAME}.[/bold red]\nUsing default KPI parameters and overwriting the file.", title="[bold red]Config Load Error[/bold red]", border_style="red"))
            _save_kpi_parameters(DEFAULT_KPI_PARAMETERS)
            return DEFAULT_KPI_PARAMETERS.copy()
        except IOError as error:
            console.print(Panel(f"[bold red]Error reading {CONFIG_FILE_NAME}:[/bold red]\n{error}\nUsing default KPI parameters.", title="[bold red]Config Load Error[/bold red]", border_style="red"))
            return DEFAULT_KPI_PARAMETERS.copy()
    else:
        # Config file doesn't exist, create it with defaults
        _save_kpi_parameters(DEFAULT_KPI_PARAMETERS)
        return DEFAULT_KPI_PARAMETERS.copy()

KPI_PARAMETERS = _load_kpi_parameters()

# --- Database and General Helper Functions ---
def _initialize_custom_player_names_table(db_connection: sqlite3.Connection):
    """Ensures the player_custom_names table exists in the database."""
    cursor = db_connection.cursor()
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_custom_names (
            account_id INTEGER PRIMARY KEY,
            custom_name TEXT NOT NULL UNIQUE,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        db_connection.commit()
    except sqlite3.Error as error:
        console.print(Panel(f"[bold red]SQLite error creating 'player_custom_names' table:[/bold red]\n{error}", title="[bold red]DB Error[/bold red]", border_style="red"))

def get_database_connection() -> sqlite3.Connection:
    """Establishes and returns a connection to the SQLite database."""
    try:
        db_connection = sqlite3.connect(DB_NAME)
        db_connection.row_factory = sqlite3.Row # Access columns by name
        _initialize_custom_player_names_table(db_connection) # Ensure custom names table is ready
        return db_connection
    except sqlite3.Error as error:
        console.print(Panel(f"[bold red]SQLite connection error:[/bold red]\n{error}", title="[bold red]DB Connection Error[/bold red]", border_style="red"))
        raise typer.Exit(code=1)

def get_hero_id_from_name_in_database(hero_name_input: str, db_connection: sqlite3.Connection) -> Optional[int]:
    """
    Attempts to find a hero_id from a given hero name string.
    Handles partial matches and 'npc_dota_hero_' prefix.
    """
    cursor = db_connection.cursor()
    try:
        # Exact match first (case-insensitive)
        cursor.execute("SELECT hero_id, name FROM heroes WHERE LOWER(name) = LOWER(?)", (hero_name_input,))
        hero_row = cursor.fetchone()
        if hero_row: return int(hero_row["hero_id"])

        # Try stripping npc_dota_hero_ prefix and doing a LIKE search
        processed_hero_name = hero_name_input.replace("npc_dota_hero_", "")
        cursor.execute("SELECT hero_id, name FROM heroes WHERE name LIKE ? OR name LIKE ?",
                       (f'%{processed_hero_name}%', f'%npc_dota_hero_{processed_hero_name}%'))
        possible_matches = cursor.fetchall()
        
        if len(possible_matches) == 1:
            console.print(f"[dim]Found unique hero match: '{possible_matches[0]['name']}' (ID: {possible_matches[0]['hero_id']})[/dim]")
            return int(possible_matches[0]["hero_id"])
        elif len(possible_matches) > 1:
            console.print(f"[yellow]Multiple heroes found for '{hero_name_input}'. Please be more specific. Matches found:[/yellow]")
            for match_info in possible_matches:
                console.print(f"  - {match_info['name']} (ID: {match_info['hero_id']})")
            return None
        # If no matches by now, it's not found
    except sqlite3.Error as error:
        console.print(Panel(f"[bold red]SQLite error searching for hero '{hero_name_input}':[/bold red]\n{error}", title="[bold red]DB Error[/bold red]", border_style="red"))
    return None

def _get_all_heroes_id_to_name_map_from_db(db_connection: sqlite3.Connection) -> Dict[int, str]:
    """Loads all hero IDs and their internal names from the 'heroes' table."""
    cursor = db_connection.cursor()
    hero_id_to_name_map: Dict[int, str] = {}
    try:
        cursor.execute("SELECT hero_id, name FROM heroes")
        for hero_row in cursor.fetchall():
            hero_id_to_name_map[hero_row["hero_id"]] = hero_row["name"]
    except sqlite3.Error as error:
        console.print(Panel(f"[bold red]SQLite error loading hero data:[/bold red]\n{error}\nHero names might not display correctly.", title="[bold red]DB Error[/bold red]", border_style="red"))
        return {} # Return empty map on error
    if not hero_id_to_name_map:
        console.print("[yellow]Warning: Hero map from database is empty. Hero names may not display correctly.[/yellow]")
    return hero_id_to_name_map

def _get_all_player_custom_names_map_from_db(db_connection: sqlite3.Connection) -> Dict[int, str]:
    """Loads all account IDs and their assigned custom names."""
    cursor = db_connection.cursor()
    custom_names_map: Dict[int, str] = {}
    try:
        cursor.execute("SELECT account_id, custom_name FROM player_custom_names")
        for name_row in cursor.fetchall():
            custom_names_map[name_row["account_id"]] = name_row["custom_name"]
    except sqlite3.Error as error:
        console.print(Panel(f"[bold red]SQLite error loading custom player names:[/bold red]\n{error}", title="[bold red]DB Error[/bold red]", border_style="red"))
    return custom_names_map

def _visualize_json_structure_with_gravis(json_data: Dict[str, Any], item_identifier: Any, item_type_name: str = "Match") -> bool:
    """
    Generates an interactive HTML graph of a JSON structure using Gravis and NetworkX.
    Returns True if visualization was successful or attempted, False if libraries missing or critical error.
    """
    if not VISUALIZATION_LIBRARIES_AVAILABLE:
        console.print(Panel("[bold yellow]Visualization libraries (gravis, networkx) are not installed.[/bold yellow]\nTo install: `pip install gravis networkx`\nSkipping visualization.", title="[yellow]Missing Libraries[/yellow]", border_style="yellow"))
        return False
    if not json_data or not isinstance(json_data, (dict, list)): # Check if data is suitable
        console.print(Panel("[yellow]Invalid or empty JSON data provided for visualization.[/yellow]", title="[yellow]Visualization Warning[/yellow]", border_style="yellow"))
        return False

    console.print(f"[info]Generating interactive graph for {item_type_name} ID {item_identifier}...[/info]")
    if gv: # Gravis might be None if import failed but VISUALIZATION_LIBRARIES_AVAILABLE was somehow true
        console.print(f"[dim]Using Gravis version: {gv.__version__}[/dim]")

    graph = nx.DiGraph() # Initialize a directed graph

    def add_nodes_and_edges_recursively(data_node: Any, parent_node_name: Optional[str] = None):
        """Helper to recursively build the graph from JSON data."""
        default_shape = 'ellipse' # Default shape for nodes
        if isinstance(data_node, dict):
            for key, value in data_node.items():
                current_node_name = f"{parent_node_name}.{key}" if parent_node_name else str(key)
                # Truncate long values for node titles to keep graph readable
                value_snippet = str(value)[:147] + "..." if len(str(value)) > 150 else str(value)
                graph.add_node(current_node_name, title=value_snippet, label=str(key), color='lightblue', size=10, shape=default_shape)
                if parent_node_name:
                    graph.add_edge(parent_node_name, current_node_name)
                if isinstance(value, (dict, list)): # Recurse for nested structures
                    add_nodes_and_edges_recursively(value, current_node_name)
        elif isinstance(data_node, list):
            if not data_node and parent_node_name: # Handle empty lists explicitly
                empty_list_node_name = f"{parent_node_name}.[]"
                graph.add_node(empty_list_node_name, title="Empty List", label="[]", color='lightgrey', size=8, shape='rectangle')
                if parent_node_name:
                    graph.add_edge(parent_node_name, empty_list_node_name)
                return
            for index, item in enumerate(data_node):
                current_node_name = f"{parent_node_name}[{index}]"
                value_snippet = str(item)[:147] + "..." if len(str(item)) > 150 else str(item)
                graph.add_node(current_node_name, title=value_snippet, label=f"[{index}]", color='lightgreen', size=10, shape='box')
                if parent_node_name:
                    graph.add_edge(parent_node_name, current_node_name)
                if isinstance(item, (dict, list)): # Recurse for nested items in list
                    add_nodes_and_edges_recursively(item, current_node_name)

    # Create a root node for the graph
    root_node_label = f"{item_type_name} ID: {item_identifier}"
    graph.add_node(root_node_label, label=root_node_label, color='salmon', size=15, shape='diamond')
    add_nodes_and_edges_recursively(json_data, root_node_label) # Start recursion

    if not graph.nodes() or (len(graph.nodes()) == 1 and root_node_label in graph.nodes()):
         console.print(Panel("[yellow]Warning: The graph is empty or contains only the root node. Visualization might not be useful.[/yellow]", title="[yellow]Visualization Warning[/yellow]", border_style="yellow"))
         # If only root, add full JSON to its title for some utility
         if root_node_label in graph.nodes:
            graph.nodes[root_node_label]['title'] = json.dumps(json_data, indent=2)


    output_filename_absolute = os.path.abspath(f"{item_type_name.lower()}_{item_identifier}_visualization.html")
    try:
        # Attempt to create and export the D3 visualization
        figure = gv.d3(graph, graph_height=800, node_label_data_source='label', 
                         show_menu=True, zoom_factor=0.7, details_min_height=150, 
                         details_max_height=300, use_edge_size_normalization=True, 
                         edge_size_data_source='weight', use_node_size_normalization=True, 
                         node_size_data_source='size')
        figure.export_html(output_filename_absolute, overwrite=True)
        console.print(f"[green]Successfully generated interactive visualization: {output_filename_absolute}[/green]")
        if webbrowser: # Try to open in browser if library is available
            try:
                webbrowser.open(f"file://{output_filename_absolute}")
            except webbrowser.Error as wb_error:
                console.print(f"[yellow]Could not open visualization in browser: {wb_error}. Please open the file manually.[/yellow]")
        return True
    except TypeError as type_error: # Gravis can have TypeErrors with version mismatches or unexpected data
        console.print(f"[bold red]Gravis TypeError (likely version incompatibility or data issue): {type_error}[/bold red]")
        console.print("[info]Attempting a more basic Gravis visualization call as a fallback...[/info]")
        try:
            # Fallback to a simpler Gravis call without some advanced options
            figure_simple = gv.d3(graph, graph_height=800, node_label_data_source='label')
            figure_simple.export_html(output_filename_absolute, overwrite=True)
            console.print(f"[green]Successfully generated basic interactive visualization (fallback): {output_filename_absolute}[/green]")
            if webbrowser:
                try:
                    webbrowser.open(f"file://{output_filename_absolute}")
                except webbrowser.Error as wb_error:
                    console.print(f"[yellow]Could not open basic visualization in browser: {wb_error}.[/yellow]")
            return True
        except Exception as error_simple_gravis:
            console.print(Panel(f"[bold red]Error during basic Gravis (fallback) attempt:[/bold red]\n{error_simple_gravis}", title="[bold red]Visualization Error[/bold red]", border_style="red"))
            return False # Fallback also failed
    except Exception as error_gravis: # Catch any other Gravis-related errors
        console.print(Panel(f"[bold red]General error during Gravis visualization:[/bold red]\n{error_gravis}", title="[bold red]Visualization Error[/bold red]", border_style="red"))
        return False

def _parse_hero_name_from_internal_key(internal_hero_key: Optional[str]) -> str:
    """Converts 'npc_dota_hero_antimage' to 'antimage'."""
    if not internal_hero_key: return "Unknown"
    return internal_hero_key.replace("npc_dota_hero_", "") if internal_hero_key.startswith("npc_dota_hero_") else internal_hero_key

def _get_hero_display_name_from_hero_id(hero_id: Optional[int], hero_id_to_name_map: Dict[int, str]) -> str:
    """Gets a parsed, human-readable hero name from a hero ID using the provided map."""
    if hero_id is None: return "N/A"
    internal_name = hero_id_to_name_map.get(hero_id)
    return _parse_hero_name_from_internal_key(internal_name) if internal_name else f"ID:{hero_id}" # Fallback to ID if not in map

def _identify_laning_players(players_data: List[Dict[str, Any]], hero_id_to_name_map: Dict[int, str]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Identifies players in top, mid, bot lanes for Radiant and Dire.
    Adds 'hero_name_parsed' and 'team_string' to each player dict.
    """
    lanes_assignment: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "top": {"radiant": [], "dire": []}, "mid": {"radiant": [], "dire": []},
        "bot": {"radiant": [], "dire": []}, "unknown": {"radiant": [], "dire": []} # For players with no clear lane
    }
    for player_info in players_data:
        # Determine if player is Radiant (player_slot < 100 for Radiant, >= 100 for Dire)
        is_radiant_player = player_info.get("isRadiant")
        if is_radiant_player is None: # Fallback if 'isRadiant' field is missing
            is_radiant_player = player_info.get("player_slot", 100) < 100 # Default to Dire if slot also missing
        
        team_name = "radiant" if is_radiant_player else "dire"
        assigned_lane_id = player_info.get("lane") # OpenDota's 'lane' field (1: bot/top, 2: mid, 3: top/bot)
        lane_name = "unknown" # Default lane
        
        # Translate OpenDota lane IDs to consistent lane names
        if assigned_lane_id is not None:
            if team_name == "radiant": # Radiant lane perspective
                if assigned_lane_id == 1: lane_name = "bot"    # Radiant's bottom lane
                elif assigned_lane_id == 2: lane_name = "mid"   # Middle lane
                elif assigned_lane_id == 3: lane_name = "top"   # Radiant's top lane
            else: # Dire player perspective (lanes are mirrored)
                if assigned_lane_id == 1: lane_name = "top"    # Dire's top lane (Radiant's bot)
                elif assigned_lane_id == 2: lane_name = "mid"   # Middle lane
                elif assigned_lane_id == 3: lane_name = "bot"   # Dire's bot lane (Radiant's top)
        
        # Add parsed hero name and team string to player data for easier access later
        player_info['hero_name_parsed'] = _get_hero_display_name_from_hero_id(player_info.get('hero_id'), hero_id_to_name_map)
        player_info['team_string'] = team_name
        lanes_assignment[lane_name][team_name].append(player_info)
    return lanes_assignment

def _get_player_kpis_at_specific_minute(player_match_data: Dict[str, Any], minute_mark: int, hero_id_to_name_map: Dict[int, str]) -> Dict[str, Any]:
    """
    Extracts key performance indicators (KPIs) for a player at a specific minute mark from timeseries data.
    Handles cases where timeseries data might not extend to the minute_mark.
    """
    kpis = {"lh": 0, "dn": 0, "gold": 0, "xp": 0, "gpm": 0, "xpm": 0, "level": 1, "hero": "N/A", "name": "N/A"}
    kpis["hero"] = _get_hero_display_name_from_hero_id(player_match_data.get('hero_id'), hero_id_to_name_map)
    kpis["name"] = player_match_data.get("display_name", player_match_data.get("personaname", f"Slot {player_match_data.get('player_slot', '?')}"))
    
    # Helper to safely get value from a timeseries array (e.g., gold_t, lh_t)
    def get_timeseries_value(array_name: str, default_value: int = 0) -> int:
        value_array = player_match_data.get(array_name)
        if value_array and isinstance(value_array, list):
            # If minute_mark is within array bounds, use it; otherwise, use last available value.
            return value_array[minute_mark] if minute_mark < len(value_array) else (value_array[-1] if value_array else default_value)
        return default_value # Return default if array is missing or not a list

    kpis["lh"] = get_timeseries_value("lh_t")
    kpis["dn"] = get_timeseries_value("dn_t")
    kpis["gold"] = get_timeseries_value("gold_t")
    kpis["xp"] = get_timeseries_value("xp_t")

    # Calculate GPM/XPM if minute_mark is valid
    if minute_mark > 0:
        kpis["gpm"] = round(kpis["gold"] / minute_mark) if kpis["gold"] else 0
        kpis["xpm"] = round(kpis["xp"] / minute_mark) if kpis["xp"] else 0
    
    # Determine level: use level_t if available, otherwise estimate from XP
    level_timeseries = player_match_data.get("level_t")
    if level_timeseries and isinstance(level_timeseries, list) and minute_mark < len(level_timeseries): 
        kpis["level"] = level_timeseries[minute_mark]
    else:
        # XP thresholds for levels 1-30 (simplified, actual Dota XP curve is more complex)
        xp_thresholds = [0, 240, 600, 1080, 1680, 2400, 3240, 4200, 5280, 6480, 7800, 9000, 10200, 11400, 12600] # Levels 1-15
        xp_thresholds.extend([xp_thresholds[-1] + 600 * i for i in range(1, 16)]) # Levels 16-30 approx.
        
        level = 1
        current_player_xp = kpis["xp"]
        for index, threshold in enumerate(xp_thresholds):
            if current_player_xp >= threshold:
                level = index + 1
            else:
                break
        kpis["level"] = min(level, 30) # Cap at level 30
    return kpis

def _analyze_lane_kill_death_events_for_player(
    laner_player_object: Dict[str, Any], 
    all_players_data: List[Dict[str, Any]], 
    opposing_laner_hero_names: List[str], 
    time_limit_seconds: int
) -> Tuple[int, int]:
    """
    Analyzes kills by the laner on opposing laners and deaths of the laner to opposing laners.
    """
    kills_on_opposing_laners = 0
    deaths_to_opposing_laners = 0
    laner_hero_name = laner_player_object.get('hero_name_parsed', 'UnknownHero')

    if laner_hero_name == 'UnknownHero' or not laner_hero_name: # Safety check
        return 0, 0

    # Analyze kills made by this laner
    if laner_player_object.get("kills_log"):
        for kill_event in laner_player_object["kills_log"]:
            if kill_event.get("time", float('inf')) <= time_limit_seconds: # Check time limit
                victim_hero_name_key = kill_event.get("key") # e.g., "npc_dota_hero_pudge"
                victim_hero_name_simplified = _parse_hero_name_from_internal_key(victim_hero_name_key)
                if victim_hero_name_simplified in opposing_laner_hero_names:
                    kills_on_opposing_laners += 1

    # Analyze deaths of this laner caused by opposing laners
    # Iterate through all players to find who might have killed this laner
    for potential_killer_object in all_players_data:
        # Skip self or teammates
        if potential_killer_object.get("player_slot") == laner_player_object.get("player_slot"): continue
        if potential_killer_object.get("isRadiant") == laner_player_object.get("isRadiant"): continue

        killer_hero_name_parsed = potential_killer_object.get('hero_name_parsed', 'UnknownHero') 
        # Only consider kills if the killer was one of the opposing laners
        if killer_hero_name_parsed in opposing_laner_hero_names: 
            if potential_killer_object.get("kills_log"):
                for kill_event in potential_killer_object["kills_log"]:
                    if kill_event.get("time", float('inf')) <= time_limit_seconds: # Check time limit
                        victim_hero_name_key = kill_event.get("key") 
                        victim_hero_name_simplified = _parse_hero_name_from_internal_key(victim_hero_name_key)
                        # If the victim of this kill event is our target laner
                        if victim_hero_name_simplified == laner_hero_name: 
                            deaths_to_opposing_laners += 1
    return kills_on_opposing_laners, deaths_to_opposing_laners

def _check_early_tower_status_by_time(objectives_data: Optional[List[Dict[str, Any]]], time_limit_seconds: int) -> Dict[str, Dict[str, bool]]:
    """
    Checks which Tier 1 towers have fallen by a specific time limit.
    Returns a dictionary indicating the status of T1 towers for each lane.
    """
    tower_status = { # Initialize all T1 towers as standing
        "top": {"radiant_t1_fallen": False, "dire_t1_fallen": False},
        "mid": {"radiant_t1_fallen": False, "dire_t1_fallen": False},
        "bot": {"radiant_t1_fallen": False, "dire_t1_fallen": False},
    }
    # Mapping of internal tower names to lane and team status
    tier1_tower_internal_keys = {
        "npc_dota_goodguys_tower1_top": ("top", "radiant_t1_fallen"), "npc_dota_badguys_tower1_top": ("top", "dire_t1_fallen"),
        "npc_dota_goodguys_tower1_mid": ("mid", "radiant_t1_fallen"), "npc_dota_badguys_tower1_mid": ("mid", "dire_t1_fallen"),
        "npc_dota_goodguys_tower1_bot": ("bot", "radiant_t1_fallen"), "npc_dota_badguys_tower1_bot": ("bot", "dire_t1_fallen"),
    }
    if objectives_data:
        for objective_event in objectives_data:
            # Check if it's a building kill within the time limit
            if objective_event.get("type") == "building_kill" and objective_event.get("time", float('inf')) <= time_limit_seconds:
                building_key = objective_event.get("key") # Internal name of the building
                if building_key in tier1_tower_internal_keys:
                    lane_name, status_key_part = tier1_tower_internal_keys[building_key]
                    tower_status[lane_name][status_key_part] = True # Mark tower as fallen
    return tower_status

def _extract_draft_order_string(match_data: Dict[str, Any], hero_id_to_name_map: Dict[int, str]) -> str:
    """Extracts and formats the pick/ban sequence from match data."""
    picks_bans_data = match_data.get("picks_bans")
    if not picks_bans_data or not isinstance(picks_bans_data, list): return "N/A"
    
    draft_entries = []
    # Assuming OpenDota provides picks_bans sorted by "order" field implicitly, or it's already sorted.
    # If not, it would need sorting: sorted_picks_bans = sorted(picks_bans_data, key=lambda x: x.get('order', 0))
    for draft_entry in picks_bans_data: 
        hero_name = _get_hero_display_name_from_hero_id(draft_entry.get("hero_id"), hero_id_to_name_map)
        action_type = "Pick" if draft_entry.get("is_pick") else "Ban"
        team_string = "Radiant" if draft_entry.get("team") == 0 else "Dire" # team 0 is Radiant, 1 is Dire
        draft_entries.append(f"{team_string} {action_type}: {hero_name}")
    return "; ".join(draft_entries) if draft_entries else "N/A"

def _perform_core_lane_analysis_for_match(match_id: int, db_connection: sqlite3.Connection, hero_id_to_name_map: Dict[int, str], player_custom_names_map: Dict[int, str]) -> Optional[Dict[str, Any]]:
    """
    Performs the core laning phase analysis for a given match ID.
    Returns a dictionary with analysis results or an error message.
    """
    cursor = db_connection.cursor()
    try:
        cursor.execute("SELECT data FROM matches WHERE match_id = ?", (match_id,))
        match_row_data = cursor.fetchone()
    except sqlite3.Error as error:
        return {"error": f"SQLite query error for match {match_id}: {error}"}
    
    if not match_row_data or not match_row_data["data"]:
        return {"error": f"No data found in DB for match ID {match_id}."}
    
    try:
        match_json_data = json.loads(match_row_data["data"])
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON for match ID {match_id}."}
    
    players_match_data = match_json_data.get("players", [])
    if not players_match_data:
        return {"error": f"No player data in JSON for match ID {match_id}."}

    # Pre-process player data: add display names and ensure 'isRadiant' is present
    for player_index, player_info in enumerate(players_match_data):
        player_account_id = player_info.get("account_id")
        # Use custom name if available, otherwise personaname, fallback to slot
        display_name = player_custom_names_map.get(player_account_id, player_info.get("personaname", f"Slot {player_info.get('player_slot', player_index)}"))
        players_match_data[player_index]["display_name"] = display_name
        # Ensure 'isRadiant' key exists, deriving from 'player_slot' if necessary
        if "isRadiant" not in player_info and "player_slot" in player_info:
             players_match_data[player_index]["isRadiant"] = player_info["player_slot"] < 100
    
    draft_order_string = _extract_draft_order_string(match_json_data, hero_id_to_name_map)
    lanes_player_assignments = _identify_laning_players(players_match_data, hero_id_to_name_map)
    objectives_match_data = match_json_data.get("objectives") # For tower kills
    
    # Get KPI parameters from the global config
    early_tower_status = _check_early_tower_status_by_time(objectives_match_data, KPI_PARAMETERS["early_tower_kill_time_limit_seconds"])
    kpi_minute_mark = KPI_PARAMETERS["analysis_minute_mark"]
    kill_analysis_time_limit = KPI_PARAMETERS["kill_death_analysis_time_limit_seconds"]
    score_weight_params = KPI_PARAMETERS["score_weights"]

    analysis_results = {"match_id": match_id, "draft_order": draft_order_string, "lanes": {}, "error": None}

    # Analyze each lane (top, mid, bot)
    for lane_name_key in ["top", "mid", "bot"]:
        teams_in_this_lane = lanes_player_assignments.get(lane_name_key)
        # Skip if no players were assigned to this lane (e.g., junglers, roamers not explicitly in lane)
        if not teams_in_this_lane or (not teams_in_this_lane.get("radiant") and not teams_in_this_lane.get("dire")):
            analysis_results["lanes"][lane_name_key] = {"radiant_score": 0, "dire_score": 0, "verdict_text": "No players in lane", "summary_details": {}}
            continue

        # Initialize summary data for the current lane
        lane_summary_data = {
            f"radiant_gold_at_{kpi_minute_mark}m": 0, f"dire_gold_at_{kpi_minute_mark}m": 0, 
            f"radiant_xp_at_{kpi_minute_mark}m": 0, f"dire_xp_at_{kpi_minute_mark}m": 0, 
            "radiant_lane_kills": 0, "dire_lane_kills": 0
        }
        
        # Aggregate KPIs for players in the current lane for each team
        for team_name_string, players_in_lane_for_team in teams_in_this_lane.items():
            if not players_in_lane_for_team: continue # Skip if no players for this team in this lane

            opposing_team_string = "dire" if team_name_string == "radiant" else "radiant"
            opposing_laners_in_this_lane_objects = lanes_player_assignments.get(lane_name_key, {}).get(opposing_team_string, [])
            opposing_laner_hero_names_list = [p.get('hero_name_parsed', '') for p in opposing_laners_in_this_lane_objects if p.get('hero_name_parsed')]

            for player_object_data in players_in_lane_for_team:
                kpis_at_configured_minute = _get_player_kpis_at_specific_minute(player_object_data, kpi_minute_mark, hero_id_to_name_map)
                # Analyze kills/deaths against opposing laners
                kills, _ = _analyze_lane_kill_death_events_for_player(player_object_data, players_match_data, opposing_laner_hero_names_list, kill_analysis_time_limit)
                
                team_prefix = "radiant" if team_name_string == "radiant" else "dire"
                lane_summary_data[f"{team_prefix}_gold_at_{kpi_minute_mark}m"] += kpis_at_configured_minute["gold"]
                lane_summary_data[f"{team_prefix}_xp_at_{kpi_minute_mark}m"] += kpis_at_configured_minute["xp"]
                lane_summary_data[f"{team_prefix}_lane_kills"] += kills
        
        # Calculate differences and scores based on aggregated KPIs
        num_radiant_laners = len(teams_in_this_lane.get("radiant",[])); 
        num_dire_laners = len(teams_in_this_lane.get("dire",[]))
        
        # Get thresholds for scoring from KPI_PARAMETERS
        major_gold_lead_thresh = score_weight_params["major_gold_lead_per_laner_threshold"]
        major_xp_lead_thresh = score_weight_params["major_xp_lead_per_laner_threshold"]
        minor_gold_lead_thresh = score_weight_params["minor_gold_lead_per_laner_threshold"]
        minor_xp_lead_thresh = score_weight_params["minor_xp_lead_per_laner_threshold"]

        gold_difference = lane_summary_data[f"radiant_gold_at_{kpi_minute_mark}m"] - lane_summary_data[f"dire_gold_at_{kpi_minute_mark}m"]
        xp_difference = lane_summary_data[f"radiant_xp_at_{kpi_minute_mark}m"] - lane_summary_data[f"dire_xp_at_{kpi_minute_mark}m"]
        kill_difference = lane_summary_data["radiant_lane_kills"] - lane_summary_data["dire_lane_kills"]
        radiant_lane_score, dire_lane_score = 0, 0

        # Score based on gold difference (per laner average)
        if num_radiant_laners > 0 and gold_difference > major_gold_lead_thresh * num_radiant_laners : radiant_lane_score += score_weight_params["points_for_major_lead"]
        elif num_dire_laners > 0 and gold_difference < -major_gold_lead_thresh * num_dire_laners : dire_lane_score += score_weight_params["points_for_major_lead"]
        elif num_radiant_laners > 0 and gold_difference > minor_gold_lead_thresh * num_radiant_laners : radiant_lane_score += score_weight_params["points_for_minor_lead"]
        elif num_dire_laners > 0 and gold_difference < -minor_gold_lead_thresh * num_dire_laners : dire_lane_score += score_weight_params["points_for_minor_lead"]

        # Score based on XP difference (per laner average)
        if num_radiant_laners > 0 and xp_difference > major_xp_lead_thresh * num_radiant_laners : radiant_lane_score += score_weight_params["points_for_major_lead"]
        elif num_dire_laners > 0 and xp_difference < -major_xp_lead_thresh * num_dire_laners : dire_lane_score += score_weight_params["points_for_major_lead"]
        elif num_radiant_laners > 0 and xp_difference > minor_xp_lead_thresh * num_radiant_laners : radiant_lane_score += score_weight_params["points_for_minor_lead"]
        elif num_dire_laners > 0 and xp_difference < -minor_xp_lead_thresh * num_dire_laners : dire_lane_score += score_weight_params["points_for_minor_lead"]

        # Score based on kill difference
        if kill_difference >= score_weight_params["kill_difference_for_major_points"] : radiant_lane_score += score_weight_params["points_for_major_kill_difference"]
        elif kill_difference <= -score_weight_params["kill_difference_for_major_points"] : dire_lane_score += score_weight_params["points_for_major_kill_difference"]
        elif kill_difference >= score_weight_params["kill_difference_for_minor_points"] : radiant_lane_score += score_weight_params["points_for_minor_kill_difference"]
        elif kill_difference <= -score_weight_params["kill_difference_for_minor_points"] : dire_lane_score += score_weight_params["points_for_minor_kill_difference"]
        
        # Score based on early tower kills
        current_lane_tower_status = early_tower_status.get(lane_name_key, {"radiant_t1_fallen": False, "dire_t1_fallen": False})
        if current_lane_tower_status["dire_t1_fallen"]: radiant_lane_score += score_weight_params["points_for_early_tower_kill"]
        if current_lane_tower_status["radiant_t1_fallen"]: dire_lane_score += score_weight_params["points_for_early_tower_kill"]

        # Determine lane verdict based on scores
        lane_verdict_text_raw = ""
        # Using a threshold of 1 point difference to declare a winner, otherwise "Even"
        if radiant_lane_score > dire_lane_score + 1: lane_verdict_text_raw = f"Radiant Ahead ({radiant_lane_score} vs {dire_lane_score})"
        elif dire_lane_score > radiant_lane_score + 1: lane_verdict_text_raw = f"Dire Ahead ({dire_lane_score} vs {radiant_lane_score})"
        else: lane_verdict_text_raw = f"Even Lane ({radiant_lane_score} vs {dire_lane_score})"
        
        analysis_results["lanes"][lane_name_key] = {
            "radiant_score": radiant_lane_score, "dire_score": dire_lane_score, "verdict_text": lane_verdict_text_raw,
            "summary_details": {"gold_diff": gold_difference, "xp_diff": xp_difference, "kill_diff": kill_difference, 
                                "radiant_t1_fallen": current_lane_tower_status["radiant_t1_fallen"], 
                                "dire_t1_fallen": current_lane_tower_status["dire_t1_fallen"]}
        }
    return analysis_results

# --- Typer CLI Commands ---

@app.command(name="set-player-name", help="Assign or update a custom name for a player's Account ID.")
def set_player_name(
    account_id: int = typer.Argument(..., help="The player's unique Account ID."), 
    custom_name: str = typer.Argument(..., help="The custom name to assign (e.g., 'SumaiL', 'Miracle-').")
):
    """Allows users to map an account_id to a memorable custom name."""
    console.print(Panel(f"[bold sky_blue1]Setting Custom Name for Account ID {account_id}[/bold sky_blue1]", expand=False, border_style="sky_blue1"))
    db_connection = get_database_connection()
    try:
        # INSERT OR REPLACE will update if account_id exists, or insert if new.
        db_connection.execute(
            "INSERT OR REPLACE INTO player_custom_names (account_id, custom_name, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)", 
            (account_id, custom_name)
        )
        db_connection.commit()
        console.print(f"[green]Successfully set custom name for Account ID {account_id} to '{custom_name}'.[/green]")
    except sqlite3.IntegrityError: # Handles if custom_name is already taken (UNIQUE constraint)
        console.print(Panel(f"[bold red]Error: The custom name '{custom_name}' might already be assigned to another Account ID, or another integrity constraint failed.[/bold red]", title="[bold red]Integrity Error[/bold red]", border_style="red"))
    except sqlite3.Error as error:
        console.print(Panel(f"[bold red]SQLite error setting custom name for Account ID {account_id}:[/bold red]\n{error}", title="[bold red]DB Error[/bold red]", border_style="red"))
    finally:
        db_connection.close()
@app.command(name="list-custom-names", help="Lists all custom player names stored in the database.")
def list_custom_names():
    """Displays all stored custom player names in a table."""
    panel_title = Text.from_markup("[bold sky_blue1]:scroll: Listing All Custom Player Names :scroll:[/bold sky_blue1]")
    console.print(Panel(panel_title, expand=False, border_style="sky_blue1"))
    db_connection = get_database_connection()
    try: 
        name_rows = db_connection.execute("SELECT account_id, custom_name, updated_at FROM player_custom_names ORDER BY custom_name ASC").fetchall()
    except sqlite3.Error as error: 
        console.print(Panel(f"[bold red]SQLite query error for custom names:[/bold red]\n{error}", title="[bold red]DB Error[/bold red]", border_style="red"))
        db_connection.close()
        raise typer.Exit(code=1)
    finally:
        db_connection.close()

    if not name_rows:
        console.print(Padding("[yellow]No custom player names found in the database.[/yellow]", (1,2)))
        return

    table = Table(title="Stored Custom Player Names", show_header=True, header_style="bold magenta", border_style="dim cyan", show_lines=True)
    table.add_column("Account ID", style="dim cyan", width=15, justify="center")
    table.add_column("Custom Name", style="bright_white", min_width=20)
    table.add_column("Last Updated (UTC)", style="dim", min_width=20, justify="center")
    for name_row_data in name_rows:
        table.add_row(str(name_row_data["account_id"]), name_row_data["custom_name"], name_row_data["updated_at"])
    console.print(table)
    console.print(f"\n[dim]Total custom names in DB: {len(name_rows)}[/dim]")

@app.command(name="delete-player-name", help="Deletes a custom player name by Account ID or the Custom Name itself.")
def delete_player_name(identifier_to_delete: str = typer.Argument(..., help="The Account ID or Custom Name to delete.")):
    """Deletes a custom player name mapping."""
    console.print(Panel(f"[bold sky_blue1]Deleting Custom Name for Identifier '{identifier_to_delete}'[/bold sky_blue1]", expand=False, border_style="sky_blue1"))
    db_connection = get_database_connection()
    account_id_to_delete: Optional[int] = None

    # Try to interpret identifier as Account ID first
    try:
        account_id_to_delete = int(identifier_to_delete)
    except ValueError:
        # If not an int, assume it's a custom name and try to find its Account ID
        try:
            name_row = db_connection.execute("SELECT account_id FROM player_custom_names WHERE LOWER(custom_name) = LOWER(?)", (identifier_to_delete,)).fetchone()
            if name_row:
                account_id_to_delete = name_row["account_id"]
            else:
                console.print(f"[yellow]No custom name '{identifier_to_delete}' found.[/yellow]")
                db_connection.close()
                return
        except sqlite3.Error as error:
            console.print(Panel(f"[bold red]SQLite error finding custom name '{identifier_to_delete}':[/bold red]\n{error}", title="[bold red]DB Error[/bold red]", border_style="red"))
            db_connection.close()
            return

    if account_id_to_delete is None: # Should not happen if logic above is correct, but as a safeguard
        console.print(f"[yellow]Could not resolve '{identifier_to_delete}' to an Account ID for deletion.[/yellow]")
        db_connection.close()
        return
    try:
        cursor = db_connection.execute("DELETE FROM player_custom_names WHERE account_id = ?", (account_id_to_delete,))
        db_connection.commit()
        if cursor.rowcount > 0:
            console.print(f"[green]Successfully deleted custom name mapping for Account ID {account_id_to_delete}.[/green]")
        else:
            console.print(f"[yellow]No custom name found for Account ID {account_id_to_delete} to delete (it might have been identified by name but the ID was already removed).[/yellow]")
    except sqlite3.Error as error:
        console.print(Panel(f"[bold red]SQLite error deleting custom name for Account ID {account_id_to_delete}:[/bold red]\n{error}", title="[bold red]DB Error[/bold red]", border_style="red"))
    finally:
        db_connection.close()

@app.command(name="list-heroes", help="Lists all heroes from the 'heroes' table in the database.")
def list_heroes():
    """Displays all heroes stored in the database."""
    panel_title = Text.from_markup("[bold sky_blue1]:shield: Listing All Stored Heroes :shield:[/bold sky_blue1]")
    console.print(Panel(panel_title, expand=False, border_style="sky_blue1"))
    db_connection = get_database_connection()
    try:
        heroes_rows_data = db_connection.execute("SELECT hero_id, name, fetched_at FROM heroes ORDER BY hero_id ASC").fetchall()
    except sqlite3.Error as error:
        console.print(Panel(f"[bold red]SQLite query error for heroes:[/bold red]\n{error}\n[dim]Ensure 'heroes' table exists and is populated.[/dim]", title="[bold red]DB Error[/bold red]", border_style="red"))
        db_connection.close()
        raise typer.Exit(code=1)
    finally:
        db_connection.close()

    if not heroes_rows_data:
        console.print(Padding("[yellow]No heroes found in the database. (Hint: Populate using the main data fetching script)[/yellow]", (1,2)))
        return

    table = Table(title="Stored Heroes (from Database)", show_header=True, header_style="bold magenta", border_style="dim green", show_lines=True)
    table.add_column("Hero ID", style="dim green", width=10, justify="center")
    table.add_column("Internal Name (e.g., npc_dota_hero_...)", style="bright_white", min_width=30) 
    table.add_column("Fetched At (UTC)", style="dim", min_width=20, justify="center")
    for hero_data in heroes_rows_data:
        table.add_row(str(hero_data["hero_id"]), hero_data["name"], hero_data["fetched_at"] or "N/A")
    console.print(table)
    console.print(f"\n[dim]Total heroes in DB: {len(heroes_rows_data)}[/dim]")

@app.command(name="list-matches", help="Lists stored matches with summary details. Use -l for limit, -t to filter by team.")
def list_matches(
    limit_matches: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit the number of matches displayed (most recent first)."),
    search_team_name: Optional[str] = typer.Option(None, "--team", "-t", help="Filter matches by a team name (case-insensitive search in Radiant or Dire team names/tags).")
):
    """Lists matches from the DB, with player and hero details."""
    panel_title = Text.from_markup("[bold sky_blue1]:trophy: Listing Stored Matches with Player Details :trophy:[/bold sky_blue1]")
    console.print(Panel(panel_title, expand=False, border_style="sky_blue1"))
    db_connection = get_database_connection()
    hero_id_to_name_map = _get_all_heroes_id_to_name_map_from_db(db_connection) 
    player_custom_names_map = _get_all_player_custom_names_map_from_db(db_connection)

    query_string = "SELECT match_id, data, fetched_at FROM matches "
    query_parameters_list: list = [] 

    # If team name search, add WHERE clauses (this is a basic JSON text search, not ideal for performance on large DBs)
    if search_team_name:
        # Searching within 'radiant_name', 'dire_name', 'radiant_team' (name/tag), 'dire_team' (name/tag) JSON fields
        query_string += "WHERE (data LIKE ? OR data LIKE ? OR data LIKE ? OR data LIKE ?)" 
        like_pattern = f'%"{search_team_name}"%' # Basic search for the string within JSON
        query_parameters_list.extend([like_pattern] * 4) # Add pattern for each field

    query_string += " ORDER BY fetched_at DESC" # Show most recent first
    if limit_matches is not None and limit_matches > 0:
        query_string += " LIMIT ?"
        query_parameters_list.append(limit_matches)
    
    try:
        match_rows_data = db_connection.execute(query_string, tuple(query_parameters_list)).fetchall()
    except sqlite3.Error as error:
        console.print(Panel(f"[bold red]SQLite query error for matches:[/bold red]\n{error}", title="[bold red]DB Error[/bold red]", border_style="red"))
        db_connection.close()
        raise typer.Exit(code=1)

    if not match_rows_data:
        message = f"[yellow]No matches found in the database{' matching team criteria.' if search_team_name else '.'}[/yellow]"
        console.print(Padding(message, (1,2)))
        db_connection.close()
        return

    table = Table(title="Stored Matches with Player Details", show_header=True, header_style="bold magenta", border_style="dim blue", show_lines=True, expand=True)
    table.add_column("Match ID", style="dim blue", width=12, justify="center")
    table.add_column("Radiant Team", style="bold bright_cyan", min_width=20)
    table.add_column("Dire Team", style="bold bright_red", min_width=20)
    table.add_column("Score (R-D)", justify="center", width=12, style="white")
    table.add_column("Winner", min_width=20, justify="center")
    table.add_column("Radiant Players (Name/ID, Hero)", style="bright_cyan", min_width=45, overflow="fold")
    table.add_column("Dire Players (Name/ID, Hero)", style="bright_red", min_width=45, overflow="fold")
    table.add_column("Fetched At (UTC)", style="dim", min_width=20, justify="center")

    matches_displayed_count = 0
    for match_row_item in match_rows_data:
        match_id = match_row_item["match_id"]
        fetched_at_timestamp = match_row_item["fetched_at"]
        try:
            match_json_data: Dict[str, Any] = json.loads(match_row_item["data"])
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: Could not parse JSON for match ID {match_id} from DB. Skipping.[/yellow]")
            continue

        # Extract team names, handling various possible fields from OpenDota API
        radiant_team_data = match_json_data.get('radiant_team', {})
        dire_team_data = match_json_data.get('dire_team', {})
        radiant_team_name = match_json_data.get('radiant_name') or \
                            (radiant_team_data.get('name') if isinstance(radiant_team_data, dict) else None) or \
                            (radiant_team_data.get('tag') if isinstance(radiant_team_data, dict) else None) or "Radiant"
        dire_team_name = match_json_data.get('dire_name') or \
                         (dire_team_data.get('name') if isinstance(dire_team_data, dict) else None) or \
                         (dire_team_data.get('tag') if isinstance(dire_team_data, dict) else None) or "Dire"

        # If team search, perform a more precise check after JSON parsing (SQL LIKE is broad)
        if search_team_name:
            team_name_lower = search_team_name.lower()
            if not (team_name_lower in radiant_team_name.lower() or \
                    team_name_lower in dire_team_name.lower() or \
                    (isinstance(radiant_team_data, dict) and team_name_lower in str(radiant_team_data.get('tag','')).lower()) or \
                    (isinstance(dire_team_data, dict) and team_name_lower in str(dire_team_data.get('tag','')).lower())):
                continue # Skip if this match doesn't actually match after detailed check

        radiant_score = str(match_json_data.get('radiant_score', '-'))
        dire_score_val = str(match_json_data.get('dire_score', '-'))
        winner_team_name = "N/A"
        if match_json_data.get('radiant_win') is True: winner_team_name = f"[bold bright_cyan]{radiant_team_name}[/bold bright_cyan]"
        elif match_json_data.get('radiant_win') is False: winner_team_name = f"[bold bright_red]{dire_team_name}[/bold bright_red]"

        radiant_players_details_list = []
        dire_players_details_list = []
        players_in_match_data: List[Dict[str, Any]] = match_json_data.get("players", [])

        for player_info_item in players_in_match_data:
            account_id_value = player_info_item.get("account_id")
            # Use custom name, then personaname, then account_id, then N/A
            player_identifier_string = player_custom_names_map.get(account_id_value, 
                                                                  str(account_id_value) if account_id_value is not None 
                                                                  else player_info_item.get("personaname", "N/A"))
            hero_id_value = player_info_item.get("hero_id")
            hero_name_display_string = _get_hero_display_name_from_hero_id(hero_id_value, hero_id_to_name_map) if hero_id_value else "N/A"
            # Using [bold cyan] for hero name styling
            player_detail_string = f"{player_identifier_string} ([bold cyan]{hero_name_display_string}[/bold cyan])" 

            is_radiant_player_flag = player_info_item.get("isRadiant")
            if is_radiant_player_flag is None: # Fallback if 'isRadiant' is missing
                player_slot_value = player_info_item.get("player_slot")
                if player_slot_value is not None:
                    is_radiant_player_flag = player_slot_value < 100 # Slots 0-4 are Radiant
                else:
                    is_radiant_player_flag = False # Default assumption if no slot/isRadiant info

            if is_radiant_player_flag:
                radiant_players_details_list.append(player_detail_string)
            else:
                dire_players_details_list.append(player_detail_string)

        radiant_players_display_string = "\n".join(radiant_players_details_list) if radiant_players_details_list else "N/A"
        dire_players_display_string = "\n".join(dire_players_details_list) if dire_players_details_list else "N/A"
        table.add_row(str(match_id), radiant_team_name, dire_team_name, f"{radiant_score}-{dire_score_val}", winner_team_name, radiant_players_display_string, dire_players_display_string, fetched_at_timestamp)
        matches_displayed_count +=1
    db_connection.close() 

    if matches_displayed_count > 0:
        console.print(table)
        console.print(f"\n[dim]Total matches displayed: {matches_displayed_count}[/dim]")
        if limit_matches and matches_displayed_count >= limit_matches :
            console.print(f"[dim]Showing up to {limit_matches} matches. Use --limit to change.[/dim]")
        elif search_team_name:
            console.print(f"[dim]Showing matches filtered by team: '{search_team_name}'[/dim]")
    elif search_team_name: # Only print this if a search was active and yielded no results after JSON parsing
        console.print(Padding(f"[yellow]No matches found where Radiant or Dire team name/tag contains '{search_team_name}' after parsing JSON.[/yellow]", (1,2)))


@app.command(name="find-player-hero", help="Finds matches for a player, optionally on a specific hero. Use -l for limit.")
def find_player_hero_matches(
    player_identifier_argument: str = typer.Argument(..., help="Player's Account ID, assigned Custom Name, or current Persona Name."),
    hero_name_input_argument: Optional[str] = typer.Argument(None, help="Optional: The name of the hero (e.g., 'antimage', 'npc_dota_hero_juggernaut'). If omitted, lists all matches for the player."),
    limit_matches_option: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit the number of matches displayed.")
):
    """Finds and lists matches for a specific player, optionally filtered by hero played."""
    db_connection = get_database_connection()
    target_hero_id_value: Optional[int] = None
    is_hero_search_active = False
    hero_id_to_name_map = _get_all_heroes_id_to_name_map_from_db(db_connection) 
    player_custom_names_map = _get_all_player_custom_names_map_from_db(db_connection)
    actual_hero_name_for_display = ""

    if hero_name_input_argument:
        is_hero_search_active = True
        target_hero_id_value = get_hero_id_from_name_in_database(hero_name_input_argument, db_connection) 
        if target_hero_id_value is None:
            console.print(Panel(f"[bold red]Hero '{hero_name_input_argument}' not found or ambiguous.[/bold red]\nTry 'list-heroes' to see available internal names.", title="[bold red]Hero Not Found[/bold red]", border_style="red"))
            db_connection.close()
            raise typer.Exit(code=1)
        actual_hero_name_for_display = _get_hero_display_name_from_hero_id(target_hero_id_value, hero_id_to_name_map)
        console.print(f"[info]Targeting Hero ID: {target_hero_id_value} ('{actual_hero_name_for_display}')[/info]")
        panel_title_text = f"[bold sky_blue1]Player '{player_identifier_argument}' on Hero '{actual_hero_name_for_display}'[/bold sky_blue1]"
        table_title_suffix_text = f" on Hero '{actual_hero_name_for_display}' (ID: {target_hero_id_value})"
    else:
        panel_title_text = f"[bold sky_blue1]All matches for Player '{player_identifier_argument}'[/bold sky_blue1]"
        table_title_suffix_text = ""
        console.print(f"[info]Listing all matches for player identifier '{player_identifier_argument}'[/info]")
    console.print(Panel(panel_title_text, expand=False, border_style="sky_blue1"))

    # Resolve player identifier to account_id if possible
    search_target_account_id_value: Optional[int] = None
    search_by_personaname_fallback_flag = False
    display_search_term_string = player_identifier_argument # What to show in messages
    try:
        search_target_account_id_value = int(player_identifier_argument)
        custom_name_for_id = player_custom_names_map.get(search_target_account_id_value)
        display_search_term_string = f"{custom_name_for_id} (ID: {search_target_account_id_value})" if custom_name_for_id else f"ID: {search_target_account_id_value}"
        console.print(f"[dim]Searching by Account ID: {search_target_account_id_value}[/dim]")
    except ValueError: # Not an integer, try custom name then personaname
        found_in_custom_names = False
        for account_id_val, custom_name_val in player_custom_names_map.items():
            if player_identifier_argument.lower() == custom_name_val.lower():
                search_target_account_id_value = account_id_val
                display_search_term_string = f"{custom_name_val} (resolved to ID: {account_id_val})"
                console.print(f"[dim]Searching by Custom Name '{player_identifier_argument}', resolved to Account ID: {search_target_account_id_value}[/dim]")
                found_in_custom_names = True
                break
        if not found_in_custom_names:
            search_by_personaname_fallback_flag = True # Fallback to searching personaname in JSON
            display_search_term_string = f"Persona Name containing '{player_identifier_argument}'"
            console.print(f"[dim]Searching by Persona Name containing: '{player_identifier_argument}'[/dim]")

    # Initial SQL query: broad filter by account_id (if known) or personaname substring in JSON data
    # This is a pre-filter; detailed check happens after JSON parsing.
    query_string = "SELECT match_id, data, fetched_at FROM matches WHERE data LIKE ? ORDER BY fetched_at DESC"
    # Use account_id for LIKE if available, otherwise the raw player_identifier_argument for personaname search
    sql_search_term_string = str(search_target_account_id_value) if search_target_account_id_value is not None else player_identifier_argument
    query_parameters = (f'%{sql_search_term_string}%',) # Wildcards for LIKE

    try:
        match_rows_data = db_connection.execute(query_string, query_parameters).fetchall()
    except sqlite3.Error as error:
        console.print(Panel(f"[bold red]SQLite query error:[/bold red]\n{error}", title="[bold red]DB Error[/bold red]", border_style="red"))
        db_connection.close()
        raise typer.Exit(code=1)

    if not match_rows_data:
        console.print(Padding(f"[yellow]No matches in DB potentially involving '{player_identifier_argument}' based on initial SQL filter.[/yellow]", (1,2)))
        db_connection.close()
        return

    found_matches_details_list = []
    for match_row_item in match_rows_data:
        try:
            match_json_data: Dict[str, Any] = json.loads(match_row_item["data"])
        except json.JSONDecodeError:
            continue # Skip if JSON is invalid

        player_found_in_match_and_met_criteria_flag = False
        hero_played_by_target_player_id_in_this_match: Optional[int] = None

        # Iterate through players in this match's JSON to find the target player
        for player_info_item in match_json_data.get("players", []):
            player_matches_identifier_criteria_flag = False
            if search_target_account_id_value is not None: # If we have an account_id
                if player_info_item.get("account_id") == search_target_account_id_value:
                    player_matches_identifier_criteria_flag = True
            elif search_by_personaname_fallback_flag: # If searching by personaname
                current_player_name = player_info_item.get("personaname")
                if current_player_name and isinstance(current_player_name, str) and \
                   player_identifier_argument.lower() in current_player_name.lower():
                    player_matches_identifier_criteria_flag = True

            if player_matches_identifier_criteria_flag:
                hero_played_by_target_player_id_in_this_match = player_info_item.get("hero_id")
                if is_hero_search_active: # If also searching for a specific hero
                    if hero_played_by_target_player_id_in_this_match == target_hero_id_value:
                        player_found_in_match_and_met_criteria_flag = True
                        break # Found player on correct hero
                else: # Not searching for a specific hero, just the player
                    player_found_in_match_and_met_criteria_flag = True
                    break # Found player

        if player_found_in_match_and_met_criteria_flag:
            # Extract match details for the table
            radiant_team_data = match_json_data.get('radiant_team', {})
            dire_team_data = match_json_data.get('dire_team', {})
            radiant_team_name = match_json_data.get('radiant_name') or (radiant_team_data.get('name') if isinstance(radiant_team_data, dict) else None) or (radiant_team_data.get('tag') if isinstance(radiant_team_data, dict) else None) or "Radiant"
            dire_team_name = match_json_data.get('dire_name') or (dire_team_data.get('name') if isinstance(dire_team_data, dict) else None) or (dire_team_data.get('tag') if isinstance(dire_team_data, dict) else None) or "Dire"

            radiant_score_val = str(match_json_data.get('radiant_score', '-'))
            dire_score_val = str(match_json_data.get('dire_score', '-'))
            winner_team_name = "N/A"
            if match_json_data.get('radiant_win') is True: winner_team_name = f"[bold bright_cyan]{radiant_team_name}[/bold bright_cyan]"
            elif match_json_data.get('radiant_win') is False: winner_team_name = f"[bold bright_red]{dire_team_name}[/bold bright_red]"

            player_hero_display_name = _get_hero_display_name_from_hero_id(hero_played_by_target_player_id_in_this_match, hero_id_to_name_map) if hero_played_by_target_player_id_in_this_match else "N/A"
            
            found_matches_details_list.append({
                "match_id": str(match_row_item["match_id"]),
                "radiant_name": radiant_team_name,
                "dire_name": dire_team_name,
                "score": f"{radiant_score_val}-{dire_score_val}",
                "winner": winner_team_name,
                "fetched_at": match_row_item["fetched_at"],
                "player_hero": player_hero_display_name # Hero played by the target player
            })
            if limit_matches_option and len(found_matches_details_list) >= limit_matches_option:
                break # Stop if limit reached
    db_connection.close()

    if not found_matches_details_list:
        message = f"[yellow]No matches found where player '{display_search_term_string}'"
        if is_hero_search_active and target_hero_id_value is not None:
             message += f" played hero '{actual_hero_name_for_display}'."
        else:
            message += "."
        console.print(Padding(message, (1,2)))
        return

    table_title_string = f"Matches: Player '{display_search_term_string}'{table_title_suffix_text}"
    table = Table(title=table_title_string, show_header=True, header_style="bold magenta", border_style="white", show_lines=True)
    table.add_column("Match ID", style="white", width=12, justify="center")
    table.add_column("Player's Hero", style="bold cyan", min_width=20) # Corrected style
    table.add_column("Radiant Team", style="bold bright_cyan", min_width=20)
    table.add_column("Dire Team", style="bold bright_red", min_width=20)
    table.add_column("Score (R-D)", justify="center", width=12, style="white")
    table.add_column("Winner", min_width=20, justify="center")
    table.add_column("Fetched At (UTC)", style="dim", min_width=20, justify="center")
    for detail_item in found_matches_details_list:
        table.add_row(detail_item["match_id"], detail_item["player_hero"], detail_item["radiant_name"], detail_item["dire_name"], detail_item["score"], detail_item["winner"], detail_item["fetched_at"])
    console.print(table)
    console.print(f"\n[dim]Found {len(found_matches_details_list)} match(es) meeting the criteria.[/dim]")
    if limit_matches_option and len(found_matches_details_list) >= limit_matches_option:
        console.print(f"[dim]Showing up to {limit_matches_option} matches. Use --limit to change.[/dim]")

@app.command(name="show-match-json", help="Shows JSON for a match. Use -v to visualize, -o to save to file.")
def show_match_json(
    match_id_argument: int = typer.Argument(..., help="The Match ID."),
    visualize_flag: bool = typer.Option(False, "--visualize", "-v", help="Generate an interactive visual map of the JSON structure (opens in browser). Requires gravis and networkx.", is_flag=True),
    output_file_path: Optional[str] = typer.Option(None, "--output-file", "-o", help="Path to save the raw JSON data (e.g., match_data.json).", show_default=False, dir_okay=False, writable=True )
):
    """Displays or saves the raw JSON data for a specific match."""
    console.print(Panel(f"[bold sky_blue1]JSON Data for Match ID {match_id_argument}[/bold sky_blue1]", expand=False, border_style="sky_blue1"))
    db_connection = get_database_connection()
    match_row_data: Optional[sqlite3.Row] = None
    output_file_absolute_path: Optional[str] = None

    try:
        match_row_data = db_connection.execute("SELECT data FROM matches WHERE match_id = ?", (match_id_argument,)).fetchone()
    except sqlite3.Error as error:
        console.print(Panel(f"[bold red]SQLite query error for match JSON:[/bold red]\n{error}", title="[bold red]DB Error[/bold red]", border_style="red"))
        if db_connection: db_connection.close()
        raise typer.Exit(code=1)

    if not match_row_data or not match_row_data["data"]:
        console.print(Padding(f"[yellow]No data found in DB for match ID {match_id_argument}.[/yellow]", (1,2)))
        if db_connection: db_connection.close()
        return

    json_string_data_from_db = match_row_data["data"]
    match_json_parsed_data: Optional[Dict[str, Any]] = None
    visualization_was_successful = False

    try:
        match_json_parsed_data = json.loads(json_string_data_from_db) # Try to parse the JSON
        
        if output_file_path: # If output file is requested
            try:
                output_file_absolute_path = os.path.abspath(output_file_path)
                output_directory = os.path.dirname(output_file_absolute_path)
                if output_directory and not os.path.exists(output_directory): # Create dir if not exists
                    os.makedirs(output_directory)
                    console.print(f"[info]Created directory: {output_directory}[/info]")
                with open(output_file_absolute_path, 'w') as file_writer:
                    json.dump(match_json_parsed_data, file_writer, indent=2) # Write formatted JSON
                console.print(f"[green]Successfully saved JSON data for match {match_id_argument} to: {output_file_absolute_path}[/green]")
            except Exception as file_system_error: # Catch broad exceptions for file operations
                console.print(Panel(f"[bold red]An unexpected error occurred while preparing to save to file '{output_file_path}':[/bold red]\n{file_system_error}", title="[bold red]File Save Error[/bold red]", border_style="red"))
        
        if visualize_flag: # If visualization is requested
            console.print("[info]Visualization requested. Attempting to generate graph...[/info]")
            if not VISUALIZATION_LIBRARIES_AVAILABLE: 
                console.print(Panel("[bold yellow]Visualization libraries (gravis, networkx) are not installed. Cannot visualize.[/bold yellow]\nTo install them, run: `pip install gravis networkx`", title="[yellow]Missing Libraries[/yellow]", border_style="yellow"))
            elif match_json_parsed_data is not None: # Ensure JSON was parsed
                visualization_was_successful = _visualize_json_structure_with_gravis(match_json_parsed_data, match_id_argument, "Match")
                if not visualization_was_successful:
                    console.print("[yellow]Visualization generation failed or was skipped.[/yellow]")
            else:
                console.print("[yellow]Cannot visualize as JSON data was not parsed successfully.[/yellow]")
        
        # Determine if JSON should be printed to console
        should_print_to_console = True
        if output_file_path and output_file_absolute_path and os.path.exists(output_file_absolute_path):
            should_print_to_console = False # Don't print if successfully saved to file
        if visualize_flag and visualization_was_successful:
            should_print_to_console = False # Don't print if successfully visualized
        
        # If saving/visualization failed or wasn't requested, but user might still want to see it
        if not should_print_to_console and \
           ((visualize_flag and not visualization_was_successful) or \
            (output_file_path and (not output_file_absolute_path or not os.path.exists(output_file_absolute_path)))):
            if typer.confirm("Visualization/Saving failed or was not performed. Print JSON to console instead?", default=False):
                should_print_to_console = True
        
        if should_print_to_console and match_json_parsed_data:
            console.print(Panel(Syntax(json.dumps(match_json_parsed_data, indent=2), "json", theme="paraiso-dark", line_numbers=True, word_wrap=True), title=f"JSON for Match ID {match_id_argument}", border_style="green", expand=False))
    
    except json.JSONDecodeError: # If the data from DB is not valid JSON
        console.print(Panel(f"[bold red]Error: Stored data for match ID {match_id_argument} is not valid JSON.[/bold red]", title="[bold red]JSON Error[/bold red]", border_style="red"))
        if output_file_path: # Still try to save the raw, invalid string if requested
            try:
                output_file_absolute_path = os.path.abspath(output_file_path)
                output_directory = os.path.dirname(output_file_absolute_path)
                if output_directory and not os.path.exists(output_directory): os.makedirs(output_directory)
                with open(output_file_absolute_path, 'w') as file_writer:
                    file_writer.write(json_string_data_from_db)
                console.print(f"[yellow]Saved raw (invalid JSON) data for match {match_id_argument} to: {output_file_absolute_path}[/yellow]")
            except IOError as io_error:
                console.print(Panel(f"[bold red]Error saving raw invalid JSON to file '{output_file_path}':[/bold red]\n{io_error}", title="[bold red]File Save Error[/bold red]", border_style="red"))
        else: # If not saving, print a snippet of the invalid data
            console.print("[info]Raw data from DB (which is not valid JSON):")
            console.print(json_string_data_from_db[:1000] + "..." if len(json_string_data_from_db) > 1000 else json_string_data_from_db)
    finally:
        if db_connection: db_connection.close()

@app.command(name="db-stats", help="Shows basic statistics about the database (match counts, hero counts, etc.).")
def database_statistics():
    """Displays overall statistics for the connected database."""
    panel_title = Text.from_markup("[bold sky_blue1]:bar_chart: Database Statistics :bar_chart:[/bold sky_blue1]")
    console.print(Panel(panel_title, expand=False, border_style="sky_blue1"))
    db_connection = get_database_connection()
    statistics_dictionary = {}
    try:
        statistics_dictionary["Total Matches"] = db_connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        statistics_dictionary["Total Heroes"] = db_connection.execute("SELECT COUNT(*) FROM heroes").fetchone()[0]
        statistics_dictionary["Total Custom Player Names"] = db_connection.execute("SELECT COUNT(*) FROM player_custom_names").fetchone()[0]
        
        # Get date range of fetched matches
        match_dates_row = db_connection.execute("SELECT MIN(fetched_at), MAX(fetched_at) FROM matches WHERE fetched_at IS NOT NULL AND fetched_at != ''").fetchone()
        if match_dates_row and match_dates_row[0] is not None and match_dates_row[1] is not None:
            statistics_dictionary["Matches Fetched Between (UTC)"] = f"{match_dates_row[0]} [dim]and[/dim] {match_dates_row[1]}"
        else:
            statistics_dictionary["Matches Fetched Between (UTC)"] = "N/A"
            
    except sqlite3.Error as error:
        console.print(Panel(f"[bold red]SQLite query error for DB stats:[/bold red]\n{error}", title="[bold red]DB Error[/bold red]", border_style="red"))
        if db_connection: db_connection.close()
        raise typer.Exit(code=1)
    finally:
        if db_connection: db_connection.close()

    table = Table(title="Database Overview", show_lines=True, header_style="bold magenta", border_style="dim yellow")
    table.add_column("Statistic", style="bold yellow", min_width=30)
    table.add_column("Value", style="bright_white")
    for key, value_item in statistics_dictionary.items():
        table.add_row(key, str(value_item))
    console.print(table)

@app.command(name="player-summary", help="Summary for a player (matches, top heroes, win rate).")
def player_summary(player_identifier_argument: str = typer.Argument(..., help="Player's Account ID, Custom Name, or current Persona Name.")):
    """Generates a performance summary for a specified player."""
    panel_title = Text.from_markup(f"[bold sky_blue1]:bust_in_silhouette: Summary for Player '{player_identifier_argument}' :bust_in_silhouette:[/bold sky_blue1]")
    console.print(Panel(panel_title, expand=False, border_style="sky_blue1"))
    db_connection = get_database_connection()
    player_custom_names_map = _get_all_player_custom_names_map_from_db(db_connection)
    hero_id_to_name_map = _get_all_heroes_id_to_name_map_from_db(db_connection) 

    target_account_id_value: Optional[int] = None
    search_by_personaname_fallback_flag = False
    display_player_term_string = player_identifier_argument # For display messages
    sql_like_prefilter_term_string: str # For the initial SQL LIKE clause

    # Resolve player identifier
    try:
        target_account_id_value = int(player_identifier_argument)
        custom_name = player_custom_names_map.get(target_account_id_value)
        display_player_term_string = f"{custom_name} (ID: {target_account_id_value})" if custom_name else f"ID: {target_account_id_value}"
        # For account ID, search for the exact account_id field in JSON
        sql_like_prefilter_term_string = f'%"account_id": {target_account_id_value}%' 
        console.print(f"[dim]Interpreted player identifier as Account ID: {target_account_id_value}[/dim]")
    except ValueError: # Not an int, try custom name then personaname
        resolved_by_custom_name_flag = False
        for account_id_val, custom_name_val in player_custom_names_map.items():
            if player_identifier_argument.lower() == custom_name_val.lower():
                target_account_id_value = account_id_val
                display_player_term_string = f"{custom_name_val} (resolved to ID: {account_id_val})"
                sql_like_prefilter_term_string = f'%"account_id": {target_account_id_value}%'
                console.print(f"[dim]Resolved Custom Name '{player_identifier_argument}' to Account ID: {target_account_id_value}[/dim]")
                resolved_by_custom_name_flag = True
                break
        if not resolved_by_custom_name_flag:
            search_by_personaname_fallback_flag = True
            # Escape quotes for JSON string search if searching by personaname
            escaped_player_identifier = player_identifier_argument.replace('"', '""') 
            sql_like_prefilter_term_string = f'%"personaname": "%{escaped_player_identifier}%"%' # Search for substring in personaname
            console.print(f"[dim]Searching by Persona Name containing: '{player_identifier_argument}'[/dim]")

    query_string = "SELECT data FROM matches WHERE data LIKE ? ORDER BY fetched_at DESC"
    query_parameters = (sql_like_prefilter_term_string,)

    matches_played_count = 0
    wins_count = 0
    hero_performance_data: Dict[int, Dict[str, int]] = defaultdict(lambda: {"picks": 0, "wins": 0})

    try:
        match_rows_data = db_connection.execute(query_string, query_parameters).fetchall()
    except sqlite3.Error as error:
        console.print(Panel(f"[bold red]SQLite error fetching matches for player summary:[/bold red]\n{error}", title="[bold red]DB Error[/bold red]", border_style="red"))
        if db_connection: db_connection.close()
        return

    if not match_rows_data:
        console.print(Padding(f"[yellow]No matches found in DB potentially involving '{display_player_term_string}' based on initial SQL filter.[/yellow]", (1,2)))
        if db_connection: db_connection.close()
        return

    # Process each potentially relevant match
    for match_row_item in match_rows_data:
        try:
            match_json_data: Dict[str, Any] = json.loads(match_row_item["data"])
        except json.JSONDecodeError:
            continue # Skip invalid JSON

        player_in_this_match_info = None # Store the dict of the target player in this match
        # Detailed check for the player within the parsed JSON
        for player_info_item_loop in match_json_data.get("players", []):
            current_account_id_value = player_info_item_loop.get("account_id")
            current_personaname_value = player_info_item_loop.get("personaname")
            player_matches_criteria_flag = False

            if target_account_id_value is not None: # If we resolved to an account_id
                if current_account_id_value == target_account_id_value:
                    player_matches_criteria_flag = True
            elif search_by_personaname_fallback_flag: # If searching by personaname substring
                if current_personaname_value and isinstance(current_personaname_value, str) and \
                   player_identifier_argument.lower() in current_personaname_value.lower():
                    player_matches_criteria_flag = True
            
            if player_matches_criteria_flag:
                player_in_this_match_info = player_info_item_loop
                break # Found the player in this match

        if player_in_this_match_info:
            matches_played_count += 1
            hero_id_value = player_in_this_match_info.get("hero_id")
            
            # Determine if the player won
            player_is_radiant_value = player_in_this_match_info.get("isRadiant")
            if player_is_radiant_value is None and "player_slot" in player_in_this_match_info: # Fallback
                player_is_radiant_value = player_in_this_match_info["player_slot"] < 100

            radiant_won_match_flag = match_json_data.get("radiant_win") 
            player_won_this_match_flag = False
            if radiant_won_match_flag is not None and player_is_radiant_value is not None:
                if (radiant_won_match_flag is True and player_is_radiant_value) or \
                   (radiant_won_match_flag is False and not player_is_radiant_value):
                    wins_count += 1
                    player_won_this_match_flag = True

            if hero_id_value: # Track hero performance
                hero_performance_data[hero_id_value]["picks"] += 1
                if player_won_this_match_flag:
                    hero_performance_data[hero_id_value]["wins"] += 1
    if db_connection: db_connection.close()

    if matches_played_count == 0:
        console.print(Padding(f"[yellow]No confirmed matches found for player '{display_player_term_string}' after detailed check.[/yellow]", (1,2)))
        return

    win_rate_percentage = (wins_count / matches_played_count * 100) if matches_played_count > 0 else 0

    summary_text = Text.from_markup(f"[b]Player:[/b] {display_player_term_string}\n"
                                     f"[b]Total Matches Found:[/b] {matches_played_count}\n"
                                     f"[b]Wins:[/b] {wins_count}\n"
                                     f"[b]Win Rate:[/b] {win_rate_percentage:.2f}%")
    console.print(Panel(summary_text, title="[b]Overall Player Performance[/b]", border_style="green", expand=False))

    if hero_performance_data:
        console.print("\n[bold bright_magenta]Most Played Heroes (Top 5):[/bold bright_magenta]")
        # Sort heroes by picks, then by win rate
        sorted_hero_performance_list = sorted(
            hero_performance_data.items(),
            key=lambda item: (item[1]["picks"], (item[1]['wins'] / item[1]['picks'] * 100) if item[1]['picks'] > 0 else 0),
            reverse=True
        )

        top_heroes_table = Table(show_header=True, header_style="bold cyan", border_style="dim magenta", show_lines=True)
        top_heroes_table.add_column("Hero", style="bold cyan") # Corrected style
        top_heroes_table.add_column("Picks", style="magenta", justify="center")
        top_heroes_table.add_column("Win Rate (%)", style="green", justify="center")

        for hero_id_value, stats_dictionary in sorted_hero_performance_list[:5]: # Display top 5
            hero_display_name_string = _get_hero_display_name_from_hero_id(hero_id_value, hero_id_to_name_map) 
            hero_win_rate = (stats_dictionary['wins'] / stats_dictionary['picks'] * 100) if stats_dictionary['picks'] > 0 else 0
            top_heroes_table.add_row(hero_display_name_string, str(stats_dictionary['picks']), f"{hero_win_rate:.2f}%")
        console.print(top_heroes_table)
    else:
        console.print(Padding("[yellow]No hero pick data available for this player in the stored matches.[/yellow]", (1,0)))

@app.command(name="hero-summary", help="Summary for a hero (pick count, win rate, top players).")
def hero_summary(hero_identifier_argument: str = typer.Argument(..., help="The name (e.g. 'antimage') or ID of the hero.")):
    """Generates a performance summary for a specified hero."""
    panel_title = Text.from_markup(f"[bold sky_blue1]:dragon_face: Summary for Hero '{hero_identifier_argument}' :dragon_face:[/bold sky_blue1]")
    console.print(Panel(panel_title, expand=False, border_style="sky_blue1"))
    db_connection = get_database_connection()
    hero_id_to_name_map = _get_all_heroes_id_to_name_map_from_db(db_connection) 
    player_custom_names_map = _get_all_player_custom_names_map_from_db(db_connection)
    
    target_hero_id_value: Optional[int] = None
    # Resolve hero identifier
    try:
        target_hero_id_value = int(hero_identifier_argument)
        console.print(f"[dim]Interpreted hero identifier as Hero ID: {target_hero_id_value}[/dim]")
    except ValueError: # Not an int, try name
        target_hero_id_value = get_hero_id_from_name_in_database(hero_identifier_argument, db_connection) 
        if target_hero_id_value is None:
            console.print(Panel(f"[bold red]Hero name '{hero_identifier_argument}' not found or ambiguous. Try 'list-heroes'.[/bold red]", title="[bold red]Hero Not Found[/bold red]", border_style="red"))
            if db_connection: db_connection.close()
            return
        console.print(f"[dim]Resolved hero name '{hero_identifier_argument}' to ID: {target_hero_id_value}[/dim]")

    actual_hero_name_for_display = _get_hero_display_name_from_hero_id(target_hero_id_value, hero_id_to_name_map)
    # SQL pre-filter for matches containing this hero_id
    sql_like_prefilter_term_string = f'%"hero_id": {target_hero_id_value}%' 
    query_string = "SELECT data FROM matches WHERE data LIKE ? ORDER BY fetched_at DESC"
    query_parameters = (sql_like_prefilter_term_string,)

    total_hero_picks_count = 0
    total_hero_wins_count = 0 
    # Store performance of players with this hero: {account_id: {"picks": x, "wins": y}}
    player_performance_with_hero_data: Dict[Any, Dict[str, int]] = defaultdict(lambda: {"picks": 0, "wins": 0})

    try:
        match_rows_data = db_connection.execute(query_string, query_parameters).fetchall()
    except sqlite3.Error as error:
        console.print(Panel(f"[bold red]SQLite error fetching matches for hero summary:[/bold red]\n{error}", title="[bold red]DB Error[/bold red]", border_style="red"))
        if db_connection: db_connection.close()
        return

    if not match_rows_data:
        console.print(Padding(f"[yellow]No matches found in DB potentially featuring hero '{actual_hero_name_for_display}' (ID: {target_hero_id_value}) based on initial SQL filter.[/yellow]", (1,2)))
        if db_connection: db_connection.close()
        return

    # Process each potentially relevant match
    for match_row_item in match_rows_data:
        try:
            match_json_data: Dict[str, Any] = json.loads(match_row_item["data"])
        except json.JSONDecodeError:
            continue

        player_who_picked_hero_info = None # Details of the player who picked the target hero
        # Find the player who picked the target hero in this match
        for player_info_item_loop in match_json_data.get("players", []):
            if player_info_item_loop.get("hero_id") == target_hero_id_value:
                player_who_picked_hero_info = player_info_item_loop
                break 

        if player_who_picked_hero_info:
            total_hero_picks_count += 1
            account_id_of_picker = player_who_picked_hero_info.get("account_id")
            
            # Determine if the picker won
            picker_is_radiant_value = player_who_picked_hero_info.get("isRadiant")
            if picker_is_radiant_value is None and "player_slot" in player_who_picked_hero_info: # Fallback
                picker_is_radiant_value = player_who_picked_hero_info["player_slot"] < 100

            radiant_won_match_flag = match_json_data.get("radiant_win")
            picker_won_this_match_flag = False
            if radiant_won_match_flag is not None and picker_is_radiant_value is not None:
                if (radiant_won_match_flag is True and picker_is_radiant_value) or \
                   (radiant_won_match_flag is False and not picker_is_radiant_value):
                    total_hero_wins_count += 1
                    picker_won_this_match_flag = True

            # Use account_id as key, or "UnknownPlayer" if account_id is missing
            player_key_for_dict = account_id_of_picker if account_id_of_picker is not None else "UnknownPlayer"
            player_performance_with_hero_data[player_key_for_dict]["picks"] += 1
            if picker_won_this_match_flag:
                player_performance_with_hero_data[player_key_for_dict]["wins"] += 1
    if db_connection: db_connection.close()

    if total_hero_picks_count == 0:
        console.print(Padding(f"[yellow]Hero '{actual_hero_name_for_display}' (ID: {target_hero_id_value}) was not confirmed picked in any stored matches after detailed check.[/yellow]", (1,2)))
        return

    overall_hero_win_rate_percentage = (total_hero_wins_count / total_hero_picks_count * 100) if total_hero_picks_count > 0 else 0
    
    hero_summary_text = Text.from_markup(f"[b]Hero:[/b] {actual_hero_name_for_display} (ID: {target_hero_id_value})\n"
                                          f"[b]Total Picks in DB:[/b] {total_hero_picks_count}\n"
                                          f"[b]Overall Wins (when hero played):[/b] {total_hero_wins_count}\n"
                                          f"[b]Overall Win Rate (when hero played):[/b] {overall_hero_win_rate_percentage:.2f}%")
    console.print(Panel(hero_summary_text, title="[b]Overall Hero Performance[/b]", border_style="orange3", expand=False))


    if player_performance_with_hero_data:
        console.print("\n[bold bright_yellow]Most Frequent Players (Top 5, sorted by picks then win rate):[/bold bright_yellow]")
        
        # Filter out "UnknownPlayer" before sorting for display
        valid_player_performance_data = {k: v for k, v in player_performance_with_hero_data.items() if k != "UnknownPlayer"}
        sorted_player_performance_list = sorted(
            valid_player_performance_data.items(),
            key=lambda item: (item[1]["picks"], (item[1]['wins'] / item[1]['picks'] * 100) if item[1]['picks'] > 0 else 0), # Sort by picks, then win rate
            reverse=True
        )

        top_players_table = Table(show_header=True, header_style="bold cyan", border_style="dim yellow", show_lines=True)
        top_players_table.add_column("Player (Name/ID)", style="green")
        top_players_table.add_column("Picks of Hero", style="magenta", justify="center")
        top_players_table.add_column("Win Rate (%) with Hero", style="blue", justify="center")

        for account_id_value, stats_dictionary in sorted_player_performance_list[:5]: # Display top 5
            player_display_name_string = player_custom_names_map.get(account_id_value, f"ID: {account_id_value}") # Use custom name or ID
            player_hero_win_rate = (stats_dictionary['wins'] / stats_dictionary['picks'] * 100) if stats_dictionary['picks'] > 0 else 0
            top_players_table.add_row(player_display_name_string, str(stats_dictionary['picks']), f"{player_hero_win_rate:.2f}%")
        console.print(top_players_table)
    else:
        console.print(Padding("[yellow]No specific player performance data available for this hero.[/yellow]", (1,0)))

@app.command(name="analyze-lanes", help="Analyzes laning phase for a match ID (uses configured KPIs).")
def analyze_lanes_command(match_id_argument: int = typer.Argument(..., help="The Match ID to analyze.")):
    """Analyzes and displays laning phase performance for a given match."""
    panel_title = Text.from_markup(f"[bold sky_blue1]:chart_with_upwards_trend: Laning Phase Analysis for Match ID {match_id_argument} :chart_with_downwards_trend:[/bold sky_blue1]")
    console.print(Panel(panel_title, expand=False, border_style="sky_blue1"))
    db_connection = get_database_connection()
    hero_id_to_name_map = _get_all_heroes_id_to_name_map_from_db(db_connection)
    player_custom_names_map = _get_all_player_custom_names_map_from_db(db_connection)

    # Perform the core analysis logic
    analysis_data = _perform_core_lane_analysis_for_match(match_id_argument, db_connection, hero_id_to_name_map, player_custom_names_map)

    if not analysis_data or analysis_data.get("error"):
        console.print(Panel(f"[bold red]Could not analyze lanes for match {match_id_argument}:[/bold red]\n{analysis_data.get('error', 'Unknown error')}", title="[bold red]Analysis Error[/bold red]", border_style="red"))
        if db_connection: db_connection.close()
        return

    # Re-fetch match data for display purposes (player names, heroes in lanes)
    # This is done because _perform_core_lane_analysis_for_match might not return all raw player details needed for the tables.
    try: 
        match_row_data = db_connection.execute("SELECT data FROM matches WHERE match_id = ?", (match_id_argument,)).fetchone()
        match_json_data_for_tables = json.loads(match_row_data["data"]) if match_row_data and match_row_data["data"] else None
    except Exception as error: # Catch broad error as this is for display enhancement
        console.print(Panel(f"[bold red]Error re-fetching match data for console display:[/bold red]\n{error}", title="[bold red]DB Error[/bold red]", border_style="red"))
        if db_connection: db_connection.close() 
        return # Can proceed with analysis_data even if this fails, but tables will be less detailed
    finally:
        if db_connection: db_connection.close() # Close connection from this scope

    if not match_json_data_for_tables:
        console.print(Panel(f"[bold red]Could not re-fetch match data for console display for match {match_id_argument}. Detailed player tables might be incomplete.[/bold red]", title="[bold red]Data Error[/bold red]", border_style="red"))
        # Fallback: use players_match_data from within _perform_core_lane_analysis_for_match if it were accessible
        # For now, we'll proceed, but tables might lack some player-specific info if this block fails.
        players_data_for_tables_display = [] # Empty list to prevent errors below
        lanes_assignments_for_tables_display = {"top":{}, "mid":{}, "bot":{}} # Empty structure
    else:
        players_data_for_tables_display = match_json_data_for_tables.get("players", [])
        # Enrich this player data with display names and isRadiant, similar to _perform_core_lane_analysis_for_match
        for player_index, player_info_item in enumerate(players_data_for_tables_display):
            player_account_id = player_info_item.get("account_id")
            display_name = player_custom_names_map.get(player_account_id, player_info_item.get("personaname", f"Slot {player_info_item.get('player_slot', player_index)}"))
            players_data_for_tables_display[player_index]["display_name"] = display_name
            if "isRadiant" not in player_info_item and "player_slot" in player_info_item:
                 players_data_for_tables_display[player_index]["isRadiant"] = player_info_item["player_slot"] < 100
        lanes_assignments_for_tables_display = _identify_laning_players(players_data_for_tables_display, hero_id_to_name_map)
    
    console.print(Padding(f"[info]Draft Order: {analysis_data.get('draft_order', 'N/A')}[/info]", (1,0,1,0)))

    primary_minute_mark = KPI_PARAMETERS["analysis_minute_mark"]
    secondary_minute_mark = KPI_PARAMETERS["display_secondary_minute_mark"]
    tower_limit_display_minutes = KPI_PARAMETERS['early_tower_kill_time_limit_seconds'] // 60
    kill_limit_display_minutes = KPI_PARAMETERS['kill_death_analysis_time_limit_seconds'] // 60

    # Display analysis for each lane
    for lane_name_key, lane_information in analysis_data.get("lanes", {}).items():
        if lane_name_key == "unknown": continue # Skip "unknown" lane for this detailed display

        console.print(Panel(f"Lane Analysis: [b]{lane_name_key.upper()}[/b]", style="bold yellow", expand=False, border_style="yellow"))
        
        teams_in_lane_for_table_display = lanes_assignments_for_tables_display.get(lane_name_key, {"radiant": [], "dire": []})

        # Display detailed player stats table for each team in the lane
        for team_name_string, players_in_lane_for_team_display in teams_in_lane_for_table_display.items():
            if not players_in_lane_for_team_display:
                console.print(Padding(f"[dim]No {team_name_string} players in {lane_name_key.upper()} lane for detailed table.[/dim]", (0,2)))
                continue
            
            team_color_style = "bright_cyan" if team_name_string == "radiant" else "bright_red"
            table = Table(
                title=f"{team_name_string.capitalize()} - {lane_name_key.upper()} @ {secondary_minute_mark}m / {primary_minute_mark}m",
                title_style=f"bold {team_color_style}", show_header=True, header_style="bold magenta", 
                border_style=f"dim {team_color_style}", show_lines=True
            )
            table.add_column("Player", style="white", min_width=15, overflow="fold")
            table.add_column("Hero", style="bold cyan", min_width=15) # Corrected style
            table.add_column(f"Lvl ({secondary_minute_mark}m/{primary_minute_mark}m)", justify="center")
            table.add_column(f"LH ({secondary_minute_mark}m/{primary_minute_mark}m)", justify="center")
            table.add_column(f"DN ({secondary_minute_mark}m/{primary_minute_mark}m)", justify="center")
            table.add_column(f"Gold ({secondary_minute_mark}m/{primary_minute_mark}m)", justify="center")
            table.add_column(f"GPM ({secondary_minute_mark}m/{primary_minute_mark}m)", justify="center")
            table.add_column(f"XP ({secondary_minute_mark}m/{primary_minute_mark}m)", justify="center")
            table.add_column(f"XPM ({secondary_minute_mark}m/{primary_minute_mark}m)", justify="center")
            table.add_column(f"Kills (vs Lane @{kill_limit_display_minutes}m)", justify="center", header_style="bold green")
            table.add_column(f"Deaths (to Lane @{kill_limit_display_minutes}m)", justify="center", header_style="bold red")

            opposing_team_string_value = "dire" if team_name_string == "radiant" else "radiant"
            # Get opposing laners' hero names for kill/death analysis context
            opposing_laners_hero_names_list_val = [
                p.get('hero_name_parsed', '') for p in lanes_assignments_for_tables_display.get(lane_name_key,{}).get(opposing_team_string_value,[]) 
                if p.get('hero_name_parsed')
            ]

            for player_object_data_item in players_in_lane_for_team_display:
                # Get KPIs at both primary and secondary minute marks for display
                kpis_at_secondary_minute = _get_player_kpis_at_specific_minute(player_object_data_item, secondary_minute_mark, hero_id_to_name_map)
                kpis_at_primary_minute = _get_player_kpis_at_specific_minute(player_object_data_item, primary_minute_mark, hero_id_to_name_map)
                kills, deaths = _analyze_lane_kill_death_events_for_player(player_object_data_item, players_data_for_tables_display, opposing_laners_hero_names_list_val, KPI_PARAMETERS["kill_death_analysis_time_limit_seconds"])
                table.add_row(
                    player_object_data_item.get("display_name"), kpis_at_primary_minute["hero"], 
                    f"{kpis_at_secondary_minute['level']}/{kpis_at_primary_minute['level']}",
                    f"{kpis_at_secondary_minute['lh']}/{kpis_at_primary_minute['lh']}",
                    f"{kpis_at_secondary_minute['dn']}/{kpis_at_primary_minute['dn']}",
                    f"{kpis_at_secondary_minute['gold']}/{kpis_at_primary_minute['gold']}",
                    f"{kpis_at_secondary_minute['gpm']}/{kpis_at_primary_minute['gpm']}",
                    f"{kpis_at_secondary_minute['xp']}/{kpis_at_primary_minute['xp']}",
                    f"{kpis_at_secondary_minute['xpm']}/{kpis_at_primary_minute['xpm']}",
                    str(kills), str(deaths)
                )
            console.print(table)

        # Display overall lane summary (gold/xp diff, kills, towers, verdict)
        lane_summary_panel_content = Text()
        lane_summary_details = lane_information.get("summary_details", {})
        lane_summary_panel_content.append(f"Total Gold Adv @{primary_minute_mark}m (Radiant - Dire): ", style="bold")
        lane_summary_panel_content.append(f"{lane_summary_details.get('gold_diff', 0):+G}\n", style="gold1" if lane_summary_details.get('gold_diff',0)>0 else ("red1" if lane_summary_details.get('gold_diff',0)<0 else "white"))
        lane_summary_panel_content.append(f"Total XP Adv @{primary_minute_mark}m (Radiant - Dire): ", style="bold")
        lane_summary_panel_content.append(f"{lane_summary_details.get('xp_diff', 0):+G}\n", style="cyan1" if lane_summary_details.get('xp_diff',0)>0 else ("red1" if lane_summary_details.get('xp_diff',0)<0 else "white"))
        lane_summary_panel_content.append(f"Net Kills in Lane @{kill_limit_display_minutes}m (Radiant - Dire): ", style="bold")
        lane_summary_panel_content.append(f"{lane_summary_details.get('kill_diff', 0):+}\n", style="green1" if lane_summary_details.get('kill_diff',0)>0 else ("red1" if lane_summary_details.get('kill_diff',0)<0 else "white"))
        lane_summary_panel_content.append(f"Radiant T1 Fallen by {tower_limit_display_minutes}min: ", style="bold")
        lane_summary_panel_content.append(f"{lane_summary_details.get('radiant_t1_fallen', False)}\n", style="red" if lane_summary_details.get('radiant_t1_fallen') else "green")
        lane_summary_panel_content.append(f"Dire T1 Fallen by {tower_limit_display_minutes}min: ", style="bold")
        lane_summary_panel_content.append(f"{lane_summary_details.get('dire_t1_fallen', False)}\n", style="red" if lane_summary_details.get('dire_t1_fallen') else "green")
        
        verdict_text_for_console_display = lane_information.get('verdict_text', 'N/A')
        verdict_style = "white" # Default style
        if "Radiant Ahead" in verdict_text_for_console_display: verdict_style = "bold bright_cyan"
        elif "Dire Ahead" in verdict_text_for_console_display: verdict_style = "bold bright_red"
        
        lane_summary_panel_content.append(f"Verdict: ", style="bold")
        lane_summary_panel_content.append(Text.from_markup(f"[{verdict_style}]{verdict_text_for_console_display}[/{verdict_style}]"))

        console.print(Panel(lane_summary_panel_content, title=f"Overall {lane_name_key.upper()} Lane Summary (@{primary_minute_mark}min)", expand=False, border_style="green"))
        console.print("-" * console.width) # Visual separator between lanes

@app.command(name="export-match-ids", help="Exports all match IDs from the database to a CSV file.")
def export_match_ids(output_file_path_arg: str = typer.Option(..., "--output-file", "-o", help="Path to save the CSV file (e.g., all_match_ids.csv).", dir_okay=False, writable=True)):
    """Exports all match_ids from the 'matches' table to a CSV file."""
    console.print(Panel(f"[bold sky_blue1]Exporting All Match IDs to '{output_file_path_arg}'[/bold sky_blue1]", expand=False, border_style="sky_blue1"))
    db_connection = get_database_connection()
    try:
        match_id_rows_data = db_connection.execute("SELECT match_id FROM matches ORDER BY match_id ASC").fetchall()
    except sqlite3.Error as error:
        console.print(Panel(f"[bold red]SQLite error fetching match IDs:[/bold red]\n{error}", title="[bold red]DB Error[/bold red]", border_style="red"))
        db_connection.close()
        raise typer.Exit(code=1)
    finally:
        db_connection.close()

    if not match_id_rows_data:
        console.print(Padding("[yellow]No match IDs found in the database to export.[/yellow]",(1,2)))
        return

    try:
        output_file_absolute_path = os.path.abspath(output_file_path_arg)
        output_directory = os.path.dirname(output_file_absolute_path)
        if output_directory and not os.path.exists(output_directory): # Create dir if not exists
            os.makedirs(output_directory)
            console.print(f"[info]Created directory: {output_directory}[/info]")

        with open(output_file_absolute_path, 'w', newline='') as csv_file_writer:
            csv_writer_object = csv.writer(csv_file_writer)
            csv_writer_object.writerow(["match_id"]) # CSV Header
            for row_item_data in match_id_rows_data:
                csv_writer_object.writerow([row_item_data["match_id"]])
        console.print(f"[green]Successfully exported {len(match_id_rows_data)} match IDs to: {output_file_absolute_path}[/green]")
    except Exception as file_system_error: # Catch broad file operation errors
        console.print(Panel(f"[bold red]An unexpected error occurred while preparing to save to file '{output_file_path_arg}':[/bold red]\n{file_system_error}", title="[bold red]File Save Error[/bold red]", border_style="red"))

@app.command(name="batch-analyze-lanes", help="Batch analyzes lanes for matches in a CSV file, outputs results to another CSV (uses configured KPIs).")
def batch_analyze_lanes(
    input_csv_file_path: str = typer.Argument(..., help="Path to the input CSV file containing match IDs (must have a 'match_id' header).", exists=True, file_okay=True, dir_okay=False, readable=True),
    output_csv_file_path: str = typer.Option(..., "--output-file", "-o", help="Path to save the batch analysis results CSV (e.g., batch_lane_analysis.csv).", dir_okay=False, writable=True)
):
    """Performs lane analysis for a list of match IDs from a CSV and outputs results to another CSV."""
    console.print(Panel(f"[bold sky_blue1]Batch Lane Analysis from '{input_csv_file_path}' to '{output_csv_file_path}'[/bold sky_blue1]", expand=False, border_style="sky_blue1"))
    
    match_ids_to_process_list = []
    try: # Read match IDs from input CSV
        with open(input_csv_file_path, 'r', newline='') as csv_file_reader:
            csv_dict_reader = csv.DictReader(csv_file_reader)
            if "match_id" not in csv_dict_reader.fieldnames: # Ensure 'match_id' column exists
                console.print(Panel(f"[bold red]Error: Input CSV '{input_csv_file_path}' must contain a 'match_id' column header.[/bold red]", title="[bold red]CSV Error[/bold red]", border_style="red"))
                raise typer.Exit(code=1)
            for row_item_data in csv_dict_reader:
                try:
                    match_ids_to_process_list.append(int(row_item_data["match_id"]))
                except ValueError: # Skip non-integer match_ids
                    console.print(f"[yellow]Warning: Skipping invalid match_id '{row_item_data['match_id']}' in '{input_csv_file_path}'.[/yellow]")
    except Exception as csv_read_error:
        console.print(Panel(f"[bold red]An unexpected error occurred while reading '{input_csv_file_path}':[/bold red]\n{csv_read_error}", title="[bold red]CSV Read Error[/bold red]", border_style="red"))
        raise typer.Exit(code=1)

    if not match_ids_to_process_list:
        console.print(Padding(f"[yellow]No valid match IDs found in '{input_csv_file_path}' to process.[/yellow]", (1,2)))
        return

    db_connection = get_database_connection() # Single connection for the batch
    hero_id_to_name_map = _get_all_heroes_id_to_name_map_from_db(db_connection) 
    player_custom_names_map = _get_all_player_custom_names_map_from_db(db_connection)

    output_file_absolute_path = os.path.abspath(output_csv_file_path)
    output_directory = os.path.dirname(output_file_absolute_path)
    if output_directory and not os.path.exists(output_directory): # Create output dir if needed
        try:
            os.makedirs(output_directory)
            console.print(f"[info]Created directory: {output_directory}[/info]")
        except OSError as os_error:
            console.print(Panel(f"[bold red]Error creating output directory '{output_directory}':[/bold red]\n{os_error}", title="[bold red]Directory Error[/bold red]", border_style="red"))
            db_connection.close()
            raise typer.Exit(code=1)
            
    csv_field_names_list = [ # Define CSV output columns
        "match_id", "draft_order",
        "top_radiant_score", "top_dire_score", "top_verdict",
        "mid_radiant_score", "mid_dire_score", "mid_verdict",
        "bot_radiant_score", "bot_dire_score", "bot_verdict",
        "analysis_error_message" # To log any errors during analysis of a specific match
    ]
    processed_ok_count = 0
    error_count = 0

    # Use Rich Progress for visual feedback during batch processing
    with Progress(
        TextColumn("[progress.description]{task.description}", style="cyan"),
        BarColumn(bar_width=None, style="cyan", complete_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%", style="green"),
        TimeRemainingColumn(), TimeElapsedColumn(), console=console, transient=True # transient hides progress bar on completion
    ) as progress_bar:
        analysis_task = progress_bar.add_task("Analyzing matches...", total=len(match_ids_to_process_list))
        try:
            with open(output_file_absolute_path, 'w', newline='') as csv_output_file_writer:
                csv_dict_writer_object = csv.DictWriter(csv_output_file_writer, fieldnames=csv_field_names_list)
                csv_dict_writer_object.writeheader() # Write CSV header
                
                for match_id_loop_item in match_ids_to_process_list:
                    analysis_result_data = _perform_core_lane_analysis_for_match(match_id_loop_item, db_connection, hero_id_to_name_map, player_custom_names_map)
                    row_to_write_to_csv = {"match_id": match_id_loop_item, "analysis_error_message": None}
                    
                    if analysis_result_data and not analysis_result_data.get("error"):
                        row_to_write_to_csv["draft_order"] = analysis_result_data.get("draft_order", "N/A")
                        for lane_name in ["top", "mid", "bot"]:
                            lane_data_item = analysis_result_data.get("lanes", {}).get(lane_name, {})
                            row_to_write_to_csv[f"{lane_name}_radiant_score"] = lane_data_item.get("radiant_score", 0)
                            row_to_write_to_csv[f"{lane_name}_dire_score"] = lane_data_item.get("dire_score", 0)
                            row_to_write_to_csv[f"{lane_name}_verdict"] = lane_data_item.get("verdict_text", "N/A")
                        processed_ok_count +=1
                    else: 
                        row_to_write_to_csv["analysis_error_message"] = analysis_result_data.get("error", "Unknown analysis error") if analysis_result_data else "Core analysis returned None"
                        error_count +=1
                    csv_dict_writer_object.writerow(row_to_write_to_csv)
                    progress_bar.update(analysis_task, advance=1) # Update progress bar
        except Exception as batch_processing_error: # Catch broad errors during file writing or analysis loop
            console.print(Panel(f"[bold red]An unexpected error occurred during batch processing:[/bold red]\n{batch_processing_error}", title="[bold red]Batch Error[/bold red]", border_style="red"))
            db_connection.close()
            raise typer.Exit(code=1)
    db_connection.close() # Close DB connection after batch is done
    console.print(f"\n[green]Batch lane analysis complete.[/green]")
    console.print(f"Successfully processed and wrote {processed_ok_count} matches to '{output_file_absolute_path}'.")
    if error_count > 0:
        console.print(f"[yellow]Encountered errors for {error_count} matches (details in CSV).[/yellow]")

@app.command(name="manage-kpi-config", help=f"View or update lane analysis KPI parameters (stored in {CONFIG_FILE_NAME}).")
def manage_kpi_parameters_config(
    show_current_config: bool = typer.Option(True, "--show/--no-show", help="Show current KPI parameters."),
    reset_config_to_defaults: bool = typer.Option(False, "--reset-defaults", help="Reset all parameters to their default values.", is_flag=True),
    # Options to update individual KPI parameters
    analysis_minute_mark_config: Optional[int] = typer.Option(None, help="Primary minute for KPI calculations (e.g., GPM, Gold)."),
    kill_death_limit_seconds_config: Optional[int] = typer.Option(None, help="Time limit (seconds) for K/D analysis in lane."),
    tower_kill_limit_seconds_config: Optional[int] = typer.Option(None, help="Time limit (seconds) for 'early' tower fall."),
    display_secondary_minute_mark_config: Optional[int] = typer.Option(None, help="Secondary minute for table display (e.g., 8min/10min stats)."),
    major_gold_threshold_config: Optional[int] = typer.Option(None, help="Threshold for major gold lead per laner."),
    minor_gold_threshold_config: Optional[int] = typer.Option(None, help="Threshold for minor gold lead per laner."),
    major_xp_threshold_config: Optional[int] = typer.Option(None, help="Threshold for major XP lead per laner."),
    minor_xp_threshold_config: Optional[int] = typer.Option(None, help="Threshold for minor XP lead per laner."),
    points_for_major_lead_config: Optional[int] = typer.Option(None, help="Points for a major gold/XP lead."),
    points_for_minor_lead_config: Optional[int] = typer.Option(None, help="Points for a minor gold/XP lead."),
    kill_diff_major_points_threshold_config: Optional[int] = typer.Option(None, help="Net kill difference for major points."),
    points_for_major_kill_difference_config: Optional[int] = typer.Option(None, help="Points for major kill difference."),
    kill_diff_minor_points_threshold_config: Optional[int] = typer.Option(None, help="Net kill difference for minor points."),
    points_for_minor_kill_difference_config: Optional[int] = typer.Option(None, help="Points for minor kill difference."),
    points_for_tower_kill_config: Optional[int] = typer.Option(None, help="Points for an early tower kill.")
):
    """Allows viewing and updating of KPI parameters stored in the config file."""
    global KPI_PARAMETERS # Allow modification of the global KPI_PARAMETERS dict
    current_loaded_parameters = _load_kpi_parameters() # Load fresh from file
    parameters_were_changed = False

    if reset_config_to_defaults:
        current_loaded_parameters = DEFAULT_KPI_PARAMETERS.copy() # Reset to hardcoded defaults
        _save_kpi_parameters(current_loaded_parameters) # Save reset config
        KPI_PARAMETERS = current_loaded_parameters.copy() # Update global runtime config
        console.print(Panel(f"[green]KPI parameters have been reset to defaults and saved to {CONFIG_FILE_NAME}.[/green]", title="[green]Config Reset[/green]", border_style="green"))
        if not show_current_config: return # Exit if only resetting and not showing

    # Map CLI options to parameter keys for easier updating
    update_map = {
        "analysis_minute_mark": analysis_minute_mark_config,
        "kill_death_analysis_time_limit_seconds": kill_death_limit_seconds_config,
        "early_tower_kill_time_limit_seconds": tower_kill_limit_seconds_config,
        "display_secondary_minute_mark": display_secondary_minute_mark_config
    }
    for key, value in update_map.items():
        if value is not None: # If a CLI option was provided for this key
            current_loaded_parameters[key] = value
            parameters_were_changed = True
    
    # Map CLI options for score_weights sub-dictionary
    score_weights_update_map = {
        "major_gold_lead_per_laner_threshold": major_gold_threshold_config,
        "minor_gold_lead_per_laner_threshold": minor_gold_threshold_config,
        "major_xp_lead_per_laner_threshold": major_xp_threshold_config,
        "minor_xp_lead_per_laner_threshold": minor_xp_threshold_config,
        "points_for_major_lead": points_for_major_lead_config,
        "points_for_minor_lead": points_for_minor_lead_config,
        "kill_difference_for_major_points": kill_diff_major_points_threshold_config,
        "points_for_major_kill_difference": points_for_major_kill_difference_config,
        "kill_difference_for_minor_points": kill_diff_minor_points_threshold_config,
        "points_for_minor_kill_difference": points_for_minor_kill_difference_config,
        "points_for_early_tower_kill": points_for_tower_kill_config
    }
    # Ensure 'score_weights' exists and is a dict before updating
    if "score_weights" not in current_loaded_parameters or not isinstance(current_loaded_parameters["score_weights"], dict):
        current_loaded_parameters["score_weights"] = DEFAULT_KPI_PARAMETERS["score_weights"].copy()
        parameters_were_changed = True # Mark as changed if we had to reinitialize this part
    
    score_weights_dict = current_loaded_parameters["score_weights"]
    for key, value in score_weights_update_map.items():
        if value is not None: # If a CLI option was provided for this score_weight key
            score_weights_dict[key] = value
            parameters_were_changed = True

    if parameters_were_changed and not reset_config_to_defaults: # Save if changes were made (and not just reset)
        _save_kpi_parameters(current_loaded_parameters)
        KPI_PARAMETERS = current_loaded_parameters.copy() # Update global runtime config
        console.print(Panel(f"[green]KPI parameters updated and saved to {CONFIG_FILE_NAME}.[/green]", title="[green]Config Updated[/green]", border_style="green"))

    if show_current_config: # Display the current (possibly updated) configuration
        console.print(Panel(f"Current Lane Analysis KPI Parameters (from [underline blue]{CONFIG_FILE_NAME}[/underline blue])", expand=False, border_style="blue", title_align="left", padding=(1,2)))
        
        general_table = Table(title="[b]General Timing & Analysis Parameters[/b]", show_header=False, box=None, row_styles=["dim", ""])
        general_table.add_column("Parameter", style="cyan", no_wrap=True)
        general_table.add_column("Value", style="bright_white")
        for key in ["analysis_minute_mark", "kill_death_analysis_time_limit_seconds", "early_tower_kill_time_limit_seconds", "display_secondary_minute_mark"]:
            readable_key = key.replace("_", " ").capitalize()
            general_table.add_row(readable_key, str(current_loaded_parameters.get(key)))
        console.print(Padding(general_table, (0,2)))

        weights_table = Table(title="[b]Score Weights[/b]", show_header=False, box=None, row_styles=["dim", ""])
        weights_table.add_column("Parameter", style="cyan", no_wrap=True)
        weights_table.add_column("Value", style="bright_white")
        if "score_weights" in current_loaded_parameters and isinstance(current_loaded_parameters["score_weights"], dict):
            for key, value in current_loaded_parameters["score_weights"].items():
                readable_key = key.replace("_", " ").capitalize()
                weights_table.add_row(readable_key, str(value))
        else: # Should not happen if _load_kpi_parameters works correctly
            weights_table.add_row("Score weights configuration is missing or invalid.", "")
        console.print(Padding(weights_table, (0,2)))
        console.print(f"\n[dim]These parameters are loaded from '{CONFIG_FILE_NAME}'. If the file is missing or invalid, defaults are used and a new file is created.[/dim]")

if __name__ == "__main__":
    # Entry point for the CLI application
    console.print(Panel(Text.from_markup("[bold green]:computer: OpenDota DB Viewer & Analyzer CLI :chart_increasing:[/bold green]"), expand=False, border_style="green"))
    app()

