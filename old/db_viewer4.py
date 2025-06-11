# merged_db_viewer_compact.py
# CLI tool to view and analyze OpenDota SQLite DB. Local DB interaction only.
# Features: custom player names, raw match JSON view/viz/output,
# player/hero summaries, DB stats, lane analysis (with configurable KPIs),
# list heroes/matches, find player-hero matches, export match IDs, batch lane analysis.

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

# Optional Visualization
try:
    import gravis as gv
    import networkx as nx
    import webbrowser
    VIS_LIBS_AVAILABLE = True
except ImportError:
    VIS_LIBS_AVAILABLE = False
    gv = nx = webbrowser = None

app = typer.Typer(help="CLI for OpenDota SQLite DB. Local DB only.")
console = Console()

DB_NAME = "opendota_league_info.db"
CONFIG_FILE_NAME = "lane_analysis_config.json"

# --- KPI Configuration Default Values ---
DEFAULT_KPI_PARAMS = {
    "analysis_minute_mark": 10,  # Primary minute for KPI calculations (e.g., GPM, XPM, Gold, XP)
    "kill_death_analysis_time_limit_seconds": 600, # Time limit for considering kills/deaths in lane (e.g., 10 * 60)
    "early_tower_kill_time_limit_seconds": 720, # Time limit for "early" tower fall (e.g., 12 * 60)
    "score_weights": {
        "major_gold_lead_per_laner_threshold": 750,
        "minor_gold_lead_per_laner_threshold": 300,
        "major_xp_lead_per_laner_threshold": 1000,
        "minor_xp_lead_per_laner_threshold": 500,
        "points_for_major_lead": 2,
        "points_for_minor_lead": 1,
        "kill_difference_for_major_points": 2, # Net kill diff (e.g., 2 more kills)
        "points_for_major_kill_difference": 2,
        "kill_difference_for_minor_points": 1, # Net kill diff (e.g., 1 more kill)
        "points_for_minor_kill_difference": 1,
        "points_for_early_tower_kill": 3
    },
    "display_secondary_minute_mark": 8 # Secondary minute mark for display in tables (e.g. 8min/10min stats)
}

# --- KPI Configuration Helper Functions ---
def _save_kpi_params(params: Dict[str, Any]):
    """Saves KPI parameters to the config file."""
    try:
        with open(CONFIG_FILE_NAME, 'w') as f:
            json.dump(params, f, indent=4)
        # console.print(f"[dim]KPI parameters saved to {CONFIG_FILE_NAME}[/dim]")
    except IOError as e:
        console.print(f"[red]Error saving KPI config to {CONFIG_FILE_NAME}: {e}[/red]")

def _load_kpi_params() -> Dict[str, Any]:
    """Loads KPI parameters from config file, or uses defaults if not found/invalid."""
    if os.path.exists(CONFIG_FILE_NAME):
        try:
            with open(CONFIG_FILE_NAME, 'r') as f:
                params = json.load(f)
            # Basic validation: ensure top-level keys and 'score_weights' exist
            if "score_weights" not in params:
                console.print(f"[yellow]Warning: 'score_weights' missing in {CONFIG_FILE_NAME}. Applying defaults for it.[/yellow]")
                params["score_weights"] = DEFAULT_KPI_PARAMS["score_weights"]
            
            # Check and fill missing keys with defaults to ensure robustness
            current_params = DEFAULT_KPI_PARAMS.copy()
            current_params.update(params) # User params override defaults
            # Ensure nested score_weights are also updated correctly
            if isinstance(params.get("score_weights"), dict):
                 current_params["score_weights"] = DEFAULT_KPI_PARAMS["score_weights"].copy()
                 current_params["score_weights"].update(params["score_weights"])
            else: # if score_weights was invalid or not a dict
                 current_params["score_weights"] = DEFAULT_KPI_PARAMS["score_weights"].copy()


            if params != current_params: # If we had to merge defaults
                # console.print(f"[dim]Merging loaded KPI params with defaults due to missing/updated keys.[/dim]")
                _save_kpi_params(current_params) # Save the merged version
            return current_params
        except json.JSONDecodeError:
            console.print(f"[red]Error decoding JSON from {CONFIG_FILE_NAME}. Using default KPI parameters and overwriting.[/red]")
            _save_kpi_params(DEFAULT_KPI_PARAMS)
            return DEFAULT_KPI_PARAMS.copy()
        except IOError as e:
            console.print(f"[red]Error reading {CONFIG_FILE_NAME}: {e}. Using default KPI parameters.[/red]")
            return DEFAULT_KPI_PARAMS.copy()
    else:
        # console.print(f"[info]{CONFIG_FILE_NAME} not found. Creating with default KPI parameters.[/info]")
        _save_kpi_params(DEFAULT_KPI_PARAMS)
        return DEFAULT_KPI_PARAMS.copy()

# Initialize KPI parameters globally
KPI_PARAMS = _load_kpi_params()


# --- Database and General Helper Functions ---
def _init_custom_player_names_table(conn: sqlite3.Connection):
    """Ensure 'player_custom_names' table exists."""
    cursor = conn.cursor()
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_custom_names (
            account_id INTEGER PRIMARY KEY,
            custom_name TEXT NOT NULL UNIQUE,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.commit()
    except sqlite3.Error as e:
        console.print(f"[red]SQLite err creating 'player_custom_names': {e}[/red]")

def get_db_connection() -> sqlite3.Connection:
    """Establishes DB connection."""
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        _init_custom_player_names_table(conn)
        return conn
    except sqlite3.Error as e:
        console.print(f"[red]SQLite conn err: {e}[/red]")
        raise typer.Exit(code=1)

def get_hero_id_from_name_db(hero_name_in: str, conn: sqlite3.Connection) -> Optional[int]:
    """Queries 'heroes' table for hero_id from hero name (case-insensitive)."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT hero_id, name FROM heroes WHERE LOWER(name) = LOWER(?)", (hero_name_in,))
        row = cursor.fetchone()
        if row: return int(row["hero_id"])

        proc_hero_name = hero_name_in.replace("npc_dota_hero_", "")
        cursor.execute("SELECT hero_id, name FROM heroes WHERE name LIKE ? OR name LIKE ?",
                       (f'%{proc_hero_name}%', f'%npc_dota_hero_{proc_hero_name}%'))
        matches = cursor.fetchall()
        
        if len(matches) == 1:
            console.print(f"[dim]Found unique hero: '{matches[0]['name']}' (ID: {matches[0]['hero_id']})[/dim]")
            return int(matches[0]["hero_id"])
        elif len(matches) > 1:
            console.print(f"[yellow]Multiple heroes for '{hero_name_in}'. Be specific. Matches:[/yellow]")
            for m in matches: console.print(f"  - {m['name']} (ID: {m['hero_id']})")
            return None
    except sqlite3.Error as e:
        console.print(f"[red]SQLite err searching hero '{hero_name_in}': {e}[/red]")
    return None

def _get_all_heroes_map_from_db(conn: sqlite3.Connection) -> Dict[int, str]:
    """Loads all hero IDs and internal names from 'heroes' table."""
    cursor = conn.cursor()
    h_map: Dict[int, str] = {}
    try:
        cursor.execute("SELECT hero_id, name FROM heroes")
        for row in cursor.fetchall(): h_map[row["hero_id"]] = row["name"]
    except sqlite3.Error as e:
        console.print(f"[red]SQLite err loading heroes: {e}[/red]. Names may be incorrect.")
        return {}
    if not h_map: console.print("[yellow]Warn: Hero map empty. Names may be incorrect.[/yellow]")
    return h_map

def _get_all_player_custom_names_map_from_db(conn: sqlite3.Connection) -> Dict[int, str]:
    """Loads all custom player names."""
    cursor = conn.cursor()
    names_map: Dict[int, str] = {}
    try:
        cursor.execute("SELECT account_id, custom_name FROM player_custom_names")
        for row in cursor.fetchall(): names_map[row["account_id"]] = row["custom_name"]
    except sqlite3.Error as e:
        console.print(f"[red]SQLite err loading custom names: {e}[/red]")
    return names_map

def _visualize_json_structure_gravis(json_data: Dict[str, Any], item_id: Any, item_type: str = "Match") -> bool:
    """Generates interactive HTML graph of JSON structure."""
    if not VIS_LIBS_AVAILABLE:
        console.print("[yellow]Viz libs not installed. Skipping. (pip install gravis networkx)[/yellow]")
        return False
    if not json_data or not isinstance(json_data, (dict, list)):
        console.print("[yellow]Invalid/empty JSON for viz.[/yellow]")
        return False

    console.print(f"[info]Generating graph for {item_type} ID {item_id}...[/info]")
    if gv: console.print(f"[info]Gravis ver: {gv.__version__}[/info]")

    graph = nx.DiGraph()
    def add_nodes(data_node: Any, parent_name: Optional[str] = None):
        shape = 'ellipse'
        if isinstance(data_node, dict):
            for k, v in data_node.items():
                curr_name = f"{parent_name}.{k}" if parent_name else str(k)
                val_snip = str(v)[:147] + "..." if len(str(v)) > 150 else str(v)
                graph.add_node(curr_name, title=val_snip, label=str(k), color='lightblue', size=10, shape=shape)
                if parent_name: graph.add_edge(parent_name, curr_name)
                if isinstance(v, (dict, list)): add_nodes(v, curr_name)
        elif isinstance(data_node, list):
            if not data_node and parent_name:
                empty_node = f"{parent_name}.[]"
                graph.add_node(empty_node, title="Empty List", label="[]", color='lightgrey', size=8, shape='rectangle')
                if parent_name: graph.add_edge(parent_name, empty_node)
                return
            for i, item in enumerate(data_node):
                curr_name = f"{parent_name}[{i}]"
                val_snip = str(item)[:147] + "..." if len(str(item)) > 150 else str(item)
                graph.add_node(curr_name, title=val_snip, label=f"[{i}]", color='lightgreen', size=10, shape='box')
                if parent_name: graph.add_edge(parent_name, curr_name)
                if isinstance(item, (dict, list)): add_nodes(item, curr_name)

    root_label = f"{item_type} ID: {item_id}"
    graph.add_node(root_label, label=root_label, color='salmon', size=15, shape='diamond')
    add_nodes(json_data, root_label)

    if not graph.nodes() or (len(graph.nodes()) == 1 and root_label in graph.nodes()):
         console.print("[yellow]Warn: Graph empty/root-only. Viz might not be useful.[/yellow]")
         graph.nodes[root_label]['title'] = json.dumps(json_data, indent=2)

    out_file_abs = os.path.abspath(f"{item_type.lower()}_{item_id}_visualization.html")
    try:
        fig = gv.d3(graph, graph_height=800, node_label_data_source='label', show_menu=True, zoom_factor=0.7, details_min_height=150, details_max_height=300, use_edge_size_normalization=True, edge_size_data_source='weight', use_node_size_normalization=True, node_size_data_source='size')
        fig.export_html(out_file_abs, overwrite=True)
        console.print(f"[green]Viz saved: {out_file_abs}[/green]")
        if webbrowser:
            try: webbrowser.open(f"file://{out_file_abs}")
            except webbrowser.Error as wb_err: console.print(f"[yellow]Browser open err: {wb_err}. Open manually.[/yellow]")
        return True
    except TypeError as te:
        console.print(f"[red]Gravis TypeError: {te}[/red]. Trying basic call...")
        try:
            fig_simple = gv.d3(graph, graph_height=800, node_label_data_source='label')
            fig_simple.export_html(out_file_abs, overwrite=True)
            console.print(f"[green]Basic viz saved: {out_file_abs}[/green]")
            if webbrowser:
                try: webbrowser.open(f"file://{out_file_abs}")
                except webbrowser.Error as wb_err: console.print(f"[yellow]Browser open err (basic): {wb_err}.[/yellow]")
            return True
        except Exception as e_simple: console.print(f"[red]Basic Gravis err: {e_simple}[/red]"); return False
    except Exception as e_gv: console.print(f"[red]Gravis viz err: {e_gv}[/red]"); return False

def _parse_hero_name_from_log_key(npc_key: Optional[str]) -> str:
    """Extracts hero name from 'npc_dota_hero_bloodseeker' -> 'bloodseeker'."""
    if not npc_key: return "Unknown"
    return npc_key.replace("npc_dota_hero_", "") if npc_key.startswith("npc_dota_hero_") else npc_key

def _get_hero_display_name_from_id(h_id: Optional[int], h_map: Dict[int, str]) -> str:
    """Gets displayable hero name from ID via hero_map."""
    if h_id is None: return "N/A"
    internal_name = h_map.get(h_id)
    return _parse_hero_name_from_log_key(internal_name) if internal_name else f"ID:{h_id}"

def _identify_laners(players: List[Dict[str, Any]], h_map: Dict[int, str]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Identifies players per lane, enriches with parsed hero name and team string."""
    lanes_assign: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "top": {"radiant": [], "dire": []}, "mid": {"radiant": [], "dire": []},
        "bot": {"radiant": [], "dire": []}, "unknown": {"radiant": [], "dire": []}
    }
    for p in players:
        is_rad = p.get("isRadiant")
        if is_rad is None: is_rad = p.get("player_slot", 100) < 100
        team = "radiant" if is_rad else "dire"
        lane_id = p.get("lane")
        lane_name = "unknown"
        if lane_id is not None:
            if team == "radiant":
                if lane_id == 1: lane_name = "bot"
                elif lane_id == 2: lane_name = "mid"
                elif lane_id == 3: lane_name = "top"
            else: # Dire
                if lane_id == 1: lane_name = "top"
                elif lane_id == 2: lane_name = "mid"
                elif lane_id == 3: lane_name = "bot"
        p['hero_name_parsed'] = _get_hero_display_name_from_id(p.get('hero_id'), h_map)
        p['team_str'] = team
        lanes_assign[lane_name][team].append(p)
    return lanes_assign

def _get_player_kpis_at_minute(p_data: Dict[str, Any], minute: int, h_map: Dict[int, str]) -> Dict[str, Any]:
    """Extracts player KPIs at a specific minute."""
    kpis = {"lh": 0, "dn": 0, "gold": 0, "xp": 0, "gpm": 0, "xpm": 0, "level": 1, "hero": "N/A", "name": "N/A"}
    kpis["hero"] = _get_hero_display_name_from_id(p_data.get('hero_id'), h_map)
    kpis["name"] = p_data.get("display_name", p_data.get("personaname", f"Slot {p_data.get('player_slot', '?')}"))
    
    def get_t_val(arr_name: str, default: int = 0) -> int:
        arr = p_data.get(arr_name)
        if arr and isinstance(arr, list):
            return arr[minute] if minute < len(arr) else (arr[-1] if arr else default)
        return default

    kpis["lh"], kpis["dn"], kpis["gold"], kpis["xp"] = get_t_val("lh_t"), get_t_val("dn_t"), get_t_val("gold_t"), get_t_val("xp_t")
    if minute > 0:
        kpis["gpm"] = round(kpis["gold"] / minute) if kpis["gold"] else 0
        kpis["xpm"] = round(kpis["xp"] / minute) if kpis["xp"] else 0
    
    lvl_t = p_data.get("level_t")
    if lvl_t and isinstance(lvl_t, list) and minute < len(lvl_t): kpis["level"] = lvl_t[minute]
    else:
        xp_thresh = [0, 240, 600, 1080, 1680, 2400, 3240, 4200, 5280, 6480, 7800, 9000, 10200, 11400, 12600] # Standard Dota 2 XP thresholds up to level 15
        # For levels > 15, XP requirement increases by 600 per level. Max level is 30.
        xp_thresh.extend([xp_thresh[-1] + 600 * i for i in range(1, 16)]) # Levels 16-30

        lvl = 1
        current_xp = kpis["xp"]
        for i, t in enumerate(xp_thresh): # Max level 30
            if current_xp >= t: lvl = i + 1 
            else: break
        kpis["level"] = min(lvl, 30)
    return kpis

def _analyze_lane_kill_death_events(laner: Dict[str, Any], all_players: List[Dict[str, Any]], opp_hero_names: List[str], time_limit: int) -> Tuple[int, int]:
    """Analyzes kills by/deaths of laner against opposing laners within time_limit."""
    kills_on_opp, deaths_to_opp = 0, 0
    laner_h_name = laner.get('hero_name_parsed', 'Unknown')
    if laner_h_name == 'Unknown' or not laner_h_name: return 0,0

    if laner.get("kills_log"):
        for k_event in laner["kills_log"]:
            if k_event.get("time", float('inf')) <= time_limit:
                victim_h_name = _parse_hero_name_from_log_key(k_event.get("key"))
                if victim_h_name in opp_hero_names: kills_on_opp += 1

    for killer_obj in all_players:
        if killer_obj.get("player_slot") == laner.get("player_slot") or killer_obj.get("isRadiant") == laner.get("isRadiant"): continue
        killer_h_name = killer_obj.get('hero_name_parsed', 'Unknown')
        if killer_h_name in opp_hero_names and killer_obj.get("kills_log"):
            for k_event in killer_obj["kills_log"]:
                if k_event.get("time", float('inf')) <= time_limit:
                    victim_h_name = _parse_hero_name_from_log_key(k_event.get("key"))
                    if victim_h_name == laner_h_name: deaths_to_opp += 1
    return kills_on_opp, deaths_to_opp

def _check_early_tower_status(objectives: Optional[List[Dict[str, Any]]], time_limit: int) -> Dict[str, Dict[str, bool]]:
    """Checks T1 tower status by time_limit."""
    status = {"top": {"radiant_t1_fallen": False, "dire_t1_fallen": False}, "mid": {"radiant_t1_fallen": False, "dire_t1_fallen": False}, "bot": {"radiant_t1_fallen": False, "dire_t1_fallen": False}}
    t1_keys = { "npc_dota_goodguys_tower1_top": ("top", "radiant_t1_fallen"), "npc_dota_badguys_tower1_top": ("top", "dire_t1_fallen"), "npc_dota_goodguys_tower1_mid": ("mid", "radiant_t1_fallen"), "npc_dota_badguys_tower1_mid": ("mid", "dire_t1_fallen"), "npc_dota_goodguys_tower1_bot": ("bot", "radiant_t1_fallen"), "npc_dota_badguys_tower1_bot": ("bot", "dire_t1_fallen")}
    if objectives:
        for obj in objectives:
            if obj.get("type") == "building_kill" and obj.get("time", float('inf')) <= time_limit:
                b_key = obj.get("key")
                if b_key in t1_keys:
                    lane, stat_key = t1_keys[b_key]
                    status[lane][stat_key] = True
    return status

def _extract_draft_order_str(match_data: Dict[str, Any], h_map: Dict[int, str]) -> str:
    """Extracts and formats pick/ban draft order."""
    picks_bans = match_data.get("picks_bans")
    if not picks_bans or not isinstance(picks_bans, list): return "N/A"
    entries = []
    for entry in picks_bans:
        hero_name = _get_hero_display_name_from_id(entry.get("hero_id"), h_map)
        action = "Pick" if entry.get("is_pick") else "Ban"
        team_str = "Radiant" if entry.get("team") == 0 else "Dire"
        entries.append(f"{team_str} {action}: {hero_name}")
    return "; ".join(entries) if entries else "N/A"

def _perform_core_lane_analysis(m_id: int, conn: sqlite3.Connection, h_map: Dict[int, str], custom_names: Dict[int, str]) -> Optional[Dict[str, Any]]:
    """Core laning phase analysis for a match. Returns structured data, using global KPI_PARAMS."""
    cursor = conn.cursor()
    try: cursor.execute("SELECT data FROM matches WHERE match_id = ?", (m_id,)); row = cursor.fetchone()
    except sqlite3.Error as e: return {"error": f"SQLite err for match {m_id}: {e}"}
    if not row or not row["data"]: return {"error": f"No data for match {m_id}."}
    try: m_data = json.loads(row["data"])
    except json.JSONDecodeError: return {"error": f"Invalid JSON for match {m_id}."}
    
    players = m_data.get("players", [])
    if not players: return {"error": f"No player data in JSON for match {m_id}."}

    for p_idx, p_info in enumerate(players):
        acc_id = p_info.get("account_id")
        d_name = p_info.get("personaname", f"Slot {p_info.get('player_slot', p_idx)}")
        if acc_id and acc_id in custom_names: d_name = custom_names[acc_id]
        players[p_idx]["display_name"] = d_name
        if "isRadiant" not in p_info and "player_slot" in p_info: players[p_idx]["isRadiant"] = p_info["player_slot"] < 100
    
    draft_str = _extract_draft_order_str(m_data, h_map)
    lanes_assign = _identify_laners(players, h_map)
    objectives = m_data.get("objectives")
    # Use KPI_PARAMS for time limits
    tower_stat = _check_early_tower_status(objectives, KPI_PARAMS["early_tower_kill_time_limit_seconds"])
    current_kpi_minute = KPI_PARAMS["analysis_minute_mark"]
    current_kill_time_limit = KPI_PARAMS["kill_death_analysis_time_limit_seconds"]
    weights = KPI_PARAMS["score_weights"]


    results = {"match_id": m_id, "draft_order": draft_str, "lanes": {}, "error": None}

    for lane_key in ["top", "mid", "bot"]:
        teams_in_lane = lanes_assign.get(lane_key)
        if not teams_in_lane or (not teams_in_lane["radiant"] and not teams_in_lane["dire"]):
            results["lanes"][lane_key] = {"radiant_score": 0, "dire_score": 0, "verdict_text": "No players", "summary": {}}
            continue

        lane_sum = {f"rad_gold_{current_kpi_minute}m": 0, f"dire_gold_{current_kpi_minute}m": 0, 
                    f"rad_xp_{current_kpi_minute}m": 0, f"dire_xp_{current_kpi_minute}m": 0, 
                    "rad_kills": 0, "dire_kills": 0}
        
        for team_str, team_players in teams_in_lane.items():
            if not team_players: continue
            opp_team_str = "dire" if team_str == "radiant" else "radiant"
            opp_laners = [p.get('hero_name_parsed', '') for p in lanes_assign[lane_key].get(opp_team_str, []) if p.get('hero_name_parsed')]
            for p_obj in team_players:
                kpis_at_mark = _get_player_kpis_at_minute(p_obj, current_kpi_minute, h_map) # Use configured minute
                kills, _ = _analyze_lane_kill_death_events(p_obj, players, opp_laners, current_kill_time_limit) # Use configured time
                
                prefix = "rad" if team_str == "radiant" else "dire"
                lane_sum[f"{prefix}_gold_{current_kpi_minute}m"] += kpis_at_mark["gold"]
                lane_sum[f"{prefix}_xp_{current_kpi_minute}m"] += kpis_at_mark["xp"]
                lane_sum[f"{prefix}_kills"] += kills
        
        num_rad_laners = len(teams_in_lane.get("radiant",[])); num_dire_laners = len(teams_in_lane.get("dire",[]))
        
        # Use thresholds from KPI_PARAMS.score_weights
        gold_thresh_major = weights["major_gold_lead_per_laner_threshold"]
        xp_thresh_major = weights["major_xp_lead_per_laner_threshold"]
        gold_thresh_minor = weights["minor_gold_lead_per_laner_threshold"]
        xp_thresh_minor = weights["minor_xp_lead_per_laner_threshold"]

        gold_diff = lane_sum[f"rad_gold_{current_kpi_minute}m"] - lane_sum[f"dire_gold_{current_kpi_minute}m"]
        xp_diff = lane_sum[f"rad_xp_{current_kpi_minute}m"] - lane_sum[f"dire_xp_{current_kpi_minute}m"]
        kill_diff = lane_sum["rad_kills"] - lane_sum["dire_kills"]
        rad_score, dire_score = 0, 0

        if num_rad_laners > 0 and gold_diff > gold_thresh_major * num_rad_laners : rad_score += weights["points_for_major_lead"]
        elif num_dire_laners > 0 and gold_diff < -gold_thresh_major * num_dire_laners : dire_score += weights["points_for_major_lead"]
        elif num_rad_laners > 0 and gold_diff > gold_thresh_minor * num_rad_laners : rad_score += weights["points_for_minor_lead"]
        elif num_dire_laners > 0 and gold_diff < -gold_thresh_minor * num_dire_laners : dire_score += weights["points_for_minor_lead"]

        if num_rad_laners > 0 and xp_diff > xp_thresh_major * num_rad_laners : rad_score += weights["points_for_major_lead"]
        elif num_dire_laners > 0 and xp_diff < -xp_thresh_major * num_dire_laners : dire_score += weights["points_for_major_lead"]
        elif num_rad_laners > 0 and xp_diff > xp_thresh_minor * num_rad_laners : rad_score += weights["points_for_minor_lead"]
        elif num_dire_laners > 0 and xp_diff < -xp_thresh_minor * num_dire_laners : dire_score += weights["points_for_minor_lead"]

        if kill_diff >= weights["kill_difference_for_major_points"] : rad_score += weights["points_for_major_kill_difference"]
        elif kill_diff <= -weights["kill_difference_for_major_points"] : dire_score += weights["points_for_major_kill_difference"]
        elif kill_diff >= weights["kill_difference_for_minor_points"] : rad_score += weights["points_for_minor_kill_difference"] # Note: using >= for minor as well
        elif kill_diff <= -weights["kill_difference_for_minor_points"] : dire_score += weights["points_for_minor_kill_difference"]
        
        lane_tower_stat = tower_stat.get(lane_key, {"radiant_t1_fallen": False, "dire_t1_fallen": False})
        if lane_tower_stat["dire_t1_fallen"]: rad_score += weights["points_for_early_tower_kill"]
        if lane_tower_stat["radiant_t1_fallen"]: dire_score += weights["points_for_early_tower_kill"]

        verdict = ""
        if rad_score > dire_score + 1: verdict = f"Radiant Ahead ({rad_score}v{dire_score})"
        elif dire_score > rad_score + 1: verdict = f"Dire Ahead ({dire_score}v{rad_score})"
        else: verdict = f"Even Lane ({rad_score}v{dire_score})"
        
        results["lanes"][lane_key] = {
            "radiant_score": rad_score, "dire_score": dire_score, "verdict_text": verdict,
            "summary_details": {"gold_diff": gold_diff, "xp_diff": xp_diff, "kill_diff": kill_diff, 
                                "radiant_t1_fallen": lane_tower_stat["radiant_t1_fallen"], 
                                "dire_t1_fallen": lane_tower_stat["dire_t1_fallen"]}
        }
    return results

@app.command(name="set-player-name", help="Set custom name for Account ID.")
def set_player_name(account_id: int = typer.Argument(..., help="Player's Account ID."), name: str = typer.Argument(..., help="Custom name.")):
    console.print(Panel(f"Set Name: ID {account_id}", expand=False))
    conn = get_db_connection()
    try:
        conn.execute("INSERT OR REPLACE INTO player_custom_names (account_id, custom_name, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)", (account_id, name))
        conn.commit()
        console.print(f"[green]Set name for {account_id} to '{name}' ok.[/green]")
    except sqlite3.IntegrityError: console.print(f"[red]Err: Name '{name}' may exist or integrity fail.[/red]")
    except sqlite3.Error as e: console.print(f"[red]SQLite err setting name for {account_id}: {e}[/red]")
    finally: conn.close()

@app.command(name="list-custom-names", help="List all custom player names.")
def list_custom_names():
    console.print(Panel("List Custom Names", expand=False))
    conn = get_db_connection()
    try: rows = conn.execute("SELECT account_id, custom_name, updated_at FROM player_custom_names ORDER BY custom_name ASC").fetchall()
    except sqlite3.Error as e: console.print(f"[red]SQLite err: {e}[/red]"); conn.close(); raise typer.Exit(code=1)
    finally: conn.close()
    if not rows: console.print("[yellow]No custom names found.[/yellow]"); return
    table = Table(title="Custom Player Names", show_header=True, header_style="magenta")
    table.add_column("Acc ID", width=15, justify="center"); table.add_column("Custom Name", min_width=20); table.add_column("Updated (UTC)", min_width=20, justify="center")
    for r in rows: table.add_row(str(r["account_id"]), r["custom_name"], r["updated_at"])
    console.print(table); console.print(f"\nTotal: {len(rows)}")

@app.command(name="delete-player-name", help="Delete custom name by Account ID or Custom Name.")
def delete_player_name(identifier: str = typer.Argument(..., help="Account ID or Custom Name.")):
    console.print(Panel(f"Delete Name: '{identifier}'", expand=False))
    conn = get_db_connection()
    acc_id_del: Optional[int] = None
    try: acc_id_del = int(identifier)
    except ValueError:
        try:
            row = conn.execute("SELECT account_id FROM player_custom_names WHERE LOWER(custom_name) = LOWER(?)", (identifier,)).fetchone()
            if row: acc_id_del = row["account_id"]
            else: console.print(f"[yellow]Name '{identifier}' not found.[/yellow]"); conn.close(); return
        except sqlite3.Error as e: console.print(f"[red]SQLite err finding name '{identifier}': {e}[/red]"); conn.close(); return
    if acc_id_del is None: console.print(f"[yellow]Could not resolve '{identifier}'.[/yellow]"); conn.close(); return
    try:
        cursor = conn.execute("DELETE FROM player_custom_names WHERE account_id = ?", (acc_id_del,))
        conn.commit()
        if cursor.rowcount > 0: console.print(f"[green]Deleted name for Acc ID {acc_id_del}.[/green]")
        else: console.print(f"[yellow]No name for Acc ID {acc_id_del} to delete.[/yellow]")
    except sqlite3.Error as e: console.print(f"[red]SQLite err deleting name for {acc_id_del}: {e}[/red]")
    finally: conn.close()

@app.command(name="list-heroes", help="List all heroes from DB.")
def list_heroes():
    console.print(Panel("List Heroes", expand=False))
    conn = get_db_connection()
    try: rows = conn.execute("SELECT hero_id, name, fetched_at FROM heroes ORDER BY hero_id ASC").fetchall()
    except sqlite3.Error as e: console.print(f"[red]SQLite err: {e}[/red]. Ensure 'heroes' table ok."); conn.close(); raise typer.Exit(code=1)
    finally: conn.close()
    if not rows: console.print("[yellow]No heroes in DB. (Populate via main script?)[/yellow]"); return
    table = Table(title="Stored Heroes", show_header=True, header_style="magenta")
    table.add_column("ID", width=10, justify="center"); table.add_column("Internal Name", min_width=30); table.add_column("Fetched (UTC)", min_width=20, justify="center")
    for h in rows: table.add_row(str(h["hero_id"]), h["name"], h["fetched_at"] or "N/A")
    console.print(table); console.print(f"\nTotal heroes: {len(rows)}")

@app.command(name="list-matches", help="List matches. Opts: -l limit, -t team filter.")
def list_matches(limit: Optional[int] = typer.Option(None, "-l", help="Num matches (most recent)."), search_team: Optional[str] = typer.Option(None, "-t", help="Filter by team name (case-insensitive).")):
    console.print(Panel("List Matches", expand=False))
    conn = get_db_connection()
    h_map = _get_all_heroes_map_from_db(conn); custom_names = _get_all_player_custom_names_map_from_db(conn)
    query = "SELECT match_id, data, fetched_at FROM matches "
    params_list: list = []
    if search_team: query += "WHERE (data LIKE ? OR data LIKE ? OR data LIKE ? OR data LIKE ?)"; params_list.extend([f'%"{search_team}"%'] * 4)
    query += " ORDER BY fetched_at DESC"
    if limit and limit > 0: query += " LIMIT ?"; params_list.append(limit)
    
    try: match_rows = conn.execute(query, tuple(params_list)).fetchall()
    except sqlite3.Error as e: console.print(f"[red]SQLite err: {e}[/red]"); conn.close(); raise typer.Exit(code=1)

    if not match_rows:
        console.print(f"[yellow]No matches found{' for team ' + search_team if search_team else ''}.[/yellow]"); conn.close(); return

    table = Table(title="Stored Matches", show_header=True, header_style="magenta")
    cols = ["MatchID", "Rad Team", "Dire Team", "Score", "Winner", "Rad Players (Name/ID, Hero)", "Dire Players (Name/ID, Hero)", "Fetched (UTC)"]
    widths = [12, 20, 20, 12, 20, 45, 45, 20]
    styles = ["dim", "cyan", "orange3", "default", "default", "cyan", "orange3", "dim"]
    justifies = ["center", "default", "default", "center", "center", "default", "default", "center"]
    for i, c in enumerate(cols): table.add_column(c, min_width=widths[i], style=styles[i], justify=justifies[i], overflow="fold" if "Players" in c else "default")

    displayed_count = 0
    for row in match_rows:
        m_id, fetched = row["match_id"], row["fetched_at"]
        try: m_data: Dict[str, Any] = json.loads(row["data"])
        except json.JSONDecodeError: console.print(f"[yellow]Warn: Skip JSON parse err for {m_id}.[/yellow]"); continue

        r_team_d, d_team_d = m_data.get('radiant_team', {}), m_data.get('dire_team', {})
        r_name = m_data.get('radiant_name') or (r_team_d.get('name') if isinstance(r_team_d, dict) else None) or (r_team_d.get('tag') if isinstance(r_team_d, dict) else None) or "Radiant"
        d_name = m_data.get('dire_name') or (d_team_d.get('name') if isinstance(d_team_d, dict) else None) or (d_team_d.get('tag') if isinstance(d_team_d, dict) else None) or "Dire"

        if search_team:
            s_lower = search_team.lower()
            if not (s_lower in r_name.lower() or s_lower in d_name.lower() or \
                    (isinstance(r_team_d, dict) and s_lower in str(r_team_d.get('tag','')).lower()) or \
                    (isinstance(d_team_d, dict) and s_lower in str(d_team_d.get('tag','')).lower())): continue
        
        r_score, d_score = str(m_data.get('radiant_score', '-')), str(m_data.get('dire_score', '-'))
        winner = "N/A"
        if m_data.get('radiant_win') is True: winner = f"[cyan b]{r_name}[/cyan b]"
        elif m_data.get('radiant_win') is False: winner = f"[orange3 b]{d_name}[/orange3 b]"

        rad_p_list, dire_p_list = [], []
        for p in m_data.get("players", []):
            acc_id = p.get("account_id")
            p_id_str_val = custom_names.get(acc_id, str(acc_id) if acc_id else p.get("personaname", "N/A"))
            h_name_disp = _get_hero_display_name_from_id(p.get("hero_id"), h_map)
            p_detail = f"{p_id_str_val} ({h_name_disp})"
            is_rad = p.get("isRadiant", p.get("player_slot", 100) < 100)
            (rad_p_list if is_rad else dire_p_list).append(p_detail)
        
        table.add_row(str(m_id), r_name, d_name, f"{r_score}-{d_score}", winner, "\n".join(rad_p_list) or "N/A", "\n".join(dire_p_list) or "N/A", fetched)
        displayed_count +=1
    conn.close()
    if displayed_count > 0:
        console.print(table); console.print(f"\nDisplayed: {displayed_count}")
        if limit and displayed_count >= limit: console.print(f"Limit: {limit}. Use -l to change.")
        elif search_team: console.print(f"Filtered by team: '{search_team}'")
    elif search_team: console.print(f"[yellow]No matches for team '{search_team}' post-parse.[/yellow]")

@app.command(name="find-player-hero", help="Find matches for player (opt. hero). Opt: -l limit.")
def find_player_hero_matches(p_id_str_arg: str = typer.Argument(..., help="Player AccID/CustomName/Persona."), h_name_in: Optional[str] = typer.Argument(None, help="Hero name (e.g. 'antimage')."), limit: Optional[int] = typer.Option(None, "-l", help="Num matches.")):
    conn = get_db_connection()
    target_h_id: Optional[int] = None; h_search = False; h_map = _get_all_heroes_map_from_db(conn); custom_names = _get_all_player_custom_names_map_from_db(conn)
    actual_h_name = ""
    if h_name_in:
        h_search = True; target_h_id = get_hero_id_from_name_db(h_name_in, conn)
        if target_h_id is None: console.print(f"[red]Hero '{h_name_in}' not found/ambiguous.[/red]"); conn.close(); raise typer.Exit(code=1)
        actual_h_name = _get_hero_display_name_from_id(target_h_id, h_map)
        console.print(f"[info]Target Hero: {target_h_id} ('{actual_h_name}')[/info]")
        panel_title = f"Player '{p_id_str_arg}' on Hero '{actual_h_name}'"
        table_suffix = f" on Hero '{actual_h_name}' (ID: {target_h_id})"
    else: panel_title = f"Matches for Player '{p_id_str_arg}'"; table_suffix = ""
    console.print(Panel(panel_title, expand=False))

    search_acc_id: Optional[int] = None; by_persona = False; display_term = p_id_str_arg
    try:
        search_acc_id = int(p_id_str_arg)
        cust_name = custom_names.get(search_acc_id)
        display_term = f"{cust_name} (ID: {search_acc_id})" if cust_name else f"ID: {search_acc_id}"
    except ValueError:
        found_cust = False
        for acc_id, c_name in custom_names.items():
            if p_id_str_arg.lower() == c_name.lower(): search_acc_id = acc_id; display_term = f"{c_name} (ID: {acc_id})"; found_cust = True; break
        if not found_cust: by_persona = True; display_term = f"Persona ~'{p_id_str_arg}'"
    
    sql_search = str(search_acc_id) if search_acc_id else p_id_str_arg
    try: match_rows = conn.execute("SELECT match_id, data, fetched_at FROM matches WHERE data LIKE ? ORDER BY fetched_at DESC", (f'%{sql_search}%',)).fetchall()
    except sqlite3.Error as e: console.print(f"[red]SQLite err: {e}[/red]"); conn.close(); raise typer.Exit(code=1)
    if not match_rows: console.print(f"[yellow]No matches in DB for '{p_id_str_arg}' (initial SQL filter).[/yellow]"); conn.close(); return

    found_list = []
    for row in match_rows:
        try: m_data: Dict[str, Any] = json.loads(row["data"])
        except json.JSONDecodeError: continue
        player_met_criteria = False; hero_played_id: Optional[int] = None
        for p in m_data.get("players", []):
            matches_id_crit = False
            if search_acc_id: matches_id_crit = p.get("account_id") == search_acc_id
            elif by_persona: matches_id_crit = p_id_str_arg.lower() in p.get("personaname", "").lower()
            if matches_id_crit:
                hero_played_id = p.get("hero_id")
                if h_search: player_met_criteria = hero_played_id == target_h_id
                else: player_met_criteria = True
                if player_met_criteria: break
        if player_met_criteria:
            r_name = m_data.get('radiant_name') or m_data.get('radiant_team', {}).get('name', "Radiant")
            d_name = m_data.get('dire_name') or m_data.get('dire_team', {}).get('name', "Dire")
            winner = "N/A"; r_score, d_score = str(m_data.get('radiant_score', '-')), str(m_data.get('dire_score', '-'))
            if m_data.get('radiant_win') is True: winner = f"[cyan b]{r_name}[/cyan b]"
            elif m_data.get('radiant_win') is False: winner = f"[orange3 b]{d_name}[/orange3 b]"
            p_hero_disp = _get_hero_display_name_from_id(hero_played_id, h_map)
            found_list.append({"id": str(row["match_id"]), "r_name": r_name, "d_name": d_name, "score": f"{r_score}-{d_score}", "winner": winner, "fetched": row["fetched_at"], "p_hero": p_hero_disp})
            if limit and len(found_list) >= limit: break
    conn.close()

    if not found_list:
        console.print(f"[yellow]No matches for player '{display_term}'{(' played hero ' + actual_h_name) if h_search else ''}.[/yellow]"); return
    
    table = Table(title=f"Matches: Player '{display_term}'{table_suffix}", show_header=True, header_style="magenta")
    cols = ["MatchID", "PlayerHero", "Rad Team", "Dire Team", "Score", "Winner", "Fetched (UTC)"]
    for c in cols: table.add_column(c, min_width=15 if "Team" in c or "Hero" in c else 10, justify="center" if "ID" in c or "Score" in c or "Fetched" in c else "default")
    for d_item in found_list: table.add_row(d_item["id"], d_item["p_hero"], d_item["r_name"], d_item["d_name"], d_item["score"], d_item["winner"], d_item["fetched"])
    console.print(table); console.print(f"\nFound {len(found_list)} match(es).")
    if limit and len(found_list) >= limit: console.print(f"Limit: {limit}. Use -l to change.")

@app.command(name="show-match-json", help="Show JSON for match. Opts: -v visualize, -o output file.")
def show_match_json(m_id_arg: int = typer.Argument(..., help="Match ID."), visualize: bool = typer.Option(False, "-v", help="Interactive viz (browser). Needs gravis/networkx.", is_flag=True), out_file: Optional[str] = typer.Option(None, "-o", help="Save JSON to file (e.g. match.json).", show_default=False, dir_okay=False, writable=True)):
    console.print(Panel(f"JSON Data: Match ID {m_id_arg}", expand=False))
    conn = get_db_connection(); out_file_abs: Optional[str] = None
    try: row = conn.execute("SELECT data FROM matches WHERE match_id = ?", (m_id_arg,)).fetchone()
    except sqlite3.Error as e: console.print(f"[red]SQLite err: {e}[/red]"); conn.close(); raise typer.Exit(code=1)
    if not row or not row["data"]: console.print(f"[yellow]No data for match {m_id_arg}.[/yellow]"); conn.close(); return

    json_str = row["data"]; parsed_json: Optional[Dict[str, Any]] = None; viz_ok = False
    try:
        parsed_json = json.loads(json_str)
        if out_file:
            try:
                out_file_abs = os.path.abspath(out_file); out_dir = os.path.dirname(out_file_abs)
                if out_dir and not os.path.exists(out_dir): os.makedirs(out_dir); console.print(f"[info]Created dir: {out_dir}[/info]")
                with open(out_file_abs, 'w') as f: json.dump(parsed_json, f, indent=2)
                console.print(f"[green]JSON saved: {out_file_abs}[/green]")
            except Exception as e_fs: console.print(f"[red]Err saving to '{out_file}': {e_fs}[/red]")
        if visualize:
            console.print("[info]Viz requested...[/info]")
            if not VIS_LIBS_AVAILABLE: console.print("[yellow]Viz libs not installed.[/yellow]")
            elif parsed_json: viz_ok = _visualize_json_structure_gravis(parsed_json, m_id_arg, "Match")
            if not viz_ok: console.print("[yellow]Viz failed/skipped.[/yellow]")
        
        print_console = not ((out_file and out_file_abs and os.path.exists(out_file_abs)) or (visualize and viz_ok))
        if not print_console and ((visualize and not viz_ok) or (out_file and (not out_file_abs or not os.path.exists(out_file_abs)))):
            if typer.confirm("Viz/Save failed. Print JSON to console?", default=False): print_console = True
        if print_console and parsed_json: console.print(Syntax(json.dumps(parsed_json, indent=2), "json", theme="material", line_numbers=True))
    except json.JSONDecodeError:
        console.print(f"[red]Err: Data for {m_id_arg} not valid JSON.[/red]")
        if out_file:
            try:
                out_file_abs = os.path.abspath(out_file); out_dir = os.path.dirname(out_file_abs)
                if out_dir and not os.path.exists(out_dir): os.makedirs(out_dir)
                with open(out_file_abs, 'w') as f: f.write(json_str)
                console.print(f"[yellow]Saved raw (invalid JSON) data for {m_id_arg} to: {out_file_abs}[/yellow]")
            except IOError as e: console.print(f"[red]Err saving raw invalid JSON to '{out_file}': {e}[/red]")
        else: console.print("[info]Raw data (invalid JSON):"); console.print(json_str[:1000] + "..." if len(json_str) > 1000 else json_str)
    finally: conn.close()

@app.command(name="db-stats", help="Show DB stats (match/hero counts, etc).")
def db_stats():
    console.print(Panel("DB Stats", expand=False))
    conn = get_db_connection(); stats_dict = {}
    try:
        stats_dict["Matches"] = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        stats_dict["Heroes"] = conn.execute("SELECT COUNT(*) FROM heroes").fetchone()[0]
        stats_dict["Custom Names"] = conn.execute("SELECT COUNT(*) FROM player_custom_names").fetchone()[0]
        dates_row = conn.execute("SELECT MIN(fetched_at), MAX(fetched_at) FROM matches WHERE fetched_at IS NOT NULL AND fetched_at != ''").fetchone()
        stats_dict["Matches Fetched (UTC)"] = f"{dates_row[0]} to {dates_row[1]}" if dates_row and dates_row[0] else "N/A"
    except sqlite3.Error as e: console.print(f"[red]SQLite err: {e}[/red]"); conn.close(); raise typer.Exit(code=1)
    finally: conn.close()
    table = Table(title="DB Overview", show_lines=True); table.add_column("Stat", style="magenta"); table.add_column("Value", style="green")
    for k, v_item in stats_dict.items(): table.add_row(k, str(v_item))
    console.print(table)

@app.command(name="player-summary", help="Summary for player (matches, top heroes, WR).")
def player_summary(p_id_str_arg: str = typer.Argument(..., help="Player AccID/CustomName/Persona.")):
    console.print(Panel(f"Summary: Player '{p_id_str_arg}'", expand=False))
    conn = get_db_connection(); custom_names = _get_all_player_custom_names_map_from_db(conn); h_map = _get_all_heroes_map_from_db(conn)
    target_acc_id: Optional[int] = None; by_persona = False; display_term = p_id_str_arg; sql_like_term: str
    try:
        target_acc_id = int(p_id_str_arg); cust_name = custom_names.get(target_acc_id)
        display_term = f"{cust_name} (ID: {target_acc_id})" if cust_name else f"ID: {target_acc_id}"
        sql_like_term = f'%"account_id": {target_acc_id}%'
    except ValueError:
        resolved_cust = False
        for acc_id, c_name in custom_names.items():
            if p_id_str_arg.lower() == c_name.lower(): target_acc_id = acc_id; display_term = f"{c_name} (ID: {acc_id})"; sql_like_term = f'%"account_id": {target_acc_id}%'; resolved_cust = True; break
        if not resolved_cust: by_persona = True; esc_id = p_id_str_arg.replace('"', '""'); sql_like_term = f'%"personaname": "%{esc_id}%"%'
    
    try: match_rows = conn.execute("SELECT data FROM matches WHERE data LIKE ? ORDER BY fetched_at DESC", (sql_like_term,)).fetchall()
    except sqlite3.Error as e: console.print(f"[red]SQLite err: {e}[/red]"); conn.close(); return
    if not match_rows: console.print(f"[yellow]No matches for '{display_term}' (SQL filter).[/yellow]"); conn.close(); return

    played, wins = 0, 0; hero_perf: Dict[int, Dict[str, int]] = defaultdict(lambda: {"picks": 0, "wins": 0})
    for row in match_rows:
        try: m_data: Dict[str, Any] = json.loads(row["data"])
        except json.JSONDecodeError: continue
        p_match_info = None
        for p_info_loop in m_data.get("players", []):
            matches_crit = False
            if target_acc_id: matches_crit = p_info_loop.get("account_id") == target_acc_id
            elif by_persona: matches_crit = p_id_str_arg.lower() in p_info_loop.get("personaname", "").lower()
            if matches_crit: p_match_info = p_info_loop; break
        if p_match_info:
            played += 1; h_id = p_match_info.get("hero_id")
            is_rad = p_match_info.get("isRadiant", p_match_info.get("player_slot", 100) < 100)
            rad_won = m_data.get("radiant_win"); p_won = False
            if rad_won is not None and is_rad is not None:
                if (rad_won and is_rad) or (not rad_won and not is_rad): wins += 1; p_won = True
            if h_id: hero_perf[h_id]["picks"] += 1; hero_perf[h_id]["wins"] += int(p_won) # Ensure wins is int for defaultdict
    conn.close()
    if played == 0: console.print(f"[yellow]No confirmed matches for '{display_term}'.[/yellow]"); return
    win_rate = (wins / played * 100) if played > 0 else 0
    console.print(f"\n--- Summary: {display_term} ---"); console.print(f"Matches: {played}, Wins: {wins}, WR: {win_rate:.2f}%")
    if hero_perf:
        console.print("\n[b]Top Heroes (Max 5):[/b]")
        sorted_hero_perf = sorted(hero_perf.items(), key=lambda i: (i[1]["picks"], (i[1]['wins'] / i[1]['picks'] * 100) if i[1]['picks'] > 0 else 0), reverse=True)
        table = Table(show_header=True, header_style="cyan"); table.add_column("Hero", style="green"); table.add_column("Picks", justify="center"); table.add_column("WR (%)", justify="center")
        for h_id_val, stats_val in sorted_hero_perf[:5]:
            h_disp_name = _get_hero_display_name_from_id(h_id_val, h_map)
            h_wr = (stats_val['wins'] / stats_val['picks'] * 100) if stats_val['picks'] > 0 else 0
            table.add_row(h_disp_name, str(stats_val['picks']), f"{h_wr:.2f}%")
        console.print(table)
    else: console.print("No hero pick data.")

@app.command(name="hero-summary", help="Summary for hero (picks, WR, top players).")
def hero_summary(h_id_str_arg: str = typer.Argument(..., help="Hero name or ID.")):
    console.print(Panel(f"Summary: Hero '{h_id_str_arg}'", expand=False))
    conn = get_db_connection(); h_map = _get_all_heroes_map_from_db(conn); custom_names = _get_all_player_custom_names_map_from_db(conn)
    target_h_id: Optional[int] = None
    try: target_h_id = int(h_id_str_arg)
    except ValueError:
        target_h_id = get_hero_id_from_name_db(h_id_str_arg, conn)
        if target_h_id is None: console.print(f"[red]Hero '{h_id_str_arg}' not found/ambiguous.[/red]"); conn.close(); return
    actual_h_name = _get_hero_display_name_from_id(target_h_id, h_map)
    sql_like_term = f'%"hero_id": {target_h_id}%'
    try: match_rows = conn.execute("SELECT data FROM matches WHERE data LIKE ? ORDER BY fetched_at DESC", (sql_like_term,)).fetchall()
    except sqlite3.Error as e: console.print(f"[red]SQLite err: {e}[/red]"); conn.close(); return
    if not match_rows: console.print(f"[yellow]No matches for '{actual_h_name}' (ID: {target_h_id}) (SQL filter).[/yellow]"); conn.close(); return

    picks, wins = 0, 0; player_perf: Dict[Any, Dict[str, int]] = defaultdict(lambda: {"picks": 0, "wins": 0}) # Account ID can be None
    for row in match_rows:
        try: m_data: Dict[str, Any] = json.loads(row["data"])
        except json.JSONDecodeError: continue
        picker_info = None
        for p_info_loop in m_data.get("players", []):
            if p_info_loop.get("hero_id") == target_h_id: picker_info = p_info_loop; break
        if picker_info:
            picks += 1; acc_id_picker = picker_info.get("account_id") # Can be None
            is_rad = picker_info.get("isRadiant", picker_info.get("player_slot", 100) < 100)
            rad_won = m_data.get("radiant_win"); picker_won = False
            if rad_won is not None and is_rad is not None:
                if (rad_won and is_rad) or (not rad_won and not is_rad): wins += 1; picker_won = True
            
            # Use a placeholder if account_id is None for defaultdict key
            player_key = acc_id_picker if acc_id_picker is not None else "UnknownPlayer" 
            player_perf[player_key]["picks"] += 1
            player_perf[player_key]["wins"] += int(picker_won) # Ensure wins is int
    conn.close()
    if picks == 0: console.print(f"[yellow]Hero '{actual_h_name}' not picked in matches.[/yellow]"); return
    overall_wr = (wins / picks * 100) if picks > 0 else 0
    console.print(f"\n--- Summary: {actual_h_name} (ID: {target_h_id}) ---"); console.print(f"Picks: {picks}, Wins: {wins}, WR: {overall_wr:.2f}%")
    if player_perf:
        console.print("\n[b]Top Players (Max 5):[/b]")
        # Filter out "UnknownPlayer" before sorting if it exists and is not desired in top players list
        valid_player_perf = {k: v for k, v in player_perf.items() if k != "UnknownPlayer"}
        sorted_p_perf = sorted(valid_player_perf.items(), key=lambda i: (i[1]["picks"], (i[1]['wins'] / i[1]['picks'] * 100) if i[1]['picks'] > 0 else 0), reverse=True)
        
        table = Table(show_header=True, header_style="cyan"); table.add_column("Player (Name/ID)", style="green"); table.add_column("Picks", justify="center"); table.add_column("WR (%) w/ Hero", justify="center")
        for acc_id_val, stats_val in sorted_p_perf[:5]:
            # acc_id_val here will be actual account_id
            p_disp_name = custom_names.get(acc_id_val, f"ID: {acc_id_val}")
            p_h_wr = (stats_val['wins'] / stats_val['picks'] * 100) if stats_val['picks'] > 0 else 0
            table.add_row(p_disp_name, str(stats_val['picks']), f"{p_h_wr:.2f}%")
        console.print(table)
    else: console.print("No player-specific data for this hero.")

@app.command(name="analyze-lanes", help="Analyze laning phase for a match ID (uses configured KPIs).")
def analyze_lanes_command(m_id_arg: int = typer.Argument(..., help="Match ID.")):
    console.print(Panel(f"Lane Analysis: Match ID {m_id_arg}", expand=False))
    conn = get_db_connection(); h_map = _get_all_heroes_map_from_db(conn); custom_names = _get_all_player_custom_names_map_from_db(conn)
    # KPI_PARAMS is global and used by _perform_core_lane_analysis
    analysis = _perform_core_lane_analysis(m_id_arg, conn, h_map, custom_names)
    if not analysis or analysis.get("error"): console.print(f"[red]Lane analysis err for {m_id_arg}: {analysis.get('error', 'Unknown')}[/red]"); conn.close(); return
    
    try: # Re-fetch for detailed tables if needed by display logic
        row_data = conn.execute("SELECT data FROM matches WHERE match_id = ?", (m_id_arg,)).fetchone()
        m_data = json.loads(row_data["data"]) if row_data and row_data["data"] else None
    except Exception as e: console.print(f"[red]Err re-fetching for display: {e}[/red]"); conn.close(); return
    finally: conn.close() # Close connection after command finishes
    if not m_data: console.print(f"[red]Could not re-fetch for display {m_id_arg}.[/red]"); return

    players_tables = m_data.get("players", []) # Fresh player data for tables
    for p_idx, p_info in enumerate(players_tables): # Enrich with display names
        acc_id = p_info.get("account_id"); d_name = p_info.get("personaname", f"Slot {p_info.get('player_slot', p_idx)}")
        if acc_id and acc_id in custom_names: d_name = custom_names[acc_id]
        players_tables[p_idx]["display_name"] = d_name
        if "isRadiant" not in p_info and "player_slot" in p_info: players_tables[p_idx]["isRadiant"] = p_info["player_slot"] < 100
    
    lanes_assign_tables = _identify_laners(players_tables, h_map)
    console.print(f"[info]Draft: {analysis.get('draft_order', 'N/A')}[/info]\n")

    # Get configured minute marks for display
    primary_minute = KPI_PARAMS["analysis_minute_mark"]
    secondary_minute = KPI_PARAMS["display_secondary_minute_mark"]
    tower_limit_display_min = KPI_PARAMS['early_tower_kill_time_limit_seconds'] // 60
    kill_limit_display_min = KPI_PARAMS['kill_death_analysis_time_limit_seconds'] // 60


    for lane_key, lane_info in analysis.get("lanes", {}).items():
        if lane_key == "unknown": continue
        console.print(Panel(f"Lane: {lane_key.upper()}", style="yellow b", expand=False))
        teams_in_lane_table = lanes_assign_tables.get(lane_key, {"radiant": [], "dire": []})
        for team_str, team_players in teams_in_lane_table.items():
            if not team_players: console.print(f"[dim]No {team_str} players in {lane_key.upper()} for table.[/dim]"); continue
            team_color = "cyan" if team_str == "radiant" else "orange3"
            table = Table(title=f"{team_str.capitalize()} - {lane_key.upper()} @ {secondary_minute}m/{primary_minute}m", title_style=f"b {team_color}", header_style="magenta")
            cols = ["Player", "Hero", "Lvl", "LH", "DN", "Gold", "GPM", "XP", "XPM", f"K (vs Lane @{kill_limit_display_min}m)", f"D (to Lane @{kill_limit_display_min}m)"]
            for c in cols: table.add_column(c, min_width=12, justify="center" if c not in ["Player","Hero"] else "default", overflow="fold" if c=="Player" else "default")
            
            opp_team_str_val = "dire" if team_str == "radiant" else "radiant"
            opp_laners_h_names = [p.get('hero_name_parsed', '') for p in lanes_assign_tables[lane_key].get(opp_team_str_val,[]) if p.get('hero_name_parsed')]

            for p_obj in team_players:
                kpis_sec = _get_player_kpis_at_minute(p_obj, secondary_minute, h_map)
                kpis_prim = _get_player_kpis_at_minute(p_obj, primary_minute, h_map)
                kills, deaths = _analyze_lane_kill_death_events(p_obj, players_tables, opp_laners_h_names, KPI_PARAMS["kill_death_analysis_time_limit_seconds"])
                table.add_row(
                    p_obj.get("display_name"), kpis_prim["hero"], 
                    f"{kpis_sec['level']}/{kpis_prim['level']}", f"{kpis_sec['lh']}/{kpis_prim['lh']}", 
                    f"{kpis_sec['dn']}/{kpis_prim['dn']}", f"{kpis_sec['gold']}/{kpis_prim['gold']}",
                    f"{kpis_sec['gpm']}/{kpis_prim['gpm']}", f"{kpis_sec['xp']}/{kpis_prim['xp']}",
                    f"{kpis_sec['xpm']}/{kpis_prim['xpm']}", str(kills), str(deaths)
                )
            console.print(table)
        
        sum_panel_content = Text()
        details = lane_info.get("summary_details", {})
        sum_panel_content.append(f"Total Gold Adv @{primary_minute}m (R-D): {details.get('gold_diff', 0):+G}\n")
        sum_panel_content.append(f"Total XP Adv @{primary_minute}m (R-D): {details.get('xp_diff', 0):+G}\n")
        sum_panel_content.append(f"Net Kills in Lane @{kill_limit_display_min}m (R-D): {details.get('kill_diff', 0):+}\n")
        sum_panel_content.append(f"Rad T1 Fallen by {tower_limit_display_min}m: {details.get('radiant_t1_fallen', False)}\n")
        sum_panel_content.append(f"Dire T1 Fallen by {tower_limit_display_min}m: {details.get('dire_t1_fallen', False)}\n")
        verdict_console = lane_info.get('verdict_text', 'N/A')
        if "Radiant Ahead" in verdict_console: verdict_console = f"[cyan b]{verdict_console}[/cyan b]"
        elif "Dire Ahead" in verdict_console: verdict_console = f"[orange3 b]{verdict_console}[/orange3 b]"
        sum_panel_content.append(f"Verdict: "); sum_panel_content.append(Text.from_markup(verdict_console))
        console.print(Panel(sum_panel_content, title=f"Overall {lane_key.upper()} Summary (@{primary_minute}min)", expand=False, border_style="green"))
        console.print("-" * console.width)

@app.command(name="export-match-ids", help="Export all match IDs to CSV.")
def export_match_ids(out_file: str = typer.Option(..., "-o", help="Output CSV file (e.g. all_ids.csv).", dir_okay=False, writable=True)):
    console.print(Panel(f"Export Match IDs to '{out_file}'", expand=False))
    conn = get_db_connection()
    try: rows = conn.execute("SELECT match_id FROM matches ORDER BY match_id ASC").fetchall()
    except sqlite3.Error as e: console.print(f"[red]SQLite err: {e}[/red]"); conn.close(); raise typer.Exit(code=1)
    finally: conn.close()
    if not rows: console.print("[yellow]No match IDs to export.[/yellow]"); return
    try:
        out_file_abs = os.path.abspath(out_file); out_dir = os.path.dirname(out_file_abs)
        if out_dir and not os.path.exists(out_dir): os.makedirs(out_dir); console.print(f"[info]Created dir: {out_dir}[/info]")
        with open(out_file_abs, 'w', newline='') as csvf:
            writer = csv.writer(csvf); writer.writerow(["match_id"])
            for r_item in rows: writer.writerow([r_item["match_id"]])
        console.print(f"[green]Exported {len(rows)} IDs to: {out_file_abs}[/green]")
    except Exception as e_fs: console.print(f"[red]Err writing to '{out_file}': {e_fs}[/red]")

@app.command(name="batch-analyze-lanes", help="Batch analyze lanes from CSV input, output to CSV (uses configured KPIs).")
def batch_analyze_lanes(in_file: str = typer.Argument(..., help="Input CSV with 'match_id' header.", exists=True, file_okay=True, dir_okay=False, readable=True), out_file: str = typer.Option(..., "-o", help="Output CSV for results (e.g. batch_analysis.csv).", dir_okay=False, writable=True)):
    console.print(Panel(f"Batch Lane Analysis: '{in_file}' -> '{out_file}'", expand=False))
    m_ids_proc = []
    try:
        with open(in_file, 'r', newline='') as csvf:
            reader = csv.DictReader(csvf)
            if "match_id" not in reader.fieldnames: console.print(f"[red]Err: '{in_file}' needs 'match_id' header.[/red]"); raise typer.Exit(code=1)
            for r_item in reader:
                try: m_ids_proc.append(int(r_item["match_id"]))
                except ValueError: console.print(f"[yellow]Warn: Skip invalid ID '{r_item['match_id']}' in '{in_file}'.[/yellow]")
    except Exception as e_csv: console.print(f"[red]Err reading '{in_file}': {e_csv}[/red]"); raise typer.Exit(code=1)
    if not m_ids_proc: console.print(f"[yellow]No valid IDs in '{in_file}'.[/yellow]"); return

    conn = get_db_connection(); h_map = _get_all_heroes_map_from_db(conn); custom_names = _get_all_player_custom_names_map_from_db(conn)
    out_file_abs = os.path.abspath(out_file); out_dir = os.path.dirname(out_file_abs)
    if out_dir and not os.path.exists(out_dir):
        try: os.makedirs(out_dir); console.print(f"[info]Created dir: {out_dir}[/info]")
        except OSError as e: console.print(f"[red]Err creating dir '{out_dir}': {e}[/red]"); conn.close(); raise typer.Exit(code=1)
            
    fields = ["match_id", "draft_order", "top_radiant_score", "top_dire_score", "top_verdict", "mid_radiant_score", "mid_dire_score", "mid_verdict", "bot_radiant_score", "bot_dire_score", "bot_verdict", "analysis_error"]
    proc_ok, proc_err = 0, 0

    with Progress(TextColumn("[prog.desc]{task.description}"), BarColumn(), TextColumn("[prog.perc]{task.percentage:>3.0f}%"), TimeRemainingColumn(), TimeElapsedColumn(), console=console, transient=True) as progress:
        task = progress.add_task("[cyan]Analyzing...", total=len(m_ids_proc))
        try:
            with open(out_file_abs, 'w', newline='') as csvout:
                writer = csv.DictWriter(csvout, fieldnames=fields); writer.writeheader()
                for m_id_loop in m_ids_proc:
                    # _perform_core_lane_analysis now uses global KPI_PARAMS
                    analysis = _perform_core_lane_analysis(m_id_loop, conn, h_map, custom_names)
                    row_write = {"match_id": m_id_loop, "analysis_error": None}
                    if analysis and not analysis.get("error"):
                        row_write["draft_order"] = analysis.get("draft_order", "N/A")
                        for lane in ["top", "mid", "bot"]:
                            lane_d = analysis.get("lanes", {}).get(lane, {})
                            row_write[f"{lane}_radiant_score"] = lane_d.get("radiant_score", 0)
                            row_write[f"{lane}_dire_score"] = lane_d.get("dire_score", 0)
                            row_write[f"{lane}_verdict"] = lane_d.get("verdict_text", "N/A")
                        proc_ok +=1
                    else: row_write["analysis_error"] = analysis.get("error", "Unknown err"); proc_err +=1
                    writer.writerow(row_write); progress.update(task, advance=1)
        except Exception as e_batch: console.print(f"[red]Batch processing err: {e_batch}[/red]"); conn.close(); raise typer.Exit(code=1)
    conn.close()
    console.print(f"\n[green]Batch complete.[/green]"); console.print(f"Processed: {proc_ok} to '{out_file_abs}'.")
    if proc_err > 0: console.print(f"[yellow]Errors for {proc_err} matches (see CSV).[/yellow]")

# --- New Typer command for KPI config management ---
@app.command(name="manage-kpi-config", help=f"View or update lane analysis KPI parameters (stored in {CONFIG_FILE_NAME}).")
def manage_kpi_config(
    show: bool = typer.Option(True, "--show/--no-show", help="Show current KPI parameters."),
    reset_to_defaults: bool = typer.Option(False, "--reset-defaults", help="Reset all parameters to their default values.", is_flag=True),
    # --- General Timing Parameters ---
    analysis_minute: Optional[int] = typer.Option(None, help="Primary minute for KPI calculations (e.g., GPM, Gold)."),
    kill_death_limit_sec: Optional[int] = typer.Option(None, help="Time limit (seconds) for K/D analysis in lane."),
    tower_kill_limit_sec: Optional[int] = typer.Option(None, help="Time limit (seconds) for 'early' tower fall."),
    display_secondary_minute: Optional[int] = typer.Option(None, help="Secondary minute for table display (e.g., 8min/10min stats)."),
    # --- Score Weight Parameters ---
    major_gold_thresh: Optional[int] = typer.Option(None, help="Threshold for major gold lead per laner."),
    minor_gold_thresh: Optional[int] = typer.Option(None, help="Threshold for minor gold lead per laner."),
    major_xp_thresh: Optional[int] = typer.Option(None, help="Threshold for major XP lead per laner."),
    minor_xp_thresh: Optional[int] = typer.Option(None, help="Threshold for minor XP lead per laner."),
    points_major_lead: Optional[int] = typer.Option(None, help="Points for a major gold/XP lead."),
    points_minor_lead: Optional[int] = typer.Option(None, help="Points for a minor gold/XP lead."),
    kill_diff_major_points_thresh: Optional[int] = typer.Option(None, help="Net kill difference for major points."),
    points_major_kill_diff: Optional[int] = typer.Option(None, help="Points for major kill difference."),
    kill_diff_minor_points_thresh: Optional[int] = typer.Option(None, help="Net kill difference for minor points."),
    points_minor_kill_diff: Optional[int] = typer.Option(None, help="Points for minor kill difference."),
    points_tower_kill: Optional[int] = typer.Option(None, help="Points for an early tower kill.")
):
    """Manages Lane Analysis KPI parameters."""
    global KPI_PARAMS # Ensure we are modifying the global instance used by analysis functions
    
    current_params = _load_kpi_params() # Load fresh from file or defaults
    params_changed = False

    if reset_to_defaults:
        current_params = DEFAULT_KPI_PARAMS.copy()
        _save_kpi_params(current_params)
        KPI_PARAMS = current_params.copy() # Update global
        console.print(f"[green]KPI parameters have been reset to defaults and saved to {CONFIG_FILE_NAME}.[/green]")
        # No need to process other options if reset
        if show: # Still show if requested
            pass # Will fall through to show logic
        else:
            return 

    # Update parameters if options are provided
    if analysis_minute is not None: current_params["analysis_minute_mark"] = analysis_minute; params_changed = True
    if kill_death_limit_sec is not None: current_params["kill_death_analysis_time_limit_seconds"] = kill_death_limit_sec; params_changed = True
    if tower_kill_limit_sec is not None: current_params["early_tower_kill_time_limit_seconds"] = tower_kill_limit_sec; params_changed = True
    if display_secondary_minute is not None: current_params["display_secondary_minute_mark"] = display_secondary_minute; params_changed = True

    # Update score weights (ensure "score_weights" dict exists)
    if "score_weights" not in current_params or not isinstance(current_params["score_weights"], dict):
        current_params["score_weights"] = DEFAULT_KPI_PARAMS["score_weights"].copy() # Initialize if missing/corrupt
        params_changed = True # Mark as changed because we had to initialize it

    sw = current_params["score_weights"] # Alias for easier access
    if major_gold_thresh is not None: sw["major_gold_lead_per_laner_threshold"] = major_gold_thresh; params_changed = True
    if minor_gold_thresh is not None: sw["minor_gold_lead_per_laner_threshold"] = minor_gold_thresh; params_changed = True
    if major_xp_thresh is not None: sw["major_xp_lead_per_laner_threshold"] = major_xp_thresh; params_changed = True
    if minor_xp_thresh is not None: sw["minor_xp_lead_per_laner_threshold"] = minor_xp_thresh; params_changed = True
    if points_major_lead is not None: sw["points_for_major_lead"] = points_major_lead; params_changed = True
    if points_minor_lead is not None: sw["points_for_minor_lead"] = points_minor_lead; params_changed = True
    if kill_diff_major_points_thresh is not None: sw["kill_difference_for_major_points"] = kill_diff_major_points_thresh; params_changed = True
    if points_major_kill_diff is not None: sw["points_for_major_kill_difference"] = points_major_kill_diff; params_changed = True
    if kill_diff_minor_points_thresh is not None: sw["kill_difference_for_minor_points"] = kill_diff_minor_points_thresh; params_changed = True
    if points_minor_kill_diff is not None: sw["points_for_minor_kill_difference"] = points_minor_kill_diff; params_changed = True
    if points_tower_kill is not None: sw["points_for_early_tower_kill"] = points_tower_kill; params_changed = True

    if params_changed and not reset_to_defaults: # Don't save again if just reset
        _save_kpi_params(current_params)
        KPI_PARAMS = current_params.copy() # Update global
        console.print(f"[green]KPI parameters updated and saved to {CONFIG_FILE_NAME}.[/green]")

    if show:
        console.print(Panel(f"Current Lane Analysis KPI Parameters (from {CONFIG_FILE_NAME})", expand=False, border_style="blue"))
        
        # Display General Parameters
        general_table = Table(title="General Timing & Analysis Parameters", show_header=False, box=None)
        general_table.add_column("Parameter", style="dim")
        general_table.add_column("Value")
        general_table.add_row("Primary Analysis Minute Mark", str(current_params.get("analysis_minute_mark")))
        general_table.add_row("Kill/Death Analysis Time Limit (seconds)", str(current_params.get("kill_death_analysis_time_limit_seconds")))
        general_table.add_row("Early Tower Kill Time Limit (seconds)", str(current_params.get("early_tower_kill_time_limit_seconds")))
        general_table.add_row("Display Secondary Minute Mark (for tables)", str(current_params.get("display_secondary_minute_mark")))
        console.print(general_table)

        # Display Score Weights
        weights_table = Table(title="Score Weights", show_header=False, box=None)
        weights_table.add_column("Parameter", style="dim")
        weights_table.add_column("Value")
        if "score_weights" in current_params and isinstance(current_params["score_weights"], dict):
            for key, value in current_params["score_weights"].items():
                # Make key more readable
                readable_key = key.replace("_", " ").capitalize()
                weights_table.add_row(readable_key, str(value))
        else:
            weights_table.add_row("Score weights configuration is missing or invalid.", "")
        console.print(weights_table)
        
        console.print(f"\n[dim]These parameters are loaded from '{CONFIG_FILE_NAME}'. If the file is missing or invalid, defaults are used and a new file is created.[/dim]")

if __name__ == "__main__":
    console.print(Panel("[b green]OpenDota DB Viewer & Analyzer[/b green]", expand=False))
    app()

