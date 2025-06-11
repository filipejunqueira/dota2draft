# dota2draft/analysis.py

import json
import os
import webbrowser
from typing import Optional, Dict, Any, List, Tuple

from rich.console import Console # Keep for Rich table/panel output
from rich.panel import Panel # Keep for Rich panel output

from .config_loader import CONFIG
from .logger_config import logger # Import the configured logger

# Optional imports for visualization
try:
    import networkx as nx
    import gravis as gv
    VISUALIZATION_LIBRARIES_AVAILABLE = True
except ImportError:
    VISUALIZATION_LIBRARIES_AVAILABLE = False
    logger.warning("Visualization libraries (gravis, networkx) not found. Visualization features will be disabled. Install with: pip install gravis networkx")

console = Console() # Keep for direct Rich object rendering

KPI_PARAMETERS = CONFIG['kpi_parameters']
logger.debug(f"KPI Parameters loaded: {KPI_PARAMETERS}")

# --- Analysis Helper Functions ---
# ... (No console.print in these helpers, mostly calculations and data transformations) ...

def _parse_hero_name_from_internal_key(internal_hero_key: Optional[str]) -> str:
    """Converts 'npc_dota_hero_antimage' to 'antimage'."""
    if not internal_hero_key: return "Unknown"
    return internal_hero_key.replace("npc_dota_hero_", "") if internal_hero_key.startswith("npc_dota_hero_") else internal_hero_key

def _get_hero_display_name_from_hero_id(hero_id: Optional[int], hero_id_to_name_map: Dict[int, str]) -> str:
    """Gets a parsed, human-readable hero name from a hero ID using the provided map."""
    if hero_id is None: return "N/A"
    internal_name = hero_id_to_name_map.get(hero_id)
    return _parse_hero_name_from_internal_key(internal_name) if internal_name else f"ID:{hero_id}"

def _identify_laning_players(players_data: List[Dict[str, Any]], hero_id_to_name_map: Dict[int, str]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    lanes_assignment = {
        "top": {"radiant": [], "dire": []}, "mid": {"radiant": [], "dire": []},
        "bot": {"radiant": [], "dire": []}, "unknown": {"radiant": [], "dire": []} 
    }
    for player_info in players_data:
        is_radiant_player = player_info.get("isRadiant", player_info.get("player_slot", 100) < 100)
        team_name = "radiant" if is_radiant_player else "dire"
        assigned_lane_id = player_info.get("lane")
        lane_name = "unknown"
        if assigned_lane_id is not None:
            if team_name == "radiant":
                if assigned_lane_id == 1: lane_name = "bot"
                elif assigned_lane_id == 2: lane_name = "mid"
                elif assigned_lane_id == 3: lane_name = "top"
            else:
                if assigned_lane_id == 1: lane_name = "top"
                elif assigned_lane_id == 2: lane_name = "mid"
                elif assigned_lane_id == 3: lane_name = "bot"
        player_info['hero_name_parsed'] = _get_hero_display_name_from_hero_id(player_info.get('hero_id'), hero_id_to_name_map)
        player_info['team_string'] = team_name
        lanes_assignment[lane_name][team_name].append(player_info)
    return lanes_assignment

def _get_player_kpis_at_specific_minute(player_match_data: Dict[str, Any], minute_mark: int, hero_id_to_name_map: Dict[int, str]) -> Dict[str, Any]:
    kpis = {"lh": 0, "dn": 0, "gold": 0, "xp": 0, "gpm": 0, "xpm": 0, "level": 1, "hero": "N/A", "name": "N/A"}
    kpis["hero"] = _get_hero_display_name_from_hero_id(player_match_data.get('hero_id'), hero_id_to_name_map)
    kpis["name"] = player_match_data.get("display_name", player_match_data.get("personaname", f"Slot {player_match_data.get('player_slot', '?')}"))
    
    def get_timeseries_value(array_name: str, default_value: int = 0) -> int:
        value_array = player_match_data.get(array_name)
        if value_array and isinstance(value_array, list):
            return value_array[minute_mark] if minute_mark < len(value_array) else (value_array[-1] if value_array else default_value)
        return default_value

    kpis["lh"] = get_timeseries_value("lh_t")
    kpis["dn"] = get_timeseries_value("dn_t")
    kpis["gold"] = get_timeseries_value("gold_t")
    kpis["xp"] = get_timeseries_value("xp_t")
    if minute_mark > 0:
        kpis["gpm"] = round(kpis["gold"] / minute_mark)
        kpis["xpm"] = round(kpis["xp"] / minute_mark)
    return kpis

def _analyze_lane_kill_death_events_for_player(laner_player_object: Dict[str, Any], all_players_data: List[Dict[str, Any]], opposing_laner_hero_names: List[str], time_limit_seconds: int) -> Tuple[int, int]:
    kills_on_opposing_laners = 0
    deaths_to_opposing_laners = 0
    laner_hero_name = laner_player_object.get('hero_name_parsed', 'UnknownHero')
    if laner_hero_name == 'UnknownHero' or not laner_hero_name: return 0, 0

    if laner_player_object.get("kills_log"):
        for kill_event in laner_player_object["kills_log"]:
            if kill_event.get("time", float('inf')) <= time_limit_seconds:
                victim_hero_name_key = kill_event.get("key")
                victim_hero_name_simplified = _parse_hero_name_from_internal_key(victim_hero_name_key)
                if victim_hero_name_simplified in opposing_laner_hero_names:
                    kills_on_opposing_laners += 1

    for potential_killer_object in all_players_data:
        if potential_killer_object.get("player_slot") == laner_player_object.get("player_slot") or potential_killer_object.get("isRadiant") == laner_player_object.get("isRadiant"): continue
        killer_hero_name_parsed = potential_killer_object.get('hero_name_parsed', 'UnknownHero')
        if killer_hero_name_parsed in opposing_laner_hero_names and potential_killer_object.get("kills_log"):
            for kill_event in potential_killer_object["kills_log"]:
                if kill_event.get("time", float('inf')) <= time_limit_seconds:
                    victim_hero_name_key = kill_event.get("key")
                    victim_hero_name_simplified = _parse_hero_name_from_internal_key(victim_hero_name_key)
                    if victim_hero_name_simplified == laner_hero_name:
                        deaths_to_opposing_laners += 1
    return kills_on_opposing_laners, deaths_to_opposing_laners

def _check_early_tower_status_by_time(objectives_data: Optional[List[Dict[str, Any]]], time_limit_seconds: int) -> Dict[str, Dict[str, bool]]:
    tower_status = {
        "top": {"radiant_t1_destroyed": False, "dire_t1_destroyed": False},
        "mid": {"radiant_t1_destroyed": False, "dire_t1_destroyed": False},
        "bot": {"radiant_t1_destroyed": False, "dire_t1_destroyed": False},
    }
    if not objectives_data: return tower_status

    tower_map = {
        "npc_dota_goodguys_tower1_top": ("top", "radiant_t1_destroyed"),
        "npc_dota_badguys_tower1_top": ("top", "dire_t1_destroyed"),
        "npc_dota_goodguys_tower1_mid": ("mid", "radiant_t1_destroyed"),
        "npc_dota_badguys_tower1_mid": ("mid", "dire_t1_destroyed"),
        "npc_dota_goodguys_tower1_bot": ("bot", "radiant_t1_destroyed"),
        "npc_dota_badguys_tower1_bot": ("bot", "dire_t1_destroyed"),
    }

    for event in objectives_data:
        if event.get("type") == "building_kill" and event.get("time", float('inf')) <= time_limit_seconds:
            tower_key = event.get("key")
            if tower_key in tower_map:
                lane, status_key = tower_map[tower_key]
                tower_status[lane][status_key] = True
    return tower_status

def _extract_draft_order_string(match_data: Dict[str, Any], hero_id_to_name_map: Dict[int, str]) -> str:
    if not match_data or 'picks_bans' not in match_data or not match_data['picks_bans']:
        return "N/A"
    draft_actions = []
    # Sort by 'order' to ensure the draft sequence is correct
    for action in sorted(match_data['picks_bans'], key=lambda x: x['order']):
        team = "Radiant" if action['team'] == 0 else "Dire"
        action_type = "Pick" if action['is_pick'] else "Ban"
        hero_name = _get_hero_display_name_from_hero_id(action['hero_id'], hero_id_to_name_map)
        draft_actions.append(f"{team} {action_type}: {hero_name}")
    return '; '.join(draft_actions)

# --- Core Analysis Function ---
def perform_core_lane_analysis_for_match(match_data: Dict[str, Any], hero_id_to_name_map: Dict[int, str]) -> Dict[str, Any]:
    """
    Performs the core laning phase analysis for a given match.
    Returns a dictionary with analysis results or an error message.
    """
    logger.debug(f"Performing core lane analysis for match ID: {match_data.get('match_id')}")
    if not match_data or 'players' not in match_data:
        logger.error("Match data is missing or does not contain 'players' information for analysis.")
        return {"error": "Match data is missing or does not contain 'players' information."}

    params = KPI_PARAMETERS
    minute_mark = params['analysis_minute_mark']
    weights = params['score_weights']
    logger.debug(f"Using minute_mark: {minute_mark} and score_weights: {weights}")

    lanes = _identify_laning_players(match_data['players'], hero_id_to_name_map)
    analysis_results = {
        "match_id": match_data.get('match_id'),
        "draft_order": _extract_draft_order_string(match_data, hero_id_to_name_map),
        "lanes": {lane: {"radiant": {}, "dire": {}} for lane in ["top", "mid", "bot"]},
        "analysis_error_message": None
    }

    early_tower_kills = _check_early_tower_status_by_time(match_data.get('objectives'), params['early_tower_kill_time_limit_seconds'])

    for lane_name in ["top", "mid", "bot"]:
        # ... (rest of the analysis logic remains, no direct console.print calls here) ...
        radiant_laners = lanes[lane_name]['radiant']
        dire_laners = lanes[lane_name]['dire']
        radiant_score, dire_score = 0, 0
        if early_tower_kills[lane_name]['dire_t1_destroyed']:
            radiant_score += weights['points_for_early_tower_kill']
        if early_tower_kills[lane_name]['radiant_t1_destroyed']:
            dire_score += weights['points_for_early_tower_kill']
        if not radiant_laners or not dire_laners:
            analysis_results['lanes'][lane_name]['radiant']['score'] = radiant_score
            analysis_results['lanes'][lane_name]['dire']['score'] = dire_score
            continue
        radiant_kpis = [ _get_player_kpis_at_specific_minute(p, minute_mark, hero_id_to_name_map) for p in radiant_laners ]
        dire_kpis = [ _get_player_kpis_at_specific_minute(p, minute_mark, hero_id_to_name_map) for p in dire_laners ]
        total_radiant_gold = sum(k['gold'] for k in radiant_kpis)
        total_dire_gold = sum(k['gold'] for k in dire_kpis)
        total_radiant_xp = sum(k['xp'] for k in radiant_kpis)
        total_dire_xp = sum(k['xp'] for k in dire_kpis)
        gold_diff = total_radiant_gold - total_dire_gold
        if gold_diff >= weights['major_gold_lead_per_laner_threshold'] * len(radiant_laners):
            radiant_score += weights['points_for_major_lead']
        elif gold_diff >= weights['minor_gold_lead_per_laner_threshold'] * len(radiant_laners):
            radiant_score += weights['points_for_minor_lead']
        elif gold_diff <= -weights['major_gold_lead_per_laner_threshold'] * len(dire_laners):
            dire_score += weights['points_for_major_lead']
        elif gold_diff <= -weights['minor_gold_lead_per_laner_threshold'] * len(dire_laners):
            dire_score += weights['points_for_minor_lead']
        xp_diff = total_radiant_xp - total_dire_xp
        if xp_diff >= weights['major_xp_lead_per_laner_threshold'] * len(radiant_laners):
            radiant_score += weights['points_for_major_lead']
        elif xp_diff >= weights['minor_xp_lead_per_laner_threshold'] * len(radiant_laners):
            radiant_score += weights['points_for_minor_lead']
        elif xp_diff <= -weights['major_xp_lead_per_laner_threshold'] * len(dire_laners):
            dire_score += weights['points_for_major_lead']
        elif xp_diff <= -weights['minor_xp_lead_per_laner_threshold'] * len(dire_laners):
            dire_score += weights['points_for_minor_lead']
        radiant_hero_names = [p['hero_name_parsed'] for p in radiant_laners]
        dire_hero_names = [p['hero_name_parsed'] for p in dire_laners]
        total_radiant_kills = sum(_analyze_lane_kill_death_events_for_player(p, match_data['players'], dire_hero_names, params['kill_death_analysis_time_limit_seconds'])[0] for p in radiant_laners)
        total_dire_kills = sum(_analyze_lane_kill_death_events_for_player(p, match_data['players'], radiant_hero_names, params['kill_death_analysis_time_limit_seconds'])[0] for p in dire_laners)
        kill_diff = total_radiant_kills - total_dire_kills
        if kill_diff >= weights['kill_difference_for_major_points']:
            radiant_score += weights['points_for_major_kill_difference']
        elif kill_diff >= weights['kill_difference_for_minor_points']:
            radiant_score += weights['points_for_minor_kill_difference']
        elif kill_diff <= -weights['kill_difference_for_major_points']:
            dire_score += weights['points_for_major_kill_difference']
        elif kill_diff <= -weights['kill_difference_for_minor_points']:
            dire_score += weights['points_for_minor_kill_difference']
        analysis_results['lanes'][lane_name]['radiant'] = {
            'score': radiant_score,
            'heroes': radiant_hero_names,
            'kills': total_radiant_kills
        }
        analysis_results['lanes'][lane_name]['dire'] = {
            'score': dire_score,
            'heroes': dire_hero_names,
            'kills': total_dire_kills
        }

    logger.info(f"Lane analysis completed for match ID: {analysis_results.get('match_id')}")
    return analysis_results

# --- Visualization Function ---
def visualize_json_structure(json_data: Dict[str, Any], item_identifier: Any, item_type_name: str = "Match") -> bool:
    if not VISUALIZATION_LIBRARIES_AVAILABLE:
        logger.warning("Skipping JSON structure visualization as Gravis/NetworkX are not installed.")
        # The console.print for missing libraries is kept in the CLI or a higher level for user visibility
        return False
    if not json_data or not isinstance(json_data, (dict, list)):
        logger.warning("Invalid or empty JSON data provided for visualization.")
        return False

    logger.info(f"Generating interactive graph for {item_type_name} ID {item_identifier}...")
    graph = nx.DiGraph()

    # ... (rest of the visualization logic remains, no direct console.print here for status)
    def add_nodes_and_edges_recursively(data_node: Any, parent_node_name: Optional[str] = None):
        if isinstance(data_node, dict):
            for key, value in data_node.items():
                current_node_name = f"{parent_node_name}.{key}" if parent_node_name else str(key)
                value_snippet = str(value)[:147] + "..." if len(str(value)) > 150 else str(value)
                graph.add_node(current_node_name, title=value_snippet, label=str(key), color='lightblue')
                if parent_node_name:
                    graph.add_edge(parent_node_name, current_node_name)
                if isinstance(value, (dict, list)):
                    add_nodes_and_edges_recursively(value, current_node_name)
        elif isinstance(data_node, list):
            for index, item in enumerate(data_node):
                current_node_name = f"{parent_node_name}[{index}]"
                value_snippet = str(item)[:147] + "..." if len(str(item)) > 150 else str(item)
                graph.add_node(current_node_name, title=value_snippet, label=f"[{index}]", color='lightgreen')
                if parent_node_name:
                    graph.add_edge(parent_node_name, current_node_name)
                if isinstance(item, (dict, list)):
                    add_nodes_and_edges_recursively(item, current_node_name)

    root_node_label = f"{item_type_name} ID: {item_identifier}"
    graph.add_node(root_node_label, label=root_node_label, color='salmon', size=15)
    add_nodes_and_edges_recursively(json_data, root_node_label)

    output_filename_absolute = os.path.abspath(f"{item_type_name.lower()}_{item_identifier}_visualization.html")
    try:
        fig = gv.d3(graph, graph_height=800, node_label_data_source='label', show_menu=True)
        fig.export_html(output_filename_absolute, overwrite=True)
        logger.info(f"Successfully generated interactive visualization: {output_filename_absolute}")
        if webbrowser:
            try:
                webbrowser.open(f"file://{output_filename_absolute}")
            except webbrowser.Error as wb_error:
                logger.warning(f"Could not open visualization in browser: {wb_error}. Please open manually: {output_filename_absolute}")
        return True
    except Exception as e:
        logger.error(f"Error during Gravis visualization for {item_identifier}: {e}", exc_info=True)
        return False
