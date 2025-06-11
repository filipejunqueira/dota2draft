# Import necessary libraries
import requests  # For making HTTP requests to the OpenDota API (getting data from the internet)
import json      # For working with JSON data (the format OpenDota API uses)
from datetime import datetime # For handling dates and times (like when data was fetched)
from typing import List, Optional, Dict, Any # For type hinting, making code more readable
import sqlite3   # For using SQLite, a simple file-based database

# Import libraries for creating a nice command-line interface (CLI)
import typer # For creating CLI commands easily
from rich.console import Console # For beautiful output in the terminal
from rich.table import Table     # For displaying data in tables
from rich.panel import Panel     # For displaying text in bordered panels
from rich.text import Text       # For styled text
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn # For showing progress bars

# Initialize Typer app for defining CLI commands
# The 'help' text will be shown if the user runs `python league_info.py --help`
app = typer.Typer(help="OpenDota API CLI tool (Class-based) with SQLite caching.")
# Initialize Rich Console for pretty printing to the terminal
console = Console()

# Define the name of our SQLite database file.
# This file will store all the data we fetch to avoid re-fetching from the API.
DB_NAME = "opendota_league_info.db"

# --- Database Manager Class ---
# This class is responsible for all direct interactions with our SQLite database.
# It handles connecting to the database, creating tables, and reading/writing data.
class DBManager:
    def __init__(self, db_name: str = DB_NAME):
        """
        Constructor for DBManager.
        When a DBManager object is created, it will:
        1. Store the database file name.
        2. Call _init_tables() to make sure all necessary tables exist.
        """
        self.db_name = db_name
        self._init_tables() # Ensure tables are ready when the manager is created

    def _get_connection(self) -> sqlite3.Connection:
        """
        Helper method to establish a connection to the SQLite database.
        It's marked with an underscore `_` to indicate it's mostly for internal use within this class.
        Returns:
            sqlite3.Connection: A connection object to the database.
        """
        conn = sqlite3.connect(self.db_name)
        # This line makes it so we can access columns by their names (like a dictionary)
        # instead of just by their index (like a list).
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """
        Initializes the database by creating necessary tables if they don't already exist.
        This is crucial for the first time the script runs or if the DB file is deleted.
        """
        conn = self._get_connection() # Get a connection to the database
        cursor = conn.cursor()       # A cursor is used to execute SQL commands

        # SQL command to create the 'matches' table.
        # IF NOT EXISTS ensures it only creates the table if one with the same name isn't already there.
        # - match_id: The unique ID for a match (PRIMARY KEY means it's unique and used for fast lookups).
        # - data: Stores the full JSON data of the match as plain text.
        # - fetched_at: Records when this data was saved (defaults to the current time).
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id INTEGER PRIMARY KEY,
            data TEXT NOT NULL,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")

        # SQL command to create the 'heroes' table.
        # - hero_id: The unique ID for a hero.
        # - name: The hero's name.
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS heroes (
            hero_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")

        # SQL command to create the 'teams' table.
        # - team_id: The unique ID for a team.
        # - name: The team's name.
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")

        # SQL command to create the 'all_leagues' table.
        # This table will store a list of all leagues fetched from the API.
        # - league_id: The unique ID for a league.
        # - name: The league's name.
        # - tier: The league's tier (e.g., "premium", "professional").
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS all_leagues (
            league_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            tier TEXT,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")

        conn.commit() # Save the changes (table creations) to the database file
        conn.close()  # Close the connection
        console.print(f"Database '{self.db_name}' initialized/checked.", style="dim")

    # --- Match Data Operations ---
    def get_match_data(self, match_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific match's full details from the 'matches' table in the database.
        Args:
            match_id: The ID of the match to look for.
        Returns:
            A dictionary containing the match data if found, otherwise None.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        # SQL command to select the 'data' column for a specific 'match_id'.
        # The '?' is a placeholder that gets replaced by the match_id value to prevent SQL injection.
        cursor.execute("SELECT data FROM matches WHERE match_id = ?", (match_id,))
        row = cursor.fetchone() # Fetches one row (should be unique due to PRIMARY KEY)
        conn.close()
        if row and row["data"]: # If a row was found and it has data
            console.print(f"[DB] Match {match_id} found.", style="bright_black")
            return json.loads(row["data"]) # Convert the JSON string back into a Python dictionary
        console.print(f"[DB] Match {match_id} not found.", style="bright_black")
        return None

    def store_match_data(self, match_id: int, match_data: Dict[str, Any]):
        """
        Stores a specific match's full details into the 'matches' table.
        If a match with the same ID already exists, it will be replaced (due to INSERT OR REPLACE).
        Args:
            match_id: The ID of the match.
            match_data: A Python dictionary containing the match's details.
        """
        conn = self._get_connection()
        try:
            # Convert the Python dictionary to a JSON string to store it in the TEXT column.
            match_data_json = json.dumps(match_data)
            # SQL command to insert (or replace) a row into the 'matches' table.
            conn.execute(
                "INSERT OR REPLACE INTO matches (match_id, data, fetched_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (match_id, match_data_json)
            )
            conn.commit() # Save the change
            console.print(f"[DB] Match {match_id} stored.", style="green dim")
        except sqlite3.Error as e: # Catch any SQLite specific errors
            console.print(f"[bold red]SQLite error storing match {match_id}: {e}[/bold red]")
        finally:
            conn.close() # Always close the connection, even if an error occurred

    # --- Hero Data Operations ---
    def get_all_heroes(self) -> Dict[int, str]:
        """
        Retrieves all heroes from the 'heroes' table and returns them as a dictionary
        mapping hero_id to hero_name. This is useful for quick lookups.
        Returns:
            A dictionary like {1: "Anti-Mage", 2: "Axe", ...}.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT hero_id, name FROM heroes")
        rows = cursor.fetchall() # Fetches all rows that match the query
        conn.close()
        hero_map = {row["hero_id"]: row["name"] for row in rows} # Create the dictionary
        if hero_map: console.print(f"[DB] Loaded {len(hero_map)} heroes.", style="bright_black")
        else: console.print("[DB] No heroes found.", style="bright_black")
        return hero_map

    def store_all_heroes(self, heroes_api_data: List[Dict[str, Any]]):
        """
        Stores a list of heroes (typically fetched from the API) into the 'heroes' table.
        It replaces any existing hero data.
        Args:
            heroes_api_data: A list of dictionaries, where each dictionary represents a hero's data from the API.
        """
        conn = self._get_connection()
        # Prepare data for batch insertion: list of tuples (hero_id, hero_name)
        heroes_to_store = []
        for h_api in heroes_api_data:
            if "id" in h_api and "localized_name" in h_api: # Ensure the necessary keys exist
                heroes_to_store.append((h_api["id"], h_api["localized_name"]))
        
        if heroes_to_store:
            try:
                # executemany is efficient for inserting multiple rows.
                # INSERT OR REPLACE updates the row if hero_id already exists, or inserts a new one.
                conn.executemany("INSERT OR REPLACE INTO heroes (hero_id, name, fetched_at) VALUES (?, ?, CURRENT_TIMESTAMP)", heroes_to_store)
                conn.commit()
                console.print(f"[DB] Stored/Updated {len(heroes_to_store)} heroes.", style="green dim")
            except sqlite3.Error as e:
                console.print(f"[bold red]SQLite error storing heroes: {e}[/bold red]")
            finally:
                conn.close()

    def clear_heroes_table(self):
        """Deletes all records from the 'heroes' table. Used for force refresh."""
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM heroes")
            conn.commit()
            console.print("[DB] Cleared 'heroes' table.", style="yellow dim")
        except sqlite3.Error as e:
            console.print(f"[bold red]SQLite error clearing heroes: {e}[/bold red]")
        finally:
            conn.close()

    # --- Team Data Operations --- (Similar structure to Hero operations)
    def get_all_teams(self) -> Dict[int, str]:
        """Retrieves all teams, mapping team_id to team_name."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT team_id, name FROM teams")
        rows = cursor.fetchall()
        conn.close()
        team_map = {row["team_id"]: row["name"] for row in rows}
        if team_map: console.print(f"[DB] Loaded {len(team_map)} teams.", style="bright_black")
        else: console.print("[DB] No teams found.", style="bright_black")
        return team_map

    def store_all_teams(self, teams_api_data: List[Dict[str, Any]]):
        """Stores a list of teams into the 'teams' table."""
        conn = self._get_connection()
        teams_to_store = []
        for team_api in teams_api_data:
            if "team_id" in team_api:
                # Prefer full name, fallback to tag if name is missing or empty
                name = team_api.get("name") or team_api.get("tag")
                if name and name.strip(): # Ensure name is not just whitespace
                    teams_to_store.append((team_api["team_id"], name.strip()))
        if teams_to_store:
            try:
                conn.executemany("INSERT OR REPLACE INTO teams (team_id, name, fetched_at) VALUES (?, ?, CURRENT_TIMESTAMP)", teams_to_store)
                conn.commit()
                console.print(f"[DB] Stored/Updated {len(teams_to_store)} teams.", style="green dim")
            except sqlite3.Error as e:
                console.print(f"[bold red]SQLite error storing teams: {e}[/bold red]")
            finally:
                conn.close()

    def clear_teams_table(self):
        """Deletes all records from the 'teams' table."""
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM teams")
            conn.commit()
            console.print("[DB] Cleared 'teams' table.", style="yellow dim")
        except sqlite3.Error as e:
            console.print(f"[bold red]SQLite error clearing teams: {e}[/bold red]")
        finally:
            conn.close()

    # --- All Leagues Data Operations ---
    def get_all_leagues(self) -> List[Dict[str, Any]]:
        """Retrieves all leagues from the 'all_leagues' table."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT league_id, name, tier FROM all_leagues")
        rows = cursor.fetchall()
        conn.close()
        # Convert SQLite rows to a list of dictionaries with expected keys
        leagues_list = [{"leagueid": r["league_id"], "name": r["name"], "tier": r["tier"]} for r in rows]
        if leagues_list: console.print(f"[DB] Loaded {len(leagues_list)} leagues.", style="bright_black")
        else: console.print("[DB] No 'all_leagues' found.", style="bright_black")
        return leagues_list

    def store_all_leagues(self, leagues_api_data: List[Dict[str, Any]]):
        """Stores a list of leagues into the 'all_leagues' table. Clears old data first."""
        conn = self._get_connection()
        # Prepare data, ensuring essential keys 'leagueid' and 'name' are present
        leagues_to_store = []
        for l_api in leagues_api_data:
            if "leagueid" in l_api and "name" in l_api:
                leagues_to_store.append((
                    l_api["leagueid"],
                    l_api["name"],
                    l_api.get("tier") # Tier can be None/null
                ))
        
        if leagues_to_store:
            try:
                # It's often better to clear old data if this table is meant to be a full snapshot
                conn.execute("DELETE FROM all_leagues") 
                conn.executemany(
                    "INSERT OR REPLACE INTO all_leagues (league_id, name, tier, fetched_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                    leagues_to_store
                )
                conn.commit()
                console.print(f"[DB] Stored/Updated {len(leagues_to_store)} in 'all_leagues'.", style="green dim")
            except sqlite3.Error as e:
                console.print(f"[bold red]SQLite error storing all_leagues: {e}[/bold red]")
            finally:
                conn.close()
    
    def clear_all_leagues_table(self):
        """Deletes all records from the 'all_leagues' table."""
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM all_leagues")
            conn.commit()
            console.print("[DB] Cleared 'all_leagues' table.", style="yellow dim")
        except sqlite3.Error as e:
            console.print(f"[bold red]SQLite error clearing all_leagues: {e}[/bold red]")
        finally:
            conn.close()


# --- API Client Class ---
# This class is responsible for making requests to the external OpenDota API.
# It centralizes all API call logic.
class OpenDotaAPIClient:
    BASE_URL = "https://api.opendota.com/api" # The common part of all OpenDota API URLs

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        A generic helper method to make a GET request to a specific API endpoint.
        Args:
            endpoint: The specific part of the URL for the desired data (e.g., "heroStats").
            params: Optional dictionary of query parameters for the request.
        Returns:
            The JSON response from the API as a Python dictionary/list, or None if an error occurs.
        """
        url = f"{self.BASE_URL}/{endpoint}" # Construct the full URL
        try:
            response = requests.get(url, params=params)
            # This will raise an HTTPError if the HTTP request returned an unsuccessful status code (4xx or 5xx).
            response.raise_for_status() 
            return response.json() # Parse the JSON response
        except requests.exceptions.HTTPError as http_err:
            console.print(f"[API ERR] HTTP error for {url}: {http_err}", style="bold red")
        except requests.exceptions.RequestException as req_err: # Other request errors (network, timeout)
            console.print(f"[API ERR] Request error for {url}: {req_err}", style="bold red")
        except json.JSONDecodeError: # If the response isn't valid JSON
            console.print(f"[API ERR] JSON decode error for {url}", style="bold red")
        return None # Return None if any error occurred

    def fetch_hero_stats(self) -> Optional[List[Dict[str, Any]]]:
        """Fetches general statistics for all heroes."""
        console.print("[API] Fetching hero stats...", style="yellow")
        return self._request("heroStats")

    def fetch_all_teams(self) -> Optional[List[Dict[str, Any]]]:
        """Fetches a list of all professional Dota 2 teams."""
        console.print("[API] Fetching all teams...", style="yellow")
        return self._request("teams")

    def fetch_all_leagues(self) -> Optional[List[Dict[str, Any]]]:
        """Fetches a list of all Dota 2 leagues/tournaments."""
        console.print("[API] Fetching all leagues...", style="yellow")
        return self._request("leagues")

    def fetch_match_details(self, match_id: int) -> Optional[Dict[str, Any]]:
        """Fetches detailed information for a specific match ID."""
        console.print(f"[API] Fetching match details for {match_id}...", style="yellow")
        return self._request(f"matches/{match_id}")
    
    def fetch_league_matches_summary(self, league_id: int) -> Optional[List[Dict[str, Any]]]:
        """Fetches a summary list of matches for a specific league ID."""
        console.print(f"[API] Fetching match summaries for league {league_id}...", style="yellow")
        return self._request(f"leagues/{league_id}/matches")


# --- Data Service Class (Orchestrator) ---
# This class acts as a central point for data retrieval.
# It decides whether to get data from the database (cache) or fetch it from the API.
# It also manages in-memory caches for frequently accessed data like hero/team maps.
class DataService:
    def __init__(self, db_manager: DBManager, api_client: OpenDotaAPIClient):
        """
        Constructor for DataService.
        Args:
            db_manager: An instance of DBManager for database operations.
            api_client: An instance of OpenDotaAPIClient for API calls.
        """
        self.db_manager = db_manager
        self.api_client = api_client
        # In-memory caches to avoid hitting the DB repeatedly for the same static data within a single script run.
        self._hero_map_cache: Optional[Dict[int, str]] = None
        self._team_map_cache: Optional[Dict[int, str]] = None
        self._all_leagues_list_cache: Optional[List[Dict[str, Any]]] = None

    def get_hero_map(self, force_refresh: bool = False) -> Dict[int, str]:
        """
        Gets the hero ID to hero name mapping.
        Strategy:
        1. If force_refresh: Clear memory cache and DB table.
        2. Try memory cache.
        3. Try DB cache (and populate memory cache).
        4. Fetch from API (store in DB, populate memory cache).
        Args:
            force_refresh: If True, bypasses all caches and fetches fresh data from API.
        Returns:
            A dictionary mapping hero_id to hero_name.
        """
        if force_refresh:
            console.print("[CACHE] Force refresh hero map.", style="yellow dim")
            self._hero_map_cache = None # Clear memory cache
            self.db_manager.clear_heroes_table() # Clear DB table
        
        if not force_refresh and self._hero_map_cache is not None:
            console.print("[CACHE] Using memory-cached hero map.", style="cyan dim")
            return self._hero_map_cache
        
        if not force_refresh: # Try DB if not forcing refresh and memory cache miss
            db_hero_map = self.db_manager.get_all_heroes()
            if db_hero_map: # If DB had data
                self._hero_map_cache = db_hero_map # Load into memory cache
                return self._hero_map_cache
        
        # If caches were empty or force_refresh was true, fetch from API
        api_data_list = self.api_client.fetch_hero_stats()
        if api_data_list:
            self.db_manager.store_all_heroes(api_data_list) # Store fresh API data in DB
            # Rebuild map from the (potentially filtered) data stored in DB, or directly from API list
            self._hero_map_cache = {h["id"]: h["localized_name"] for h in api_data_list if "id" in h and "localized_name" in h}
            return self._hero_map_cache
        
        console.print("[SERVICE WARN] Hero map could not be populated from API or DB.", style="yellow")
        return {} # Return empty map if all sources fail

    def get_team_map(self, force_refresh: bool = False) -> Dict[int, str]:
        """Gets the team ID to team name mapping, using similar caching strategy as get_hero_map."""
        if force_refresh:
            console.print("[CACHE] Force refresh team map.", style="yellow dim")
            self._team_map_cache = None
            self.db_manager.clear_teams_table()

        if not force_refresh and self._team_map_cache is not None:
            console.print("[CACHE] Using memory-cached team map.", style="cyan dim")
            return self._team_map_cache
        
        if not force_refresh:
            db_team_map = self.db_manager.get_all_teams()
            if db_team_map:
                self._team_map_cache = db_team_map
                return self._team_map_cache
        
        api_data_list = self.api_client.fetch_all_teams()
        if api_data_list:
            self.db_manager.store_all_teams(api_data_list)
            self._team_map_cache = {t["team_id"]: (t.get("name") or t.get("tag", "")).strip() 
                                   for t in api_data_list if "team_id" in t and (t.get("name") or t.get("tag"))}
            return self._team_map_cache
        
        console.print("[SERVICE WARN] Team map could not be populated.", style="yellow")
        return {}

    def get_all_leagues_list(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Gets the list of all leagues, using similar caching strategy."""
        if force_refresh:
            console.print("[CACHE] Force refresh all leagues list.", style="yellow dim")
            self._all_leagues_list_cache = None
            self.db_manager.clear_all_leagues_table()

        if not force_refresh and self._all_leagues_list_cache is not None:
            console.print("[CACHE] Using memory-cached all leagues list.", style="cyan dim")
            return self._all_leagues_list_cache
        
        if not force_refresh:
            db_leagues_list = self.db_manager.get_all_leagues() # This returns list of dicts
            if db_leagues_list:
                self._all_leagues_list_cache = db_leagues_list
                return self._all_leagues_list_cache
        
        api_data_list = self.api_client.fetch_all_leagues() # This is a list of dicts from API
        if api_data_list:
            self.db_manager.store_all_leagues(api_data_list)
            # Ensure the format stored in cache matches what get_all_leagues_from_db returns
            self._all_leagues_list_cache = [{"leagueid": l["leagueid"], "name": l["name"], "tier": l.get("tier")} 
                                           for l in api_data_list if "leagueid" in l and "name" in l]
            return self._all_leagues_list_cache
        
        console.print("[SERVICE WARN] All leagues list could not be populated.", style="yellow")
        return []

    def get_match_details(self, match_id: int) -> Optional[Dict[str, Any]]:
        """
        Gets full details for a specific match.
        Strategy:
        1. Try DB cache.
        2. Fetch from API (and store in DB).
        """
        db_data = self.db_manager.get_match_data(match_id)
        if db_data:
            return db_data # Return data from DB if found
        
        # If not in DB, fetch from API
        api_data = self.api_client.fetch_match_details(match_id)
        if api_data:
            self.db_manager.store_match_data(match_id, api_data) # Store in DB for next time
            return api_data
        
        console.print(f"[SERVICE ERR] Failed to get match {match_id} from DB or API.", style="bold red")
        return None # Return None if fetching from API also failed

# --- Display Functions ---
# These functions are responsible for presenting data to the user in a readable format,
# typically using Rich tables or panels. They don't fetch data themselves but receive it.

def display_draft_info(match_data: Dict[str, Any], hero_map: Dict[int, str]):
    """
    Displays the pick and ban sequence for a match in a table.
    Args:
        match_data: The full dictionary of match details.
        hero_map: A mapping of hero IDs to hero names.
    """
    if not hero_map: # If hero_map is empty or None
        console.print("[yellow]Warning: Hero map not available. Hero names will be displayed as IDs.[/yellow]")
        hero_map = {} # Ensure hero_map is a dict to prevent errors in .get()

    picks_bans = match_data.get("picks_bans") # Get the draft sequence from match data
    if not picks_bans: # If no draft info (e.g., not Captains Mode, or error)
        console.print(Panel("[yellow]No pick/ban information available for this match.[/yellow]\nThis might be because the match was not a Captains Mode game, the replay is not fully parsed, or it's an older match format.", title="Draft Info"))
        return

    # Create a Rich Table for displaying the draft
    table = Table(title=f"Draft Information for Match ID: {match_data.get('match_id', 'N/A')}", show_header=True, header_style="bold magenta")
    table.add_column("Order", style="dim", width=6, justify="center") # Draft step number
    table.add_column("Team", width=10, justify="center")              # Radiant or Dire
    table.add_column("Action", width=8, justify="center")             # Pick or Ban
    table.add_column("Hero", justify="left")                          # Hero Name
    table.add_column("Hero ID", style="dim", width=10, justify="center") # Hero's numerical ID

    # Populate the table row by row
    for action in picks_bans:
        action_type = "[bold green]Pick[/bold green]" if action.get("is_pick") else "[bold red]Ban[/bold red]"
        team_id = action.get("team") # 0 for Radiant, 1 for Dire
        team_style = "cyan" if team_id == 0 else "orange3"
        team_name_draft = "Radiant" if team_id == 0 else "Dire" if team_id == 1 else "Unknown"
        
        hero_id = action.get("hero_id")
        hero_name_str = hero_map.get(hero_id, f"Hero ID: {hero_id}") # Look up hero name, fallback to ID
        
        original_order = action.get("order") # API order is usually 0-indexed
        # Display order as 1-indexed for readability
        display_order_str = str(original_order + 1) if isinstance(original_order, int) else str(original_order or "")
        
        table.add_row(display_order_str, Text(team_name_draft, style=team_style), action_type, hero_name_str, str(hero_id))
    
    console.print(table) # Print the complete table


# --- Typer CLI Commands ---
# These functions define the commands a user can run from their terminal.
# They use the DataService to get data and then call display functions.

# Create global instances of our service classes.
# These will be used by the command functions.
db_manager_instance = DBManager()
api_client_instance = OpenDotaAPIClient()
data_service_instance = DataService(db_manager_instance, api_client_instance)


@app.command(name="search-leagues", help="Search for leagues/tournaments by name (uses DB cache).")
def find_leagues_command(
    search_term: str = typer.Argument(..., help="The term to search for in league names."),
    force_refresh: bool = typer.Option(False, "--force-refresh", "-r", help="Force refresh league list from API.")
):
    """
    Command to search for leagues.
    It gets a list of all leagues (from cache or API via DataService) and filters by the search term.
    """
    all_leagues = data_service_instance.get_all_leagues_list(force_refresh=force_refresh)
    if not all_leagues:
        console.print("[yellow]No league data available (failed to fetch from API and DB is empty).[/yellow]")
        raise typer.Exit() # Exit the command if no data

    matching_leagues_data = []
    # Filter the full list of leagues based on the user's search term
    for league in all_leagues: # league is a dict like {"leagueid": ..., "name": ..., "tier": ...}
        if "name" in league and league["name"] and "leagueid" in league: # Basic validation
            if search_term.lower() in league["name"].lower(): # Case-insensitive search
                matching_leagues_data.append(league) 
    
    if not matching_leagues_data:
        console.print(f"[yellow]No leagues found matching '{search_term}' in the cached/fetched list.[/yellow]")
        raise typer.Exit()

    console.print(f"\nFound {len(matching_leagues_data)} league(s) matching '[bold]{search_term}[/bold]':")
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("#", style="dim", width=5, justify="center") # User-friendly index for the displayed results
    table.add_column("League Name", min_width=30)
    table.add_column("League ID", style="magenta", justify="center")
    table.add_column("Tier", style="cyan", justify="center")
    # Display the filtered list with a new 1-based index
    for idx, league_info in enumerate(matching_leagues_data):
        table.add_row(str(idx + 1), league_info["name"], str(league_info["leagueid"]), league_info.get("tier", "N/A"))
    console.print(table)
    console.print("\nUse 'get-league-matches' or 'populate-league-matches' with a League ID.")


@app.command(name="get-league-matches", help="List matches for a league ID, resolving names.")
def get_league_matches_command(league_id: int = typer.Argument(..., help="The ID of the league.")):
    """
    Command to list matches for a specific league.
    It fetches match summaries from the API (as these are not cached per league yet)
    and uses the cached team map to display team names.
    """
    team_map = data_service_instance.get_team_map() # Get team names (from cache or API)
    if not team_map: team_map = {} # Ensure it's a dict for .get()
    
    # Fetch summary of matches in the league directly from API client
    # Note: Caching for this specific list of league match summaries is not yet implemented.
    # This means this command will always hit the API for the league's match list.
    matches_in_league = api_client_instance.fetch_league_matches_summary(league_id)
    if not matches_in_league:
        console.print(f"[yellow]Could not fetch match summaries for league {league_id}.[/yellow]")
        raise typer.Exit()

    league_matches_info = []
    for match_summary in matches_in_league:
        if "match_id" in match_summary:
            # Resolve Radiant team name
            radiant_name = match_summary.get('radiant_name')
            if not radiant_name and isinstance(match_summary.get('radiant_team'), dict): # Check nested object
                radiant_name = match_summary['radiant_team'].get('name') or match_summary['radiant_team'].get('tag')
            radiant_team_id = match_summary.get('radiant_team_id')
            if (not radiant_name or radiant_name == "Unknown Radiant") and radiant_team_id: # Fallback to team_map if ID exists
                radiant_name = team_map.get(radiant_team_id, f"ID: {radiant_team_id}")
            radiant_name = radiant_name or "Unknown Radiant" # Final fallback

            # Resolve Dire team name (similar logic)
            dire_name = match_summary.get('dire_name')
            if not dire_name and isinstance(match_summary.get('dire_team'), dict):
                dire_name = match_summary['dire_team'].get('name') or match_summary['dire_team'].get('tag')
            dire_team_id = match_summary.get('dire_team_id')
            if (not dire_name or dire_name == "Unknown Dire") and dire_team_id:
                dire_name = team_map.get(dire_team_id, f"ID: {dire_team_id}")
            dire_name = dire_name or "Unknown Dire"
            
            winner = "N/A" # Determine winner
            if match_summary.get("radiant_win") is True: winner = f"[bold cyan]{radiant_name}[/bold cyan]"
            elif match_summary.get("radiant_win") is False: winner = f"[bold orange3]{dire_name}[/bold orange3]"
            
            start_time_unix = match_summary.get("start_time") # Format start time
            start_time_str = datetime.fromtimestamp(start_time_unix).strftime('%Y-%m-%d %H:%M') if start_time_unix else "N/A"
            
            league_matches_info.append({
                "match_id": str(match_summary["match_id"]), "radiant_name": radiant_name, 
                "dire_name": dire_name, "radiant_score": str(match_summary.get("radiant_score", "-")), 
                "dire_score": str(match_summary.get("dire_score", "-")), "winner": winner, 
                "start_time": start_time_str
            })
    
    if not league_matches_info:
        console.print(f"[yellow]No matches processed for league ID {league_id}.[/yellow]")
        raise typer.Exit()
    
    league_matches_info.sort(key=lambda x: x.get('start_time', "0"), reverse=True) # Sort by start time
    
    # Display matches in a table
    console.print(f"\nFound {len(league_matches_info)} match(es) for league ID [bold magenta]{league_id}[/bold magenta]:")
    table = Table(show_header=True, header_style="bold blue", title=f"Matches for League ID {league_id}")
    table.add_column("Match ID", style="magenta", justify="center")
    table.add_column("Radiant Team", style="cyan", min_width=20)
    table.add_column("Dire Team", style="orange3", min_width=20)
    table.add_column("Score (R-D)", justify="center")
    table.add_column("Winner", min_width=20, justify="center")
    table.add_column("Start Time (UTC)", style="dim", justify="center")
    for match_info_row in league_matches_info:
        table.add_row(
            match_info_row["match_id"], match_info_row["radiant_name"], match_info_row["dire_name"], 
            f"{match_info_row['radiant_score']}-{match_info_row['dire_score']}", 
            match_info_row["winner"], match_info_row["start_time"]
        )
    console.print(table)


@app.command(name="get-draft-details", help="Fetch and display draft details for a specific match ID.")
def get_draft_details_command(match_id: int = typer.Argument(..., help="The Match ID.")):
    """Command to get and display draft details for one match, using DB cache for match data."""
    hero_map = data_service_instance.get_hero_map() # Get hero names
    match_details_data = data_service_instance.get_match_details(match_id) # Get full match data (from DB or API)
    
    if match_details_data:
        # Extract team names for the summary panel (similar logic to get_league_matches)
        r_team = (match_details_data.get('radiant_name') or 
                  (isinstance(match_details_data.get('radiant_team'), dict) and 
                   (match_details_data['radiant_team'].get('name') or match_details_data['radiant_team'].get('tag'))) or 
                  "Radiant")
        d_team = (match_details_data.get('dire_name') or 
                  (isinstance(match_details_data.get('dire_team'), dict) and 
                   (match_details_data['dire_team'].get('name') or match_details_data['dire_team'].get('tag'))) or 
                  "Dire")
        r_score = match_details_data.get('radiant_score', '-')
        d_score = match_details_data.get('dire_score', '-')
        winner_text = ""
        if match_details_data.get('radiant_win') is True: winner_text = f"Winner: [bold cyan]{r_team}[/bold cyan]"
        elif match_details_data.get('radiant_win') is False: winner_text = f"Winner: [bold orange3]{d_team}[/bold orange3]"
        
        match_title = f"Match: [cyan]{r_team}[/cyan] vs [orange3]{d_team}[/orange3] ({r_score}-{d_score}) {winner_text}"
        console.print(Panel(match_title, title=f"Details for Match ID {match_id}", expand=False))
        
        display_draft_info(match_details_data, hero_map if hero_map else {}) # Display the draft table
    else:
        console.print(f"[yellow]Could not retrieve details for match ID {match_id}.[/yellow]")


@app.command(name="populate-league-matches", help="Fetch and store full details for all matches in a league into the DB.")
def populate_league_matches_command(league_id: int = typer.Argument(..., help="The ID of the league to populate.")):
    """
    Command to iterate through all matches in a league and ensure their full details
    are stored in the local database. Uses a progress bar.
    """
    console.print(f"\nPopulating full match details for league ID [bold magenta]{league_id}[/bold magenta].")
    
    # Step 1: Fetch the list of match IDs for the league
    matches_summary = api_client_instance.fetch_league_matches_summary(league_id)
    if not matches_summary:
        console.print(f"[yellow]No match summaries found for league {league_id} to populate.[/yellow]")
        raise typer.Exit()

    match_ids_to_process = [m["match_id"] for m in matches_summary if "match_id" in m]
    if not match_ids_to_process:
        console.print(f"[yellow]No match IDs extracted for league {league_id}.[/yellow]")
        raise typer.Exit()
    console.print(f"Found {len(match_ids_to_process)} matches in league {league_id} to process.")

    # Step 2: Process each match with a progress bar
    # Initialize counters for the summary report
    processed_count = 0
    newly_fetched_and_stored = 0
    already_in_db_count = 0
    failed_to_fetch_count = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(), # Estimates time remaining
        console=console,
    ) as progress:
        task = progress.add_task(f"Populating league {league_id}...", total=len(match_ids_to_process))
        for match_id in match_ids_to_process:
            progress.update(task, advance=1, description=f"Match ID: {match_id}")
            
            # Check DB first to correctly update counters before calling the service method
            # that might fetch from API.
            if db_manager_instance.get_match_data(match_id): # This prints "Match X found"
                already_in_db_count += 1
            else: # Not in DB, so DataService will attempt API fetch
                match_detail = data_service_instance.get_match_details(match_id) # This will fetch and store
                if match_detail:
                    newly_fetched_and_stored += 1
                else:
                    failed_to_fetch_count += 1
            processed_count +=1
    
    console.print(f"\n[bold green]League {league_id} DB population attempt complete.[/bold green]")
    console.print(f"Total matches processed: {processed_count}")
    console.print(f"Matches already in DB: {already_in_db_count}")
    console.print(f"New matches fetched and stored: {newly_fetched_and_stored}")
    console.print(f"Matches failed to fetch/store: {failed_to_fetch_count}")


@app.command(name="refresh-static-data", help="Force refresh of hero, team, and all-leagues data from API into the DB.")
def refresh_static_data_command():
    """Command to force a refresh of all cached static data (heroes, teams, all_leagues list)."""
    console.print("[bold yellow]Forcing refresh of static hero, team, and all-leagues data...[/bold yellow]")
    data_service_instance.get_hero_map(force_refresh=True)
    data_service_instance.get_team_map(force_refresh=True)
    data_service_instance.get_all_leagues_list(force_refresh=True) # This will clear and repopulate the all_leagues table
    console.print("[bold green]Static data refresh complete.[/bold green]")


# This is the main entry point when the script is run.
if __name__ == "__main__":
    # The DBManager constructor calls _init_tables(), so DB is ready.
    console.print(Panel("[bold green]OpenDota Match Data Fetcher CLI (league_info.py)[/bold green]", subtitle="Class-based, SQLite Caching", expand=False))
    app() # Start the Typer CLI application

