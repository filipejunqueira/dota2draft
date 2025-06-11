# dota2draft/db.py

import sqlite3
import json
from datetime import datetime, timezone, timezone
from typing import List, Optional, Dict, Any
from .config_loader import CONFIG
from .logger_config import logger

class DBManager:
    def __init__(self, db_name: Optional[str] = None):
        """
        Initializes the DBManager, creating and holding a single database connection.
        """
        self.db_name = db_name if db_name else CONFIG["database_path"]
        self.conn = self._create_connection()
        self._init_tables()

    def _create_connection(self) -> sqlite3.Connection:
        """Establishes the database connection and sets the row factory."""
        try:
            conn = sqlite3.connect(self.db_name)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error(f"SQLite connection error to {self.db_name}: {e}")
            raise

    def close(self):
        """Closes the database connection if it is open."""
        if self.conn:
            self.conn.close()
            logger.info(f"Database connection to '{self.db_name}' closed.")

    def _init_tables(self):
        """Creates all necessary tables if they don't already exist."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY,
                data TEXT NOT NULL,
                fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS heroes (
                hero_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS all_leagues (
                league_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                tier TEXT,
                fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""")
            self.conn.commit()
            logger.info(f"Database '{self.db_name}' initialized/checked.")
        except sqlite3.Error as e:
            logger.error(f"SQLite error during table initialization: {e}")

    def get_match_data(self, match_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a specific match's full details from the database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT data FROM matches WHERE match_id = ?", (match_id,))
            row = cursor.fetchone()
            if row and row["data"]:
                logger.debug(f"[DB] Match {match_id} found.")
                return json.loads(row["data"])
            logger.debug(f"[DB] Match {match_id} not found.")
            return None
        except sqlite3.Error as e:
            logger.error(f"SQLite error getting match {match_id}: {e}")
            return None

    def store_match_data(self, match_id: int, match_data: Dict[str, Any]):
        """Stores a specific match's full details into the database."""
        try:
            match_data_json = json.dumps(match_data)
            self.conn.execute(
                "INSERT OR REPLACE INTO matches (match_id, data, fetched_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (match_id, match_data_json)
            )
            self.conn.commit()
            logger.info(f"[DB] Match {match_id} stored.")
        except sqlite3.Error as e:
            logger.error(f"SQLite error storing match {match_id}: {e}")

    def get_all_heroes(self) -> Dict[int, str]:
        """Retrieves all heroes, mapping hero_id to hero_name."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT hero_id, name FROM heroes")
            rows = cursor.fetchall()
            hero_map = {row["hero_id"]: row["name"] for row in rows}
            logger.debug(f"[DB] Loaded {len(hero_map)} heroes." if hero_map else "[DB] No heroes found.")
            return hero_map
        except sqlite3.Error as e:
            logger.error(f"SQLite error getting all heroes: {e}")
            return {}

    def store_all_heroes(self, heroes_api_data: List[Dict[str, Any]]):
        """Stores a list of heroes into the database."""
        try:
            heroes_to_store = []
            for h_api in heroes_api_data:
                if "id" in h_api and "localized_name" in h_api:
                    heroes_to_store.append((h_api["id"], h_api["localized_name"]))
            if heroes_to_store:
                self.conn.executemany("INSERT OR REPLACE INTO heroes (hero_id, name, fetched_at) VALUES (?, ?, CURRENT_TIMESTAMP)", heroes_to_store)
                self.conn.commit()
                logger.info(f"[DB] Stored/Updated {len(heroes_to_store)} heroes.")
        except sqlite3.Error as e:
            logger.error(f"SQLite error storing heroes: {e}")

    def clear_heroes_table(self):
        """Deletes all records from the 'heroes' table."""
        try:
            self.conn.execute("DELETE FROM heroes")
            self.conn.commit()
            logger.info("[DB] Cleared 'heroes' table.")
        except sqlite3.Error as e:
            logger.error(f"SQLite error clearing heroes: {e}")

    def get_all_teams(self) -> Dict[int, str]:
        """Retrieves all teams, mapping team_id to team_name."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT team_id, name FROM teams")
            rows = cursor.fetchall()
            team_map = {row["team_id"]: row["name"] for row in rows}
            logger.debug(f"[DB] Loaded {len(team_map)} teams." if team_map else "[DB] No teams found.")
            return team_map
        except sqlite3.Error as e:
            logger.error(f"SQLite error getting all teams: {e}")
            return {}

    def store_all_teams(self, teams_api_data: List[Dict[str, Any]]):
        """Stores a list of teams into the database."""
        try:
            teams_to_store = []
            for team_api in teams_api_data:
                if "team_id" in team_api:
                    name = team_api.get("name") or team_api.get("tag")
                    if name and name.strip():
                        teams_to_store.append((team_api["team_id"], name.strip()))
            if teams_to_store:
                self.conn.executemany("INSERT OR REPLACE INTO teams (team_id, name, fetched_at) VALUES (?, ?, CURRENT_TIMESTAMP)", teams_to_store)
                self.conn.commit()
                logger.info(f"[DB] Stored/Updated {len(teams_to_store)} teams.")
        except sqlite3.Error as e:
            logger.error(f"SQLite error storing teams: {e}")

    def clear_teams_table(self):
        """Deletes all records from the 'teams' table."""
        try:
            self.conn.execute("DELETE FROM teams")
            self.conn.commit()
            logger.info("[DB] Cleared 'teams' table.")
        except sqlite3.Error as e:
            logger.error(f"SQLite error clearing teams: {e}")

    def get_all_leagues(self) -> List[Dict[str, Any]]:
        """Retrieves all leagues from the database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT league_id, name, tier FROM all_leagues")
            rows = cursor.fetchall()
            leagues_list = [{"leagueid": r["league_id"], "name": r["name"], "tier": r["tier"]} for r in rows]
            logger.debug(f"[DB] Loaded {len(leagues_list)} leagues." if leagues_list else "[DB] No 'all_leagues' found.")
            return leagues_list
        except sqlite3.Error as e:
            logger.error(f"SQLite error getting all leagues: {e}")
            return []

    def store_all_leagues(self, leagues_api_data: List[Dict[str, Any]]):
        """Stores a list of leagues. Clears old data first."""
        try:
            leagues_to_store = []
            for l_api in leagues_api_data:
                if "leagueid" in l_api and "name" in l_api:
                    leagues_to_store.append((
                        l_api["leagueid"],
                        l_api["name"],
                        l_api.get("tier")
                    ))
            if leagues_to_store:
                self.conn.execute("DELETE FROM all_leagues") 
                self.conn.executemany(
                    "INSERT OR REPLACE INTO all_leagues (league_id, name, tier, fetched_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                    leagues_to_store
                )
                self.conn.commit()
                logger.info(f"[DB] Stored/Updated {len(leagues_to_store)} in 'all_leagues'.")
        except sqlite3.Error as e:
            logger.error(f"SQLite error storing all_leagues: {e}")
    
    def get_leagues_by_name(self, name_keyword: str) -> List[Dict[str, Any]]:
        """Retrieves leagues where the name matches a keyword."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT league_id, name, tier FROM all_leagues WHERE name LIKE ?", (f'%{name_keyword}%',))
            rows = cursor.fetchall()
            leagues_list = [{"leagueid": r["league_id"], "name": r["name"], "tier": r["tier"]} for r in rows]
            logger.debug(f"[DB] Found {len(leagues_list)} leagues matching '{name_keyword}'.")
            return leagues_list
        except sqlite3.Error as e:
            logger.error(f"SQLite error searching for leagues with keyword '{name_keyword}': {e}")
            return []

    def get_hero_stats(self, league_id: Optional[int] = None, after_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Calculates hero statistics from stored match data."""
        hero_stats = {}
        total_matches = 0
        start_timestamp = 0
        if after_date:
            try:
                # Make the datetime object timezone-aware (UTC) before creating the timestamp
                dt = datetime.strptime(after_date, "%Y-%m-%d")
                start_timestamp = int(dt.replace(tzinfo=timezone.utc).timestamp())
            except ValueError:
                logger.error(f"Invalid date format for after_date: '{after_date}'. Please use YYYY-MM-DD.")
                return []
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT data FROM matches")
            all_matches_raw = cursor.fetchall()

            all_heroes = self.get_all_heroes()

            for row in all_matches_raw:
                match_data = json.loads(row['data'])

                if league_id and match_data.get("leagueid") != league_id:
                    continue

                if start_timestamp and match_data.get('start_time', 0) < start_timestamp:
                    continue
                
                total_matches += 1

                if not match_data.get("picks_bans"):
                    continue

                radiant_win = match_data.get("radiant_win", False)

                for pick_ban in match_data["picks_bans"]:
                    hero_id = pick_ban.get("hero_id")
                    if not hero_id:
                        continue

                    if hero_id not in hero_stats:
                        hero_stats[hero_id] = {"picks": 0, "bans": 0, "wins": 0}

                    if pick_ban.get("is_pick"):
                        hero_stats[hero_id]["picks"] += 1
                        is_radiant_pick = pick_ban["team"] == 0
                        if (radiant_win and is_radiant_pick) or (not radiant_win and not is_radiant_pick):
                            hero_stats[hero_id]["wins"] += 1
                    else:
                        hero_stats[hero_id]["bans"] += 1

            results = []
            for hero_id, stats in hero_stats.items():
                results.append({
                    "hero_id": hero_id,
                    "hero_name": all_heroes.get(hero_id, "Unknown Hero"),
                    "picks": stats["picks"],
                    "bans": stats["bans"],
                    "wins": stats["wins"],
                    "total_matches": total_matches
                })
            return results

        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.error(f"Error getting hero stats: {e}")
            return []

    def get_player_stats(self, league_id: Optional[int] = None, after_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Calculates player statistics from stored match data."""
        player_stats = {}
        start_timestamp = 0
        if after_date:
            try:
                # Make the datetime object timezone-aware (UTC) before creating the timestamp
                dt = datetime.strptime(after_date, "%Y-%m-%d")
                start_timestamp = int(dt.replace(tzinfo=timezone.utc).timestamp())
            except ValueError:
                logger.error(f"Invalid date format for after_date: '{after_date}'. Please use YYYY-MM-DD.")
                return []
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT data FROM matches")
            all_matches_raw = cursor.fetchall()

            for row in all_matches_raw:
                match_data = json.loads(row['data'])

                if league_id and match_data.get("leagueid") != league_id:
                    continue

                if start_timestamp and match_data.get('start_time', 0) < start_timestamp:
                    continue

                radiant_win = match_data.get("radiant_win", False)

                for player_data in match_data.get("players", []):
                    account_id = player_data.get("account_id")
                    if not account_id:
                        continue

                    if account_id not in player_stats:
                        player_stats[account_id] = {
                            "name": player_data.get("personaname", "Unknown Player"),
                            "matches_played": 0,
                            "wins": 0
                        }
                    
                    player_stats[account_id]["matches_played"] += 1
                    is_radiant_player = player_data.get("isRadiant", False)
                    if (radiant_win and is_radiant_player) or (not radiant_win and not is_radiant_player):
                        player_stats[account_id]["wins"] += 1

            results = [
                {"account_id": acc_id, **stats} for acc_id, stats in player_stats.items()
            ]
            return results

        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.error(f"Error getting player stats: {e}")
            return []

    def clear_all_leagues_table(self):
        """Deletes all records from the 'all_leagues' table."""
        try:
            self.conn.execute("DELETE FROM all_leagues")
            self.conn.commit()
            logger.info("[DB] Cleared 'all_leagues' table.")
        except sqlite3.Error as e:
            logger.error(f"SQLite error clearing all_leagues: {e}")

    def get_all_matches_from_league(self, league_id: int) -> List[Dict[str, Any]]:
        """Retrieves all stored matches that belong to a specific league."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT data FROM matches")
            all_matches_raw = cursor.fetchall()
            league_matches = []
            for row in all_matches_raw:
                match_data = json.loads(row['data'])
                if match_data.get('leagueid') == league_id:
                    league_matches.append(match_data)
            logger.debug(f"[DB] Found {len(league_matches)} stored matches for league {league_id}.")
            return league_matches
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.error(f"Error getting matches for league {league_id}: {e}")
            return []
