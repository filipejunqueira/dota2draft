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
            cursor.execute("DROP TABLE IF EXISTS player_nicknames;")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_nicknames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL,
                nickname TEXT NOT NULL UNIQUE COLLATE NOCASE
            )""")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_nicknames_account_id ON player_nicknames (account_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_nicknames_nickname ON player_nicknames (nickname);")

            # Hero Nicknames Table
            # cursor.execute("DROP TABLE IF EXISTS hero_nicknames;") # Use only if schema changes require a reset
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS hero_nicknames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hero_id INTEGER NOT NULL,
                nickname TEXT NOT NULL UNIQUE COLLATE NOCASE,
                FOREIGN KEY (hero_id) REFERENCES heroes (hero_id)
            )""")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hero_nicknames_hero_id ON hero_nicknames (hero_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hero_nicknames_nickname ON hero_nicknames (nickname);")
            
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
                nicknames = self.get_hero_nicknames(hero_id)
                results.append({
                    "hero_id": hero_id,
                    "hero_name": all_heroes.get(hero_id, "Unknown Hero"),
                    "nicknames": nicknames,
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

    def set_player_nickname(self, account_id: int, nickname: str) -> bool:
        """Assigns a nickname to a player. Nicknames are unique (case-insensitive)."""
        try:
            self.conn.execute(
                "INSERT INTO player_nicknames (account_id, nickname) VALUES (?, ?)",
                (account_id, nickname)
            )
            self.conn.commit()
            logger.info(f"[DB] Assigned nickname '{nickname}' to account ID {account_id}.")
            return True
        except sqlite3.IntegrityError:
            # This can happen if the nickname is already taken (UNIQUE constraint with COLLATE NOCASE)
            # or if trying to add a duplicate nickname for the same player (though the former is more likely here)
            logger.warning(f"[DB] Failed to assign nickname '{nickname}' to account ID {account_id}. Nickname likely already in use or duplicate.")
            # Check if this player already has this nickname
            existing_nicknames = self.get_player_nicknames(account_id)
            if nickname.lower() in [n.lower() for n in existing_nicknames]:
                logger.info(f"[DB] Account ID {account_id} already has nickname '{nickname}'. No action taken.")
                return True # Considered success as the state is already achieved
            return False
        except sqlite3.Error as e:
            logger.error(f"SQLite error assigning nickname '{nickname}' for account ID {account_id}: {e}")
            return False

    def get_account_id_by_nickname(self, nickname: str) -> Optional[int]:
        """Finds a player's account ID by their nickname (case-insensitive)."""
        try:
            cursor = self.conn.cursor()
            # The COLLATE NOCASE on the table schema handles case-insensitivity
            cursor.execute("SELECT account_id FROM player_nicknames WHERE nickname = ?", (nickname,))
            row = cursor.fetchone()
            if row:
                logger.debug(f"[DB] Found account ID {row['account_id']} for nickname '{nickname}'.")
                return row['account_id']
            logger.debug(f"[DB] No account ID found for nickname '{nickname}'.")
            return None
        except sqlite3.Error as e:
            logger.error(f"SQLite error getting account ID for nickname '{nickname}': {e}")
            return None

    def get_player_nicknames(self, account_id: int) -> List[str]:
        """Retrieves all fixed nicknames for a player."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT nickname FROM player_nicknames WHERE account_id = ? ORDER BY nickname", (account_id,))
            rows = cursor.fetchall()
            nicknames = [row['nickname'] for row in rows]
            logger.debug(f"[DB] Found nicknames {nicknames} for account ID {account_id}." if nicknames else f"[DB] No nicknames found for account ID {account_id}.")
            return nicknames
        except sqlite3.Error as e:
            logger.error(f"SQLite error getting nicknames for account ID {account_id}: {e}")
            return []

    def remove_player_nickname(self, account_id: int, nickname: str) -> bool:
        """Removes a specific nickname from a player. Case-insensitive."""
        try:
            cursor = self.conn.cursor()
            # COLLATE NOCASE makes the comparison case-insensitive
            cursor.execute(
                "DELETE FROM player_nicknames WHERE account_id = ? AND nickname = ? COLLATE NOCASE",
                (account_id, nickname)
            )
            self.conn.commit()
            # cursor.rowcount will be 1 if a row was deleted, 0 otherwise
            if cursor.rowcount > 0:
                logger.info(f"[DB] Removed nickname '{nickname}' from account ID {account_id}.")
                return True
            else:
                logger.warning(f"[DB] Nickname '{nickname}' not found for account ID {account_id}. No action taken.")
                return False
        except sqlite3.Error as e:
            logger.error(f"SQLite error removing nickname '{nickname}' for account ID {account_id}: {e}")
            return False

    # --- Hero Nickname Methods ---
    def set_hero_nickname(self, hero_id: int, nickname: str) -> bool:
        """Assigns a nickname to a hero. Nicknames are unique (case-insensitive)."""
        try:
            # Verify hero_id exists
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM heroes WHERE hero_id = ?", (hero_id,))
            if not cursor.fetchone():
                logger.warning(f"[DB] Attempted to set nickname for non-existent hero_id: {hero_id}.")
                return False

            self.conn.execute(
                "INSERT INTO hero_nicknames (hero_id, nickname) VALUES (?, ?)",
                (hero_id, nickname)
            )
            self.conn.commit()
            logger.info(f"[DB] Assigned nickname '{nickname}' to hero ID {hero_id}.")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"[DB] Failed to assign nickname '{nickname}' to hero ID {hero_id}. Nickname likely already in use or duplicate for this hero.")
            existing_nicknames = self.get_hero_nicknames(hero_id)
            if nickname.lower() in [n.lower() for n in existing_nicknames]:
                logger.info(f"[DB] Hero ID {hero_id} already has nickname '{nickname}'. No action taken.")
                return True # Considered success
            return False
        except sqlite3.Error as e:
            logger.error(f"SQLite error assigning nickname '{nickname}' for hero ID {hero_id}: {e}")
            return False

    def get_hero_id_by_nickname(self, nickname: str) -> Optional[int]:
        """Finds a hero's ID by their nickname (case-insensitive)."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT hero_id FROM hero_nicknames WHERE nickname = ?", (nickname,))
            row = cursor.fetchone()
            if row:
                logger.debug(f"[DB] Found hero ID {row['hero_id']} for nickname '{nickname}'.")
                return row['hero_id']
            logger.debug(f"[DB] No hero ID found for nickname '{nickname}'.")
            return None
        except sqlite3.Error as e:
            logger.error(f"SQLite error getting hero ID for nickname '{nickname}': {e}")
            return None

    def get_hero_nicknames(self, hero_id: int) -> List[str]:
        """Retrieves all fixed nicknames for a hero."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT nickname FROM hero_nicknames WHERE hero_id = ? ORDER BY nickname", (hero_id,))
            rows = cursor.fetchall()
            nicknames = [row['nickname'] for row in rows]
            logger.debug(f"[DB] Found nicknames {nicknames} for hero ID {hero_id}." if nicknames else f"[DB] No nicknames found for hero ID {hero_id}.")
            return nicknames
        except sqlite3.Error as e:
            logger.error(f"SQLite error getting nicknames for hero ID {hero_id}: {e}")
            return []

    def remove_hero_nickname(self, hero_id: int, nickname: str) -> bool:
        """Removes a specific nickname from a hero. Case-insensitive."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "DELETE FROM hero_nicknames WHERE hero_id = ? AND nickname = ? COLLATE NOCASE",
                (hero_id, nickname)
            )
            self.conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"[DB] Removed nickname '{nickname}' from hero ID {hero_id}.")
                return True
            else:
                logger.warning(f"[DB] Nickname '{nickname}' not found for hero ID {hero_id}. No action taken.")
                return False
        except sqlite3.Error as e:
            logger.error(f"SQLite error removing nickname '{nickname}' for hero ID {hero_id}: {e}")
            return False

    def resolve_hero_identifier(self, identifier: str) -> Optional[int]:
        """Resolves a hero identifier (ID, official name, or nickname) to a hero_id."""
        # Try as integer ID first
        if identifier.isdigit():
            hero_id = int(identifier)
            cursor = self.conn.cursor()
            cursor.execute("SELECT hero_id FROM heroes WHERE hero_id = ?", (hero_id,))
            if cursor.fetchone():
                logger.debug(f"[DB] Resolved hero identifier '{identifier}' as direct hero_id: {hero_id}")
                return hero_id

        # Try as official name (case-insensitive)
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT hero_id FROM heroes WHERE name = ? COLLATE NOCASE", (identifier,))
            row = cursor.fetchone()
            if row:
                logger.debug(f"[DB] Resolved hero identifier '{identifier}' as official name to hero_id: {row['hero_id']}")
                return row['hero_id']
        except sqlite3.Error as e:
            logger.error(f"[DB] SQLite error resolving hero identifier '{identifier}' by official name: {e}")
        
        # Try as nickname (case-insensitive, handled by table schema)
        hero_id_from_nickname = self.get_hero_id_by_nickname(identifier)
        if hero_id_from_nickname is not None:
            logger.debug(f"[DB] Resolved hero identifier '{identifier}' as nickname to hero_id: {hero_id_from_nickname}")
            return hero_id_from_nickname

        logger.warning(f"[DB] Could not resolve hero identifier '{identifier}' to a known hero_id.")
        return None

    def get_downloaded_leagues_info(self) -> List[Dict[str, Any]]:
        """Retrieves information about leagues for which matches have been downloaded."""
        try:
            cursor = self.conn.cursor()
            # Get distinct league_ids from matches that have a leagueid set
            cursor.execute("SELECT DISTINCT JSON_EXTRACT(data, '$.leagueid') FROM matches WHERE JSON_VALID(data) AND JSON_EXTRACT(data, '$.leagueid') IS NOT NULL")
            league_ids_rows = cursor.fetchall()
            downloaded_league_ids = [row[0] for row in league_ids_rows if row[0] is not None]

            if not downloaded_league_ids:
                logger.info("[DB] No downloaded leagues found with associated match data.")
                return []

            # Fetch details for these league_ids from the all_leagues table
            # Using a placeholder for each ID in the IN clause
            placeholders = ','.join(['?'] * len(downloaded_league_ids))
            query = f"SELECT league_id, name, tier FROM all_leagues WHERE league_id IN ({placeholders}) ORDER BY name"
            cursor.execute(query, downloaded_league_ids)
            leagues_info_rows = cursor.fetchall()

            leagues_list = [
                {"league_id": r["league_id"], "name": r["name"], "tier": r["tier"]}
                for r in leagues_info_rows
            ]
            logger.debug(f"[DB] Found {len(leagues_list)} downloaded leagues with details.")
            return leagues_list
        except sqlite3.Error as e:
            logger.error(f"SQLite error getting downloaded leagues info: {e}")
            return []

    def get_stats_for_player(self, account_id: int) -> Optional[Dict[str, Any]]:
        """Calculates detailed statistics for a single player."""
        player_stats = {
            "matches_played": 0,
            "wins": 0,
            "heroes_played": {}
        }
        player_name = None

        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT data FROM matches")
            all_matches_raw = cursor.fetchall()

            for row in all_matches_raw:
                match_data = json.loads(row['data'])
                
                if not match_data.get("players"):
                    continue

                for player in match_data["players"]:
                    if player.get("account_id") == account_id:
                        player_stats["matches_played"] += 1
                        
                        if player.get("personaname"):
                            player_name = player.get("personaname")

                        is_radiant_player = player.get("isRadiant", player.get("player_slot", 0) < 128)
                        radiant_win = match_data.get("radiant_win", False)
                        if (is_radiant_player and radiant_win) or (not is_radiant_player and not radiant_win):
                            player_stats["wins"] += 1

                        hero_id = player.get("hero_id")
                        if hero_id:
                            if hero_id not in player_stats["heroes_played"]:
                                player_stats["heroes_played"][hero_id] = {"plays": 0, "wins": 0}
                            
                            player_stats["heroes_played"][hero_id]["plays"] += 1
                            if (is_radiant_player and radiant_win) or (not is_radiant_player and not radiant_win):
                                player_stats["heroes_played"][hero_id]["wins"] += 1
            
            if player_stats["matches_played"] == 0:
                logger.warning(f"No matches found for player {account_id}.")
                return None

            player_stats["win_rate"] = (player_stats["wins"] / player_stats["matches_played"]) * 100 if player_stats["matches_played"] > 0 else 0
            
            player_stats["player_name"] = player_name
            
            all_heroes = self.get_all_heroes()
            for hero_id, stats in player_stats["heroes_played"].items():
                stats["hero_name"] = all_heroes.get(hero_id, "Unknown Hero")

            return player_stats

        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.error(f"Error getting stats for player {account_id}: {e}")
            return None

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
