# dota2draft/core.py

from typing import List, Optional, Dict, Any, Tuple
from enum import Enum, auto

from .db import DBManager
from .api import OpenDotaAPIClient
from .logger_config import logger # Import the configured logger

class FetchStatus(Enum):
    """Enumeration for the status of a fetch operation."""
    ADDED = auto()
    SKIPPED = auto()
    FAILED = auto()

class DataService:
    def __init__(self, db_manager: DBManager, api_client: OpenDotaAPIClient):
        """
        Constructor for DataService.
        """
        self.db_manager = db_manager
        self.api_client = api_client
        self._hero_map_cache: Optional[Dict[int, str]] = None
        self._team_map_cache: Optional[Dict[int, str]] = None
        self._all_leagues_list_cache: Optional[List[Dict[str, Any]]] = None
        logger.debug("DataService initialized.")

    def get_hero_map(self, force_refresh: bool = False) -> Dict[int, str]:
        """
        Gets the hero ID to hero name mapping.
        """
        if force_refresh:
            logger.info("[CACHE] Force refreshing hero map.")
            self._hero_map_cache = None
            self.db_manager.clear_heroes_table()

        if self._hero_map_cache is not None:
            logger.debug("[CACHE] Hero map retrieved from memory cache.")
            return self._hero_map_cache

        db_hero_map = self.db_manager.get_all_heroes()
        if db_hero_map and not force_refresh:
            logger.debug("[CACHE] Hero map retrieved from DB and cached in memory.")
            self._hero_map_cache = db_hero_map
            return db_hero_map

        logger.info("[CACHE] Hero map not in cache or DB (or force refresh). Fetching from API.")
        api_data_list = self.api_client.fetch_hero_stats()
        if api_data_list:
            self.db_manager.store_all_heroes(api_data_list)
            fresh_hero_map = self.db_manager.get_all_heroes()
            self._hero_map_cache = fresh_hero_map
            logger.info(f"[CACHE] Hero map fetched from API, stored in DB, and cached. Found {len(fresh_hero_map)} heroes.")
            return fresh_hero_map
        logger.warning("[CACHE] Failed to fetch hero map from API.")
        return {}

    def get_team_map(self, force_refresh: bool = False) -> Dict[int, str]:
        """
        Gets the team ID to team name mapping.
        """
        if force_refresh:
            logger.info("[CACHE] Force refreshing team map.")
            self._team_map_cache = None
            self.db_manager.clear_teams_table()

        if self._team_map_cache is not None:
            logger.debug("[CACHE] Team map retrieved from memory cache.")
            return self._team_map_cache

        db_team_map = self.db_manager.get_all_teams()
        if db_team_map and not force_refresh:
            logger.debug("[CACHE] Team map retrieved from DB and cached in memory.")
            self._team_map_cache = db_team_map
            return db_team_map

        logger.info("[CACHE] Team map not in cache or DB (or force refresh). Fetching from API.")
        api_data_list = self.api_client.fetch_all_teams()
        if api_data_list:
            self.db_manager.store_all_teams(api_data_list)
            fresh_team_map = self.db_manager.get_all_teams()
            self._team_map_cache = fresh_team_map
            logger.info(f"[CACHE] Team map fetched from API, stored in DB, and cached. Found {len(fresh_team_map)} teams.")
            return fresh_team_map
        logger.warning("[CACHE] Failed to fetch team map from API.")
        return {}

    def get_all_leagues(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Gets the list of all leagues.
        """
        if force_refresh:
            logger.info("[CACHE] Force refreshing all leagues list.")
            self._all_leagues_list_cache = None
            self.db_manager.clear_all_leagues_table()

        if self._all_leagues_list_cache is not None:
            logger.debug("[CACHE] All leagues list retrieved from memory cache.")
            return self._all_leagues_list_cache

        db_leagues_list = self.db_manager.get_all_leagues()
        if db_leagues_list and not force_refresh:
            logger.debug("[CACHE] All leagues list retrieved from DB and cached in memory.")
            self._all_leagues_list_cache = db_leagues_list
            return db_leagues_list

        logger.info("[CACHE] All leagues list not in cache or DB (or force refresh). Fetching from API.")
        api_data_list = self.api_client.fetch_all_leagues()
        if api_data_list:
            self.db_manager.store_all_leagues(api_data_list)
            fresh_leagues_list = self.db_manager.get_all_leagues()
            self._all_leagues_list_cache = fresh_leagues_list
            logger.info(f"[CACHE] All leagues list fetched from API, stored in DB, and cached. Found {len(fresh_leagues_list)} leagues.")
            return fresh_leagues_list
        logger.warning("[CACHE] Failed to fetch all leagues list from API.")
        return []

    def get_match_details(self, match_id: int, force_refresh: bool = False) -> Tuple[FetchStatus, Optional[Dict[str, Any]]]:
        """
        Gets details for a specific match, using the database as a cache.
        Returns a status tuple indicating the outcome.
        """
        logger.debug(f"Getting match details for ID: {match_id}, Force refresh: {force_refresh}")
        if not force_refresh:
            db_data = self.db_manager.get_match_data(match_id)
            if db_data:
                logger.debug(f"Match {match_id} found in DB cache. Skipping.")
                return FetchStatus.SKIPPED, db_data
        
        logger.info(f"Match {match_id} not in DB cache or force refresh. Fetching from API.")
        try:
            api_data = self.api_client.fetch_match_details(match_id)
            if api_data:
                self.db_manager.store_match_data(match_id, api_data)
                logger.info(f"Match {match_id} fetched from API and stored in DB.")
                return FetchStatus.ADDED, api_data
            else:
                logger.warning(f"Failed to fetch match {match_id} from API (no data returned).")
                return FetchStatus.FAILED, None
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching match {match_id}: {e}", exc_info=True)
            return FetchStatus.FAILED, None
