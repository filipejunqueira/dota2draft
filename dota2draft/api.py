# dota2draft/api.py

import requests
import json
from typing import Optional, Dict, Any, List
from .logger_config import logger

class OpenDotaAPIClient:
    BASE_URL = "https://api.opendota.com/api"

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        A generic helper method to make a GET request to a specific API endpoint.
        """
        url = f"{self.BASE_URL}/{endpoint}"
        logger.debug(f"Making API request to: {url} with params: {params}")
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"[API ERR] HTTP error for {url}: {http_err} - Response: {http_err.response.text[:200]}")
        except requests.exceptions.RequestException as req_err: 
            logger.error(f"[API ERR] Request error for {url}: {req_err}")
        except json.JSONDecodeError as json_err:
            logger.error(f"[API ERR] JSON decode error for {url}: {json_err} - Response: {response.text[:200]}")
        return None 

    def fetch_hero_stats(self) -> Optional[List[Dict[str, Any]]]:
        """Fetches general statistics for all heroes."""
        logger.info("[API] Fetching hero stats...")
        return self._request("heroStats")

    def fetch_all_teams(self) -> Optional[List[Dict[str, Any]]]:
        """Fetches a list of all professional Dota 2 teams."""
        logger.info("[API] Fetching all teams...")
        return self._request("teams")

    def fetch_all_leagues(self) -> Optional[List[Dict[str, Any]]]:
        """Fetches a list of all Dota 2 leagues/tournaments."""
        logger.info("[API] Fetching all leagues...")
        return self._request("leagues")

    def fetch_match_details(self, match_id: int) -> Optional[Dict[str, Any]]:
        """Fetches detailed information for a specific match ID."""
        logger.info(f"[API] Fetching match details for {match_id}...")
        return self._request(f"matches/{match_id}")
    
    def fetch_league_matches_summary(self, league_id: int) -> Optional[List[Dict[str, Any]]]:
        """Fetches a summary list of matches for a specific league ID."""
        logger.info(f"[API] Fetching match summaries for league {league_id}...")
        matches = self._request(f"leagues/{league_id}/matches")
        if matches is None:
            # An API error occurred, which is already logged by _request.
            return None
        if not matches:
            # The request was successful, but no matches were returned.
            logger.warning(f"[API] No matches found for league {league_id}. The league might be invalid or have no matches processed by OpenDota.")
        return matches
