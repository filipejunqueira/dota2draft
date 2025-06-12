# dota2draft/api.py

import requests
import time
from typing import List, Dict, Any, Optional
from .logger_config import logger
from .config_loader import CONFIG

class OpenDotaAPIClient:
    """A client for interacting with the OpenDota API."""
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.opendota.com/api"
        self.api_key = api_key or CONFIG.get("opendota_api_key")
        # Default cooldown to 1 second as per OpenDota's free tier rate limit (60/min)
        self.rate_limit_cooldown = 1

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Makes a request to the OpenDota API and handles responses."""
        if params is None:
            params = {}
        if self.api_key:
            params['api_key'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        try:
            logger.debug(f"Making API request to: {url}")
            response = requests.get(url, params=params, timeout=15) # Added timeout
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request to {url} failed: {e}")
            return None
        finally:
            # Respect rate limits by waiting after every request
            time.sleep(self.rate_limit_cooldown)

    def fetch_matches_for_league(self, league_id: int) -> List[int]:
        """Fetches all match IDs for a given league."""
        data = self._make_request(f"leagues/{league_id}/matches")
        if isinstance(data, list):
            return [match['match_id'] for match in data if 'match_id' in match]
        logger.warning(f"Received unexpected data type from league matches endpoint for league {league_id}: {type(data)}")
        return []

    def fetch_match_details(self, match_id: int) -> Optional[Dict[str, Any]]:
        """Fetches detailed data for a specific match."""
        return self._make_request(f"matches/{match_id}")

    def fetch_all_heroes(self) -> List[Dict[str, Any]]:
        """Fetches data for all heroes."""
        return self._make_request("heroes") or []

    def fetch_all_teams(self) -> List[Dict[str, Any]]:
        """Fetches data for all teams."""
        return self._make_request("teams") or []

    def fetch_all_leagues(self) -> List[Dict[str, Any]]:
        """Fetches data for all leagues."""
        return self._make_request("leagues") or []
