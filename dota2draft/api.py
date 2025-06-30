# dota2draft/api.py

import requests
import time
import os
from typing import List, Dict, Any, Optional
from .logger_config import logger
from .config_loader import CONFIG

class OpenDotaAPIClient:
    """A client for interacting with the OpenDota API."""
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.opendota.com/api"
        self.api_key = api_key or CONFIG.get("opendota_api_key") or os.environ.get("OPENDOTA_API_KEY")
        # Adaptive rate limiting
        self.base_cooldown = 1.0  # Base cooldown for free tier
        self.last_request_time = 0
        self.consecutive_errors = 0

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Makes a request to the OpenDota API with adaptive rate limiting."""
        if params is None:
            params = {}
        if self.api_key:
            params['api_key'] = self.api_key
        
        # Adaptive rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        cooldown = self.base_cooldown * (1.5 ** self.consecutive_errors)  # Exponential backoff
        
        if time_since_last < cooldown:
            sleep_time = cooldown - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        url = f"{self.base_url}/{endpoint}"
        try:
            logger.debug(f"Making API request to: {url}")
            response = requests.get(url, params=params, timeout=15)
            
            # Check for rate limit headers and adjust accordingly
            if 'X-RateLimit-Remaining' in response.headers:
                remaining = int(response.headers.get('X-RateLimit-Remaining', 1))
                if remaining < 5:  # If close to limit, slow down
                    self.base_cooldown = min(self.base_cooldown * 1.2, 5.0)
            
            response.raise_for_status()
            self.consecutive_errors = 0  # Reset error count on success
            self.last_request_time = time.time()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Rate limited
                self.consecutive_errors += 1
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited. Waiting {retry_after}s before retry")
                time.sleep(retry_after)
            logger.error(f"HTTP error for {url}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            self.consecutive_errors += 1
            logger.error(f"API request to {url} failed: {e}")
            return None
        finally:
            self.last_request_time = time.time()

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
