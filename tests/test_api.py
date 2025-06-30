"""Tests for API client functionality."""

import pytest
import requests
from unittest.mock import Mock, patch
from dota2draft.api import OpenDotaAPIClient


class TestOpenDotaAPIClient:
    
    @pytest.fixture
    def client(self):
        return OpenDotaAPIClient()
    
    @patch('dota2draft.api.requests.get')
    def test_successful_request(self, mock_get, client):
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "data"}
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        result = client._make_request("test/endpoint")
        
        assert result == {"test": "data"}
        mock_get.assert_called_once()
    
    @patch('dota2draft.api.requests.get')
    def test_rate_limit_handling(self, mock_get, client):
        # Mock rate limited response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {'Retry-After': '1', 'X-RateLimit-Remaining': '0'}
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Rate limited")
        mock_get.return_value = mock_response
        
        with patch('dota2draft.api.time.sleep') as mock_sleep:
            result = client._make_request("test/endpoint")
            
            assert result is None
            mock_sleep.assert_called()
    
    @patch('dota2draft.api.requests.get')
    def test_adaptive_rate_limiting(self, mock_get, client):
        # Test that consecutive errors increase cooldown
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_response.raise_for_status.side_effect = Exception("Server error")
        mock_get.return_value = mock_response
        
        # Make first request (should set consecutive_errors = 1)
        client._make_request("test/endpoint")
        
        # Check that consecutive_errors increased
        assert client.consecutive_errors == 1
    
    def test_fetch_matches_for_league(self, client):
        test_data = [
            {"match_id": 123, "other_field": "value"},
            {"match_id": 456, "other_field": "value"}
        ]
        
        with patch.object(client, '_make_request', return_value=test_data):
            result = client.fetch_matches_for_league(12345)
            
            assert result == [123, 456]
    
    def test_fetch_matches_empty_response(self, client):
        with patch.object(client, '_make_request', return_value=None):
            result = client.fetch_matches_for_league(12345)
            
            assert result == []
    
    def test_environment_variable_api_key(self):
        with patch.dict('os.environ', {'OPENDOTA_API_KEY': 'test_key'}):
            client = OpenDotaAPIClient()
            assert client.api_key == 'test_key'