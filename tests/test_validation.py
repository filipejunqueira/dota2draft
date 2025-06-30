"""Tests for validation utilities."""

import pytest
from dota2draft.validation import (
    validate_match_id, validate_league_id, validate_account_id,
    validate_date_string, validate_hero_nickname, validate_draft_string,
    sanitize_filename
)
from dota2draft.exceptions import DataValidationError


class TestMatchIdValidation:
    def test_valid_match_id(self):
        assert validate_match_id(12345) == 12345
        assert validate_match_id("12345") == 12345
    
    def test_invalid_match_id_negative(self):
        with pytest.raises(DataValidationError, match="must be positive"):
            validate_match_id(-1)
    
    def test_invalid_match_id_zero(self):
        with pytest.raises(DataValidationError, match="must be positive"):
            validate_match_id(0)
    
    def test_invalid_match_id_too_large(self):
        with pytest.raises(DataValidationError, match="too large"):
            validate_match_id(9999999999999)
    
    def test_invalid_match_id_string(self):
        with pytest.raises(DataValidationError, match="Invalid match ID format"):
            validate_match_id("not_a_number")


class TestLeagueIdValidation:
    def test_valid_league_id(self):
        assert validate_league_id(123) == 123
        assert validate_league_id("456") == 456
    
    def test_invalid_league_id_negative(self):
        with pytest.raises(DataValidationError, match="must be positive"):
            validate_league_id(-1)


class TestDateValidation:
    def test_valid_date(self):
        assert validate_date_string("2023-01-01") == "2023-01-01"
        assert validate_date_string("2024-12-31") == "2024-12-31"
    
    def test_invalid_date_format(self):
        with pytest.raises(DataValidationError, match="Invalid date format"):
            validate_date_string("01-01-2023")
    
    def test_invalid_date_value(self):
        with pytest.raises(DataValidationError, match="Invalid date"):
            validate_date_string("2023-13-01")  # Month 13 doesn't exist


class TestHeroNicknameValidation:
    def test_valid_nickname(self):
        assert validate_hero_nickname("AM") == "AM"
        assert validate_hero_nickname("Anti-Mage") == "Anti-Mage"
        assert validate_hero_nickname("Hero_123") == "Hero_123"
    
    def test_empty_nickname(self):
        with pytest.raises(DataValidationError, match="cannot be empty"):
            validate_hero_nickname("")
    
    def test_too_long_nickname(self):
        long_name = "a" * 51
        with pytest.raises(DataValidationError, match="too long"):
            validate_hero_nickname(long_name)
    
    def test_invalid_characters(self):
        with pytest.raises(DataValidationError, match="invalid characters"):
            validate_hero_nickname("Hero@123")


class TestDraftStringValidation:
    def test_valid_draft_string(self):
        draft = "Radiant Pick: Axe; Dire Ban: Invoker"
        assert validate_draft_string(draft) == draft
    
    def test_empty_draft_string(self):
        with pytest.raises(DataValidationError, match="cannot be empty"):
            validate_draft_string("")
    
    def test_invalid_draft_format(self):
        with pytest.raises(DataValidationError, match="doesn't contain valid"):
            validate_draft_string("This is not a draft string")


class TestFilenameValidization:
    def test_valid_filename(self):
        assert sanitize_filename("test.csv") == "test.csv"
        assert sanitize_filename("match_123.json") == "match_123.json"
    
    def test_dangerous_filename(self):
        with pytest.raises(DataValidationError, match="dangerous path"):
            sanitize_filename("../../../etc/passwd")
    
    def test_sanitize_special_characters(self):
        result = sanitize_filename("file name with spaces.txt")
        assert result == "file_name_with_spaces.txt"