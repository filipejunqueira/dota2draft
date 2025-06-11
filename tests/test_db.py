# tests/test_db.py
import pytest
import sqlite3
from dota2draft.db import DBManager

@pytest.fixture
def db_manager():
    """
    Pytest fixture to create a DBManager instance with an in-memory SQLite database.
    This ensures that tests are isolated and don't affect the actual database.
    """
    manager = DBManager(db_name=":memory:")
    yield manager
    # Teardown: explicitly close the connection to clean up resources.
    manager.close()

def test_initialization(db_manager: DBManager):
    """
    Tests if the DBManager initializes correctly and creates all the necessary tables.
    """
    # List of expected tables
    expected_tables = {"matches", "heroes", "teams", "all_leagues"}

    cursor = db_manager.conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables_in_db = {row[0] for row in cursor.fetchall()}

    # Check if all expected tables were created
    assert expected_tables.issubset(tables_in_db)

def test_store_and_get_match(db_manager: DBManager):
    """Tests storing and retrieving a match."""
    match_id = 12345
    mock_match_data = {"match_id": match_id, "radiant_win": True, "duration": 2400}

    # Store the match
    db_manager.store_match_data(match_id, mock_match_data)

    # Retrieve the match
    retrieved_match = db_manager.get_match_data(match_id)

    assert retrieved_match is not None
    assert retrieved_match["match_id"] == match_id
    assert retrieved_match["duration"] == 2400

def test_get_hero_stats(db_manager: DBManager):
    """
    Tests the hero statistics calculation logic.
    """
    # 1. Setup: Store mock hero and match data
    mock_heroes = [
        {"id": 1, "localized_name": "Anti-Mage"},
        {"id": 2, "localized_name": "Axe"},
        {"id": 3, "localized_name": "Bane"},
    ]
    db_manager.store_all_heroes(mock_heroes)

    # Match 1: AM picks and wins, Axe is banned
    mock_match_1 = {
        "match_id": 101,
        "radiant_win": True,
        "start_time": 1672531200, # 2023-01-01
        "picks_bans": [
            {"is_pick": True, "hero_id": 1, "team": 0}, # AM picked by radiant
            {"is_pick": False, "hero_id": 2, "team": 1}, # Axe banned by dire
        ]
    }
    # Match 2: AM picks and loses, Bane picks and wins
    mock_match_2 = {
        "match_id": 102,
        "radiant_win": False,
        "start_time": 1672617600, # 2023-01-02
        "picks_bans": [
            {"is_pick": True, "hero_id": 1, "team": 0}, # AM picked by radiant
            {"is_pick": True, "hero_id": 3, "team": 1}, # Bane picked by dire
        ]
    }
    db_manager.store_match_data(101, mock_match_1)
    db_manager.store_match_data(102, mock_match_2)

    # 2. Action: Get hero stats
    hero_stats = db_manager.get_hero_stats()

    # 3. Assert: Verify the stats are correct
    stats_map = {s['hero_name']: s for s in hero_stats}

    assert len(hero_stats) == 3
    assert stats_map["Anti-Mage"]["picks"] == 2
    assert stats_map["Anti-Mage"]["bans"] == 0
    assert stats_map["Anti-Mage"]["wins"] == 1

    assert stats_map["Axe"]["picks"] == 0
    assert stats_map["Axe"]["bans"] == 1
    assert stats_map["Axe"]["wins"] == 0

    assert stats_map["Bane"]["picks"] == 1
    assert stats_map["Bane"]["bans"] == 0
    assert stats_map["Bane"]["wins"] == 1

    # Test date filter
    hero_stats_after_date = db_manager.get_hero_stats(after_date="2023-01-02")
    stats_map_after_date = {s['hero_name']: s for s in hero_stats_after_date}

    assert len(hero_stats_after_date) == 2 # Only AM and Bane in the second match
    assert stats_map_after_date["Anti-Mage"]["picks"] == 1
    assert stats_map_after_date["Anti-Mage"]["wins"] == 0
    assert stats_map_after_date["Bane"]["picks"] == 1
    assert stats_map_after_date["Bane"]["wins"] == 1
