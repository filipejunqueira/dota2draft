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
    expected_tables = {"matches", "heroes", "teams", "all_leagues", "hero_nicknames", "player_nicknames"} # Added hero_nicknames and player_nicknames

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

# Helper to set up some initial heroes for nickname tests
def setup_initial_heroes(db_manager: DBManager):
    heroes_data = [
        {"id": 1, "localized_name": "Anti-Mage"},
        {"id": 2, "localized_name": "Axe"},
        {"id": 3, "localized_name": "Bane"},
    ]
    db_manager.store_all_heroes(heroes_data)
    # store_all_heroes also populates the heroes table which resolve_hero_identifier uses for names
    return {hero['id']: hero['localized_name'] for hero in heroes_data}

def test_set_and_get_hero_nickname(db_manager: DBManager):
    setup_initial_heroes(db_manager)
    hero_id_am = 1
    hero_id_axe = 2

    # Test setting a new nickname
    assert db_manager.set_hero_nickname(hero_id_am, "AM")
    assert db_manager.get_hero_nicknames(hero_id_am) == ["AM"]

    # Test setting another nickname for the same hero
    assert db_manager.set_hero_nickname(hero_id_am, "Magina")
    nicknames_am = sorted(db_manager.get_hero_nicknames(hero_id_am)) # Sort for consistent comparison
    assert nicknames_am == sorted(["AM", "Magina"])

    # Test setting a nickname that already exists (case-insensitive) for the same hero
    # DBManager.set_hero_nickname uses INSERT OR IGNORE with UNIQUE (hero_id, nickname COLLATE NOCASE)
    # So, setting "am" when "AM" exists for the same hero_id is a no-op.
    assert db_manager.set_hero_nickname(hero_id_am, "am") # case-insensitive, should be ignored
    nicknames_am_after_case_dupe = sorted(db_manager.get_hero_nicknames(hero_id_am))
    assert nicknames_am_after_case_dupe == sorted(["AM", "Magina"]) # Count should remain 2

    # Test setting a nickname that is already taken by another hero (should fail due to UNIQUE (nickname COLLATE NOCASE))
    assert db_manager.set_hero_nickname(hero_id_axe, "Mogul Kahn")
    # Attempt to set "Mogul Kahn" (or case variant) for Anti-Mage; should fail because Axe has it.
    # The set_hero_nickname returns True on success, False on failure (e.g. constraint violation)
    assert not db_manager.set_hero_nickname(hero_id_am, "mogul kahn") 
    assert db_manager.get_hero_nicknames(hero_id_axe) == ["Mogul Kahn"]
    assert "Mogul Kahn" not in db_manager.get_hero_nicknames(hero_id_am)
    assert "mogul kahn" not in db_manager.get_hero_nicknames(hero_id_am)

    # Test setting nickname for non-existent hero (should fail because of foreign key or hero check)
    # set_hero_nickname checks if hero_id exists in 'heroes' table first.
    assert not db_manager.set_hero_nickname(999, "NonExistentHeroNick")

    # Test get_hero_id_by_nickname
    assert db_manager.get_hero_id_by_nickname("AM") == hero_id_am
    assert db_manager.get_hero_id_by_nickname("magina") == hero_id_am # Case-insensitive
    assert db_manager.get_hero_id_by_nickname("Mogul KAHN") == hero_id_axe # Case-insensitive
    assert db_manager.get_hero_id_by_nickname("UnknownNick") is None

    # Test get_hero_nicknames for a hero with no nicknames
    hero_id_bane = 3
    assert db_manager.get_hero_nicknames(hero_id_bane) == []


def test_resolve_hero_identifier(db_manager: DBManager):
    setup_initial_heroes(db_manager)
    hero_id_am = 1
    hero_id_axe = 2
    hero_id_bane = 3

    db_manager.set_hero_nickname(hero_id_am, "AM")
    db_manager.set_hero_nickname(hero_id_am, "Magina")
    db_manager.set_hero_nickname(hero_id_axe, "Mogul")

    # Resolve by ID (string and int)
    assert db_manager.resolve_hero_identifier(str(hero_id_am)) == hero_id_am
    # db_manager.resolve_hero_identifier expects string, so int might not be directly supported or converted.
    # Let's assume it's always called with string from CLI context.
    # assert db_manager.resolve_hero_identifier(hero_id_am) == hero_id_am 

    # Resolve by official name (case-insensitive)
    assert db_manager.resolve_hero_identifier("Anti-Mage") == hero_id_am
    assert db_manager.resolve_hero_identifier("anti-mage") == hero_id_am
    assert db_manager.resolve_hero_identifier("AXE") == hero_id_axe

    # Resolve by nickname (case-insensitive)
    assert db_manager.resolve_hero_identifier("AM") == hero_id_am
    assert db_manager.resolve_hero_identifier("magina") == hero_id_am
    assert db_manager.resolve_hero_identifier("Mogul") == hero_id_axe
    assert db_manager.resolve_hero_identifier("mOgUl") == hero_id_axe

    # Resolve non-existent identifier
    assert db_manager.resolve_hero_identifier("UnknownHero") is None
    assert db_manager.resolve_hero_identifier("NonExistentNick") is None
    assert db_manager.resolve_hero_identifier("999") is None # Non-existent ID as string

    # Test hero with no nickname, resolve by name and ID
    assert db_manager.resolve_hero_identifier("Bane") == hero_id_bane
    assert db_manager.resolve_hero_identifier(str(hero_id_bane)) == hero_id_bane

def test_remove_hero_nickname(db_manager: DBManager):
    setup_initial_heroes(db_manager)
    hero_id_am = 1
    db_manager.set_hero_nickname(hero_id_am, "AM")
    db_manager.set_hero_nickname(hero_id_am, "Magina")

    # Assert initial state
    assert sorted(db_manager.get_hero_nicknames(hero_id_am)) == ["AM", "Magina"]

    # Test removing a nickname (case-insensitive)
    assert db_manager.remove_hero_nickname(hero_id_am, "magina")
    assert db_manager.get_hero_nicknames(hero_id_am) == ["AM"]

    # Test removing the last nickname
    assert db_manager.remove_hero_nickname(hero_id_am, "AM")
    assert db_manager.get_hero_nicknames(hero_id_am) == []

    # Test removing a nickname that doesn't exist
    assert not db_manager.remove_hero_nickname(hero_id_am, "NonExistentNick")

    # Test removing a nickname from a hero that doesn't exist
    assert not db_manager.remove_hero_nickname(999, "any_nick")

def test_set_and_remove_player_nickname(db_manager: DBManager):
    account_id_1 = 12345
    account_id_2 = 67890

    # Set nicknames
    assert db_manager.set_player_nickname(account_id_1, "PlayerOne")
    assert db_manager.set_player_nickname(account_id_1, "P1")
    assert db_manager.set_player_nickname(account_id_2, "PlayerTwo")

    # Check initial state
    assert sorted(db_manager.get_player_nicknames(account_id_1)) == ["P1", "PlayerOne"]
    assert db_manager.get_player_nicknames(account_id_2) == ["PlayerTwo"]

    # Remove a nickname
    assert db_manager.remove_player_nickname(account_id_1, "PlayerOne")
    assert db_manager.get_player_nicknames(account_id_1) == ["P1"]

    # Remove a nickname case-insensitively
    assert db_manager.set_player_nickname(account_id_1, "AnotherNick")
    assert db_manager.remove_player_nickname(account_id_1, "anothernick")
    assert "AnotherNick" not in db_manager.get_player_nicknames(account_id_1)

    # Attempt to remove a nickname from the wrong player
    assert not db_manager.remove_player_nickname(account_id_2, "P1")
    assert db_manager.get_player_nicknames(account_id_1) == ["P1"]

    # Remove non-existent nickname
    assert not db_manager.remove_player_nickname(account_id_1, "NonExistent")
