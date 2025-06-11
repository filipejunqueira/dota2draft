# tests/test_cli.py

import pytest
from typer.testing import CliRunner

from dota2draft_cli import app
from dota2draft.db import DBManager

# Fixture to create a DBManager with an in-memory database
@pytest.fixture
def db_manager():
    """
    Pytest fixture for an in-memory DBManager instance.
    """
    manager = DBManager(db_name=":memory:")
    # Setup: Store mock hero and match data
    mock_heroes = [
        {"id": 1, "localized_name": "Anti-Mage"},
        {"id": 2, "localized_name": "Axe"},
    ]
    manager.store_all_heroes(mock_heroes)

    mock_match_1 = {
        "match_id": 101,
        "radiant_win": True,
        "start_time": 1672531200, # 2023-01-01
        "picks_bans": [
            {"is_pick": True, "hero_id": 1, "team": 0},
            {"is_pick": False, "hero_id": 2, "team": 1},
        ],
        "players": [
            {"account_id": 1001, "personaname": "PlayerA", "isRadiant": True},
            {"account_id": 1002, "personaname": "PlayerB", "isRadiant": False},
        ]
    }
    mock_match_2 = {
        "match_id": 102,
        "radiant_win": False,
        "start_time": 1672617600, # 2023-01-02
        "picks_bans": [
            {"is_pick": True, "hero_id": 1, "team": 0}, # AM picked by radiant
        ],
        "players": [
            {"account_id": 1001, "personaname": "PlayerA", "isRadiant": True},
            {"account_id": 1003, "personaname": "PlayerC", "isRadiant": False},
        ]
    }
    manager.store_match_data(101, mock_match_1)
    manager.store_match_data(102, mock_match_2)

    yield manager
    manager.close()

# Fixture for the Typer CLI runner
@pytest.fixture
def runner():
    """Pytest fixture for a Typer CliRunner."""
    return CliRunner()

def test_stats_heroes_command(runner: CliRunner, db_manager: DBManager, mocker):
    """
    Integration test for the 'stats heroes' CLI command.
    It checks if the command runs and displays the correct hero statistics.
    """
    # Mock the DBManager class in the context of the dota2draft_cli module
    # to ensure that any instance created within the CLI commands uses our test database.
    mocker.patch('dota2draft_cli.db_manager', db_manager)

    # Run the command
    result = runner.invoke(app, ["stats", "heroes"])

    # Assertions
    assert result.exit_code == 0
    assert "Hero Statistics" in result.stdout
    assert "Anti-Mage" in result.stdout
    assert "1" in result.stdout # Picks
    assert "0" in result.stdout # Bans
    assert "1" in result.stdout # Wins for AM
    assert "Axe" in result.stdout
    assert "1" in result.stdout # Bans for Axe

def test_stats_heroes_command_with_date_filter(runner: CliRunner, db_manager: DBManager, mocker):
    """
    Integration test for the 'stats heroes' command with the --after-date filter.
    """
    mocker.patch('dota2draft_cli.db_manager', db_manager)

    # Run the command with the date filter
    result = runner.invoke(app, ["stats", "heroes", "--after-date", "2023-01-02"])

    # Assertions
    assert result.exit_code == 0
    assert "Hero Statistics (after 2023-01-02)" in result.stdout
    assert "Anti-Mage" in result.stdout # AM played in the second match
    assert "Axe" not in result.stdout   # Axe was only in the first match
    assert "1" in result.stdout # Picks for AM
    assert "0" in result.stdout # Wins for AM

def test_stats_players_command(runner: CliRunner, db_manager: DBManager, mocker):
    """
    Integration test for the 'stats players' CLI command.
    """
    mocker.patch('dota2draft_cli.db_manager', db_manager)

    # Run the command
    result = runner.invoke(app, ["stats", "players"])

    # Assertions
    assert result.exit_code == 0
    assert "Player Statistics" in result.stdout
    # PlayerA: 2 matches, 1 win
    assert "PlayerA" in result.stdout
    assert "2" in result.stdout # Matches Played
    assert "1" in result.stdout # Wins
    # PlayerB: 1 match, 0 wins
    assert "PlayerB" in result.stdout
    # PlayerC: 1 match, 1 win
    assert "PlayerC" in result.stdout
