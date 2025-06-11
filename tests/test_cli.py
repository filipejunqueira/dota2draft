# tests/test_cli.py

import pytest
from typer.testing import CliRunner

# The app is now structured with sub-typers
from dota2draft_cli import app
from dota2draft.db import DBManager
from dota2draft.core import FetchStatus

# Fixture to create a DBManager with an in-memory database
@pytest.fixture
def db_manager():
    """
    Pytest fixture for an in-memory DBManager instance.
    This ensures that tests are isolated and don't affect the actual database.
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
            {"is_pick": True, "hero_id": 1, "team": 0}, # AM Pick
            {"is_pick": False, "hero_id": 2, "team": 1}, # Axe Ban
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

# Use mocker to patch the global db_manager instance used by the CLI
@pytest.fixture(autouse=True)
def patch_db_manager(mocker, db_manager):
    """
    Automatically mock the db_manager instance in the dota2draft_cli module
    for all tests in this file.
    """
    mocker.patch('dota2draft_cli.db_manager', db_manager)


def test_heroes_stats_command(runner: CliRunner):
    """
    Integration test for the 'heroes stats' CLI command.
    """
    # Run the command with the new structure
    result = runner.invoke(app, ["heroes", "stats"])

    # Assertions
    assert result.exit_code == 0
    assert "Hero Statistics" in result.stdout
    assert "Anti-Mage" in result.stdout
    assert "Axe" in result.stdout


def test_heroes_stats_command_with_date_filter(runner: CliRunner):
    """
    Integration test for the 'heroes stats' command with the --after-date filter.
    """
    # Run the command with the new structure and filter
    result = runner.invoke(app, ["heroes", "stats", "--after-date", "2023-01-02"])

    # Assertions
    assert result.exit_code == 0
    assert "Hero Statistics (after 2023-01-02)" in result.stdout
    assert "Anti-Mage" in result.stdout # AM played in the second match
    assert "Axe" not in result.stdout   # Axe was only in the first match


def test_fetch_league_command(runner: CliRunner, mocker):
    """
    Integration test for the top-level 'fetch-league' command.
    """
    # Mock the dependencies of the command
    mock_api_client = mocker.patch('dota2draft_cli.api_client')
    mock_data_service = mocker.patch('dota2draft_cli.data_service')

    # Simulate the API returning a list of match summaries
    mock_api_client.fetch_league_matches_summary.return_value = [{'match_id': 1}, {'match_id': 2}]
    # Simulate get_match_details returning a success status
    mock_data_service.get_match_details.return_value = (FetchStatus.ADDED, {})


    # 1. Run the command without the --force-refresh flag
    result = runner.invoke(app, ["fetch-league", "123"])

    assert result.exit_code == 0, result.stdout
    # Verify that get_match_details was called for each match with force_refresh=False
    mock_data_service.get_match_details.assert_any_call(1, force_refresh=False)
    mock_data_service.get_match_details.assert_any_call(2, force_refresh=False)
    assert mock_data_service.get_match_details.call_count == 2

    # Reset the mock for the next run
    mock_data_service.reset_mock()
    mock_data_service.get_match_details.return_value = (FetchStatus.ADDED, {})


    # 2. Run the command WITH the --force-refresh flag
    result_forced = runner.invoke(app, ["fetch-league", "123", "--force-refresh"])

    assert result_forced.exit_code == 0, result_forced.stdout
    # Verify that get_match_details was called for each match with force_refresh=True
    mock_data_service.get_match_details.assert_any_call(1, force_refresh=True)
    mock_data_service.get_match_details.assert_any_call(2, force_refresh=True)
    assert mock_data_service.get_match_details.call_count == 2


def test_hero_nickname_commands(runner: CliRunner):
    """
    Integration test for the hero nickname management commands (set, list, remove).
    """
    hero_identifier = "Anti-Mage"
    nickname = "AM"

    # 1. Set a nickname
    result_set = runner.invoke(app, ["heroes", "set-nickname", hero_identifier, nickname])
    assert result_set.exit_code == 0
    assert f"Success: Nickname '{nickname}' assigned to hero '{hero_identifier}'." in result_set.stdout

    # 2. List nicknames
    result_list = runner.invoke(app, ["heroes", "list-nicknames", hero_identifier])
    assert result_list.exit_code == 0
    assert f"Nicknames for {hero_identifier}" in result_list.stdout
    assert nickname in result_list.stdout

    # 3. Remove the nickname
    result_remove = runner.invoke(app, ["heroes", "remove-nickname", hero_identifier, nickname])
    assert result_remove.exit_code == 0
    assert f"Success: Nickname '{nickname}' was removed from hero '{hero_identifier}'." in result_remove.stdout

    # 4. Verify nickname is gone
    result_list_after_remove = runner.invoke(app, ["heroes", "list-nicknames", hero_identifier])
    assert result_list_after_remove.exit_code == 0
    assert "No nicknames found" in result_list_after_remove.stdout


def test_player_nickname_commands(runner: CliRunner):
    """
    Integration test for the player nickname management commands (set, list, remove).
    """
    account_id = "1001"
    nickname = "TestPlayer"

    # 1. Set a nickname
    result_set = runner.invoke(app, ["players", "set-nickname", account_id, nickname])
    assert result_set.exit_code == 0
    assert f"Successfully assigned nickname '{nickname}' to account ID {account_id}." in result_set.stdout

    # 2. List nicknames
    result_list = runner.invoke(app, ["players", "list-nicknames", account_id])
    assert result_list.exit_code == 0
    assert f"Nicknames for player {account_id}" in result_list.stdout
    assert nickname in result_list.stdout

    # 3. Remove the nickname
    result_remove = runner.invoke(app, ["players", "remove-nickname", account_id, nickname])
    assert result_remove.exit_code == 0
    assert f"Success: Nickname '{nickname}' removed from player {account_id}." in result_remove.stdout

    # 4. Verify nickname is gone
    result_list_after_remove = runner.invoke(app, ["players", "list-nicknames", account_id])
    assert result_list_after_remove.exit_code == 0
    assert "No nicknames found" in result_list_after_remove.stdout
