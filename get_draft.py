import requests
import json


def get_hero_id_to_name_map():
    """
    Fetches hero data from the OpenDota API and creates a map of hero_id to hero_name.

    Returns:
        dict: A dictionary mapping hero_id (int) to localized_name (str).
              Returns an empty dictionary if the request fails or data is malformed.
    """
    hero_map = {}
    url = "https://api.opendota.com/api/heroStats"  # This endpoint contains hero names and IDs
    print(f"Fetching hero data from: {url}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            heroes_data = response.json()
            for hero in heroes_data:
                if "id" in hero and "localized_name" in hero:
                    hero_map[hero["id"]] = hero["localized_name"]
            if not hero_map:
                print(
                    "Warning: Hero map is empty. Check API response or hero data structure."
                )
            else:
                print(f"Successfully built hero map with {len(hero_map)} heroes.")
        else:
            print(
                f"Error: Failed to fetch hero data. Status code: {response.status_code}"
            )
            print(f"Response content: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error: An exception occurred while fetching hero data: {e}")
    except json.JSONDecodeError:
        print("Error: Could not decode JSON response from hero data API.")

    return hero_map


def get_match_details(match_id):
    """
    Fetches match details from the OpenDota API for a given match_id.

    Args:
        match_id (int or str): The ID of the match to fetch.

    Returns:
        dict: A dictionary containing the match details if the request is successful,
              None otherwise.
    """
    # Construct the API URL
    url = f"https://api.opendota.com/api/matches/{match_id}"
    print(f"Fetching match data from: {url}")

    try:
        # Make the GET request to the OpenDota API
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            match_data = response.json()
            return match_data
        else:
            print(
                f"Error: API request failed with status code {response.status_code} for match {match_id}"
            )
            print(f"Response content: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(
            f"Error: An exception occurred while making the API request for match {match_id}: {e}"
        )
        return None
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON response from match data API for match {match_id}."
        )
        return None


def display_draft_info(match_data, hero_map):
    """
    Displays the pick and ban information from the match data, using hero names.

    Args:
        match_data (dict): The dictionary containing match details.
        hero_map (dict): A dictionary mapping hero_id to hero_name.
    """
    if not hero_map:
        print("Warning: Hero map is not available. Hero names will not be displayed.")

    if match_data and "picks_bans" in match_data:
        picks_bans = match_data.get("picks_bans")  # Use .get for safer access
        if picks_bans is not None:  # Check if picks_bans is present, even if empty
            if not picks_bans:  # Specifically check if it's an empty list
                print(
                    "\nNo pick/ban actions recorded in this match (picks_bans is an empty list)."
                )
                print(
                    "This might be because the match was not a Captains Mode game or the replay is not fully parsed yet."
                )
                return

            print("\n--- Draft Information (Picks and Bans) ---")
            for action in picks_bans:
                action_type = "Pick" if action.get("is_pick") else "Ban"
                team_id = action.get("team")
                team = (
                    "Radiant"
                    if team_id == 0
                    else "Dire" if team_id == 1 else f"Unknown Team ({team_id})"
                )
                hero_id = action.get("hero_id")
                order = action.get("order")

                hero_name = hero_map.get(hero_id, f"Unknown Hero (ID: {hero_id})")

                print(
                    f"Order: {order}, Team: {team}, Action: {action_type}, Hero: {hero_name} (ID: {hero_id})"
                )
        else:
            # This case handles if 'picks_bans' is explicitly null in the JSON,
            # or if .get() returned None because the key was missing (though the outer 'if' checks for key presence).
            print(
                "\nNo pick/ban information available for this match (picks_bans is null or missing)."
            )
            print(
                "This might be because the match was not a Captains Mode game or the replay is not fully parsed yet."
            )
    elif match_data:
        print("\n'picks_bans' key not found in the match data.")
        print(
            "This could mean the match was not a Captains Mode game, or the replay has not been parsed, or the API response structure has changed."
        )
        # For debugging, you might want to see the structure if 'picks_bans' is missing
        # print("\nFull match data for inspection:")
        # print(json.dumps(match_data, indent=4))
    else:
        print("No match data to display draft information from.")


# --- Main execution ---
if __name__ == "__main__":
    match_id_to_fetch = 8285119302  # Example match ID

    # Fetch the hero ID to name mapping first
    hero_name_map = get_hero_id_to_name_map()

    if not hero_name_map:
        print("Proceeding without hero names as hero map could not be fetched.")
        # Decide if you want to exit or continue without names. For now, we'll continue.

    # Fetch the match details
    match_details = get_match_details(match_id_to_fetch)

    if match_details:
        # Display the draft information using the hero map
        display_draft_info(match_details, hero_name_map)

        # You can uncomment the following line to print the entire JSON response for the match
        # print("\n--- Full Match Details (JSON) ---")
        # print(json.dumps(match_details, indent=4))
    else:
        print(f"Could not retrieve details for match ID: {match_id_to_fetch}")
