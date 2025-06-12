import typer
import requests
import json
from rich.console import Console
from graphql import get_introspection_query, build_client_schema, print_schema

console = Console()

# --- Configuration ---
STRATZ_API_URL = "https://api.stratz.com/graphql"
# IMPORTANT: Replace this with your actual Stratz API token
STRATZ_API_TOKEN = "YOUR_STRATZ_API_TOKEN_HERE"

HEADERS = {
    "User-Agent": "STRATZ_API",
    "Authorization": f"Bearer {STRATZ_API_TOKEN}"
}

# --- Introspection and Query Building ---

def get_stratz_schema():
    """Fetches the GraphQL schema from Stratz using an introspection query."""
    console.print("Fetching Stratz API schema...")
    try:
        introspection_query = get_introspection_query(descriptions=False)
        response = requests.post(STRATZ_API_URL, headers=HEADERS, json={'query': introspection_query})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Schema Fetch Failed:[/bold red] {e}")
        raise typer.Exit(code=1)

def build_query_for_type(schema_types, type_name, visited_types=None, max_depth=5):
    """Recursively builds a query string for a given type from the schema."""
    if visited_types is None:
        visited_types = set()

    if type_name in visited_types or max_depth <= 0:
        return ""

    visited_types.add(type_name)

    query_parts = []
    type_info = next((t for t in schema_types if t['name'] == type_name and t['kind'] == 'OBJECT'), None)

    if not type_info or not type_info.get('fields'):
        visited_types.remove(type_name)
        return ""

    for field in type_info['fields']:
        # Skip fields that require arguments, for simplicity
        if field.get('args') and len(field['args']) > 0:
            continue

        field_type = field['type']
        while field_type.get('ofType'):
            field_type = field_type['ofType']

        if field_type['kind'] == 'SCALAR' or field_type['kind'] == 'ENUM':
            query_parts.append(field['name'])
        elif field_type['kind'] == 'OBJECT':
            sub_query = build_query_for_type(schema_types, field_type['name'], visited_types.copy(), max_depth - 1)
            if sub_query:
                query_parts.append(f"{field['name']} {{ {sub_query} }}")

    visited_types.remove(type_name)
    return ' '.join(query_parts)

# --- Main Application ---

app = typer.Typer()

@app.command()
def fetch_match(
    match_id: int = typer.Argument(..., help="The numeric ID of the match to fetch."),
    output_file: str = typer.Option(None, "--out", "-o", help="Optional output file name."),
    max_depth: int = typer.Option(5, help="Max recursion depth for query generation.")
):
    """Dynamically builds a full query to fetch all possible data for a match and saves it."""
    if STRATZ_API_TOKEN == "YOUR_STRATZ_API_TOKEN_HERE":
        console.print("[bold red]Error:[/bold red] Please replace 'YOUR_STRATZ_API_TOKEN_HERE' in the script.")
        raise typer.Exit(code=1)

    schema_response = get_stratz_schema()
    if 'errors' in schema_response:
        console.print(f"[bold red]Error fetching schema:[/bold red] {schema_response['errors']}")
        raise typer.Exit(code=1)

    schema_types = schema_response['data']['__schema']['types']
    
    console.print("Building dynamic query from schema...")
    match_fields_query = build_query_for_type(schema_types, 'MatchType', max_depth=max_depth)
    
    if not match_fields_query:
        console.print("[bold red]Failed to build a query from the schema.[/bold red]")
        raise typer.Exit(code=1)

    final_query = f"query GetFullMatchDetails($matchId: Long!) {{ match(id: $matchId) {{ {match_fields_query} }} }}"

    console.print(f"Fetching data for match ID: [cyan]{match_id}[/cyan]...")
    try:
        response = requests.post(
            STRATZ_API_URL,
            headers=HEADERS,
            json={"query": final_query, "variables": {"matchId": match_id}}
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]API Request Failed:[/bold red] {e}")
        raise typer.Exit(code=1)

    data = response.json()

    if "errors" in data:
        console.print("[bold red]API Error:[/bold red]")
        for error in data["errors"]:
            console.print(f"- {error.get('message', 'Unknown error')}")
        raise typer.Exit(code=1)

    if not output_file:
        output_file = f"stratz_match_{match_id}_full.json"

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        console.print(f"[bold green]Success![/bold green] Full match data saved to [cyan]{output_file}[/cyan]")
    except IOError as e:
        console.print(f"[bold red]File Write Error:[/bold red] {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()

