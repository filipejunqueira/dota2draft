# dota2draft/validation.py

"""
Input validation utilities for the dota2draft package.
Provides standardized validation for user inputs to prevent injection attacks and improve robustness.
"""

import re
from typing import Any, Optional
from .exceptions import DataValidationError


def validate_match_id(match_id: Any) -> int:
    """
    Validates and converts match ID to integer.
    
    Args:
        match_id: The match ID to validate
        
    Returns:
        int: Valid match ID
        
    Raises:
        DataValidationError: If match ID is invalid
    """
    try:
        match_id_int = int(match_id)
        if match_id_int <= 0:
            raise DataValidationError(f"Match ID must be positive, got: {match_id_int}")
        if match_id_int > 999999999999:  # Reasonable upper limit
            raise DataValidationError(f"Match ID too large, got: {match_id_int}")
        return match_id_int
    except (ValueError, TypeError):
        raise DataValidationError(f"Invalid match ID format: {match_id}")


def validate_league_id(league_id: Any) -> int:
    """
    Validates and converts league ID to integer.
    
    Args:
        league_id: The league ID to validate
        
    Returns:
        int: Valid league ID
        
    Raises:
        DataValidationError: If league ID is invalid
    """
    try:
        league_id_int = int(league_id)
        if league_id_int <= 0:
            raise DataValidationError(f"League ID must be positive, got: {league_id_int}")
        if league_id_int > 999999:  # Reasonable upper limit
            raise DataValidationError(f"League ID too large, got: {league_id_int}")
        return league_id_int
    except (ValueError, TypeError):
        raise DataValidationError(f"Invalid league ID format: {league_id}")


def validate_account_id(account_id: Any) -> int:
    """
    Validates and converts account ID to integer.
    
    Args:
        account_id: The account ID to validate
        
    Returns:
        int: Valid account ID
        
    Raises:
        DataValidationError: If account ID is invalid
    """
    try:
        account_id_int = int(account_id)
        if account_id_int <= 0:
            raise DataValidationError(f"Account ID must be positive, got: {account_id_int}")
        if account_id_int > 999999999999:  # Reasonable upper limit
            raise DataValidationError(f"Account ID too large, got: {account_id_int}")
        return account_id_int
    except (ValueError, TypeError):
        raise DataValidationError(f"Invalid account ID format: {account_id}")


def validate_date_string(date_str: str) -> str:
    """
    Validates date string format (YYYY-MM-DD).
    
    Args:
        date_str: The date string to validate
        
    Returns:
        str: Valid date string
        
    Raises:
        DataValidationError: If date format is invalid
    """
    if not isinstance(date_str, str):
        raise DataValidationError(f"Date must be a string, got: {type(date_str)}")
    
    date_pattern = r'^\d{4}-\d{2}-\d{2}$'
    if not re.match(date_pattern, date_str):
        raise DataValidationError(f"Invalid date format. Expected YYYY-MM-DD, got: {date_str}")
    
    # Additional validation could include checking if it's a valid calendar date
    try:
        from datetime import datetime
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise DataValidationError(f"Invalid date: {date_str} - {e}")
    
    return date_str


def validate_hero_nickname(nickname: str) -> str:
    """
    Validates hero nickname format.
    
    Args:
        nickname: The nickname to validate
        
    Returns:
        str: Valid nickname
        
    Raises:
        DataValidationError: If nickname is invalid
    """
    if not isinstance(nickname, str):
        raise DataValidationError(f"Nickname must be a string, got: {type(nickname)}")
    
    nickname = nickname.strip()
    if not nickname:
        raise DataValidationError("Nickname cannot be empty")
    
    if len(nickname) > 50:
        raise DataValidationError(f"Nickname too long (max 50 chars), got: {len(nickname)}")
    
    # Allow alphanumeric, spaces, hyphens, underscores
    if not re.match(r'^[a-zA-Z0-9\s\-_]+$', nickname):
        raise DataValidationError(f"Nickname contains invalid characters: {nickname}")
    
    return nickname


def validate_draft_string(draft_string: str) -> str:
    """
    Validates draft string format.
    
    Args:
        draft_string: The draft string to validate
        
    Returns:
        str: Valid draft string
        
    Raises:
        DataValidationError: If draft string is invalid
    """
    if not isinstance(draft_string, str):
        raise DataValidationError(f"Draft string must be a string, got: {type(draft_string)}")
    
    draft_string = draft_string.strip()
    if not draft_string:
        raise DataValidationError("Draft string cannot be empty")
    
    if len(draft_string) > 2000:
        raise DataValidationError(f"Draft string too long (max 2000 chars), got: {len(draft_string)}")
    
    # Basic format validation - should contain pick/ban actions
    if not re.search(r'(radiant|dire)\s+(pick|ban):', draft_string, re.IGNORECASE):
        raise DataValidationError(f"Draft string doesn't contain valid pick/ban format: {draft_string[:100]}...")
    
    return draft_string


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes filename for safe file operations.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        str: Sanitized filename
        
    Raises:
        DataValidationError: If filename is invalid
    """
    if not isinstance(filename, str):
        raise DataValidationError(f"Filename must be a string, got: {type(filename)}")
    
    filename = filename.strip()
    if not filename:
        raise DataValidationError("Filename cannot be empty")
    
    # Remove or replace dangerous characters
    # Allow alphanumeric, dots, hyphens, underscores
    sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Prevent directory traversal
    if '..' in sanitized or sanitized.startswith('/') or ':' in sanitized:
        raise DataValidationError(f"Filename contains dangerous path elements: {filename}")
    
    if len(sanitized) > 255:
        raise DataValidationError(f"Filename too long (max 255 chars), got: {len(sanitized)}")
    
    return sanitized