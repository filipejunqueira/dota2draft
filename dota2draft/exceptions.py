# dota2draft/exceptions.py

"""
Custom exception classes for the dota2draft package.
Provides standardized error handling across all modules.
"""

from typing import Optional, Any


class Dota2DraftError(Exception):
    """Base exception class for all dota2draft-related errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details
        super().__init__(self.message)


class DatabaseError(Dota2DraftError):
    """Raised when database operations fail."""
    pass


class APIError(Dota2DraftError):
    """Raised when API requests fail."""
    pass


class DataValidationError(Dota2DraftError):
    """Raised when data validation fails."""
    pass


class ModelError(Dota2DraftError):
    """Raised when machine learning model operations fail."""
    pass


class ConfigurationError(Dota2DraftError):
    """Raised when configuration is invalid or missing."""
    pass


class AnalysisError(Dota2DraftError):
    """Raised when match analysis fails."""
    pass