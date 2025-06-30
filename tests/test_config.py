"""Tests for configuration management."""

import pytest
from dota2draft.config_loader import _deep_merge_config


class TestConfigMerging:
    
    def test_simple_merge(self):
        default = {"a": 1, "b": 2}
        override = {"a": 10, "c": 3}
        
        result = _deep_merge_config(default, override)
        
        assert result == {"a": 10, "b": 2, "c": 3}
    
    def test_nested_merge(self):
        default = {
            "top": 1,
            "nested": {
                "a": 1,
                "b": 2,
                "deep": {"x": 1, "y": 2}
            }
        }
        
        override = {
            "top": 10,
            "nested": {
                "b": 20,
                "c": 3,
                "deep": {"y": 20, "z": 3}
            },
            "new": "value"
        }
        
        result = _deep_merge_config(default, override)
        
        expected = {
            "top": 10,
            "nested": {
                "a": 1,
                "b": 20,
                "c": 3,
                "deep": {"x": 1, "y": 20, "z": 3}
            },
            "new": "value"
        }
        
        assert result == expected
    
    def test_override_with_different_type(self):
        default = {"value": {"nested": True}}
        override = {"value": "string"}
        
        result = _deep_merge_config(default, override)
        
        assert result == {"value": "string"}
    
    def test_empty_override(self):
        default = {"a": 1, "b": 2}
        override = {}
        
        result = _deep_merge_config(default, override)
        
        assert result == {"a": 1, "b": 2}
    
    def test_empty_default(self):
        default = {}
        override = {"a": 1, "b": 2}
        
        result = _deep_merge_config(default, override)
        
        assert result == {"a": 1, "b": 2}