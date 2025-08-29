"""
Configuration Package
=====================

Centralized configuration management for the transaction analysis system.

This module provides all configuration settings including:
- Column mappings for flexible file format support
- Performance settings for large dataset handling
- Analysis parameters and thresholds
- Export and visualization options
"""

from .settings import (
    COLUMN_MAPPINGS,
    REQUIRED_COLUMNS,
    PERFORMANCE_SETTINGS,
    CATEGORIZATION_SETTINGS,
    ANALYSIS_SETTINGS,
    FILE_SETTINGS,
    EXPORT_SETTINGS,
    DEFAULT_VALUES,
)

__all__ = [
    'COLUMN_MAPPINGS',
    'REQUIRED_COLUMNS',
    'PERFORMANCE_SETTINGS',
    'CATEGORIZATION_SETTINGS',
    'ANALYSIS_SETTINGS',
    'FILE_SETTINGS',
    'EXPORT_SETTINGS',
    'DEFAULT_VALUES',
]
