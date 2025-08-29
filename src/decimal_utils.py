"""
Decimal utilities for precise financial calculations
Provides helper functions for working with Decimal in pandas and analysis
"""

import os
import sys
from decimal import Decimal, getcontext
from typing import Any, Union, Optional, List

import pandas as pd

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.settings import DECIMAL_SETTINGS


def configure_decimal_context():
    """Configure global decimal context for financial calculations"""
    getcontext().prec = DECIMAL_SETTINGS.get('precision', 28)
    getcontext().rounding = getattr(__import__('decimal'), DECIMAL_SETTINGS.get('rounding', 'ROUND_HALF_UP'))


def to_decimal(value: Any) -> Optional[Decimal]:
    """
    Convert a value to Decimal safely
    
    Args:
        value: Value to convert (str, int, float, or Decimal)
        
    Returns:
        Decimal or None if conversion fails
    """
    if value is None or pd.isna(value):
        return None

    if isinstance(value, Decimal):
        return value

    try:
        # Convert to string first to avoid float precision issues
        if isinstance(value, float):
            return Decimal(str(value))
        else:
            return Decimal(value)
    except (ValueError, TypeError):
        return None


def decimal_series(series: pd.Series) -> pd.Series:
    """
    Convert pandas Series to Decimal values
    
    Args:
        series: Pandas Series with numeric values
        
    Returns:
        Series with Decimal values
    """
    return series.apply(to_decimal)


def decimal_sum(values: Union[List[Decimal], pd.Series]) -> Decimal:
    """
    Sum Decimal values safely
    
    Args:
        values: List or Series of Decimal values
        
    Returns:
        Sum as Decimal
    """
    if isinstance(values, pd.Series):
        values = values.dropna().tolist()

    return sum(val for val in values if val is not None)


def decimal_mean(values: Union[List[Decimal], pd.Series]) -> Optional[Decimal]:
    """
    Calculate mean of Decimal values
    
    Args:
        values: List or Series of Decimal values
        
    Returns:
        Mean as Decimal or None if no valid values
    """
    if isinstance(values, pd.Series):
        values = values.dropna().tolist()

    valid_values = [val for val in values if val is not None]

    if not valid_values:
        return None

    return decimal_sum(valid_values) / Decimal(len(valid_values))


def decimal_min(values: Union[List[Decimal], pd.Series]) -> Optional[Decimal]:
    """
    Find minimum of Decimal values
    
    Args:
        values: List or Series of Decimal values
        
    Returns:
        Minimum as Decimal or None if no valid values
    """
    if isinstance(values, pd.Series):
        values = values.dropna().tolist()

    valid_values = [val for val in values if val is not None]
    return min(valid_values) if valid_values else None


def decimal_max(values: Union[List[Decimal], pd.Series]) -> Optional[Decimal]:
    """
    Find maximum of Decimal values
    
    Args:
        values: List or Series of Decimal values
        
    Returns:
        Maximum as Decimal or None if no valid values
    """
    if isinstance(values, pd.Series):
        values = values.dropna().tolist()

    valid_values = [val for val in values if val is not None]
    return max(valid_values) if valid_values else None


def decimal_abs(value: Decimal) -> Decimal:
    """
    Absolute value of Decimal
    
    Args:
        value: Decimal value
        
    Returns:
        Absolute value as Decimal
    """
    return abs(value) if value is not None else None


def decimal_to_float(value: Optional[Decimal]) -> Optional[float]:
    """
    Convert Decimal to float for visualization/export
    Use only when precision loss is acceptable
    
    Args:
        value: Decimal value
        
    Returns:
        Float value or None
    """
    return float(value) if value is not None else None


def decimal_comparison_ops(df: pd.DataFrame, col: str, value: Union[Decimal, float, int], op: str) -> pd.Series:
    """
    Perform comparison operations on Decimal column
    
    Args:
        df: DataFrame
        col: Column name with Decimal values
        value: Value to compare against
        op: Operation ('>', '<', '>=', '<=', '==', '!=')
        
    Returns:
        Boolean Series
    """
    decimal_value = to_decimal(value)
    if decimal_value is None:
        return pd.Series([False] * len(df))

    if op == '>':
        return df[col].apply(lambda x: x > decimal_value if x is not None else False)
    elif op == '<':
        return df[col].apply(lambda x: x < decimal_value if x is not None else False)
    elif op == '>=':
        return df[col].apply(lambda x: x >= decimal_value if x is not None else False)
    elif op == '<=':
        return df[col].apply(lambda x: x <= decimal_value if x is not None else False)
    elif op == '==':
        return df[col].apply(lambda x: x == decimal_value if x is not None else False)
    elif op == '!=':
        return df[col].apply(lambda x: x != decimal_value if x is not None else False)
    else:
        raise ValueError(f"Unsupported operation: {op}")


# Configure decimal context when module is imported
configure_decimal_context()
