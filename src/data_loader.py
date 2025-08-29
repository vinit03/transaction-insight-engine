"""
Adaptive Data Loader for Transaction Analysis
Handles large files (1M+ records) with flexible column mapping
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import psutil

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.settings import (
    COLUMN_MAPPINGS, REQUIRED_COLUMNS, PERFORMANCE_SETTINGS, FILE_SETTINGS, DEFAULT_VALUES, DECIMAL_SETTINGS
)
from src.decimal_utils import decimal_series


class DataLoader:
    """High-performance data loader with adaptive column mapping"""

    def __init__(self, enable_logging: bool = True):
        self.logger = self._setup_logging() if enable_logging else None
        self.column_map = {}
        self.data_info = {}
        self.memory_usage_mb = 0

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _log(self, level: str, message: str) -> None:
        """Safe logging method"""
        if self.logger:
            getattr(self.logger, level.lower())(message)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def _detect_columns(self, df_sample: pd.DataFrame) -> Dict[str, str]:
        """
        Detect and map columns based on configuration
        Returns mapping of standard_name -> actual_column_name
        """
        column_map = {}
        available_columns = df_sample.columns.tolist()

        for standard_name, possible_names in COLUMN_MAPPINGS.items():
            found_column = None

            # Find first matching column name
            for possible_name in possible_names:
                if possible_name in available_columns:
                    found_column = possible_name
                    break

            if found_column:
                column_map[standard_name] = found_column
                self._log('debug', f"Mapped '{standard_name}' -> '{found_column}'")
            else:
                self._log('warning', f"Column '{standard_name}' not found. Tried: {possible_names}")

        return column_map

    def _validate_required_columns(self, column_map: Dict[str, str]) -> bool:
        """Validate that all required columns are present"""
        missing_columns = []

        for required_col in REQUIRED_COLUMNS:
            if required_col not in column_map:
                missing_columns.append(required_col)

        if missing_columns:
            self._log('error', f"Missing required columns: {missing_columns}")
            return False

        self._log('info', f"All required columns found: {REQUIRED_COLUMNS}")
        return True

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        if not PERFORMANCE_SETTINGS.get('dtype_optimization', True):
            return df

        start_memory = df.memory_usage(deep=True).sum() / 1024 ** 2

        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if df[col].isnull().all():
                continue

            col_min = df[col].min()
            col_max = df[col].max()

            # Optimize integers
            if df[col].dtype == 'int64':
                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df[col] = df[col].astype('int32')

            # Optimize floats
            elif df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')

        # Convert strings to categories if beneficial
        if PERFORMANCE_SETTINGS.get('use_categorical', True):
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')

        end_memory = df.memory_usage(deep=True).sum() / 1024 ** 2
        self._log('info', f"Memory optimization: {start_memory:.2f}MB -> {end_memory:.2f}MB "
                          f"(saved {start_memory - end_memory:.2f}MB)")

        return df

    def _process_dates(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Process and validate date column"""
        if date_column not in df.columns:
            self._log('warning', f"Date column '{date_column}' not found")
            return df

        try:
            # Try to convert to datetime
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

            # Check for invalid dates
            invalid_dates = df[date_column].isnull().sum()
            if invalid_dates > 0:
                self._log('warning', f"Found {invalid_dates} invalid dates")

            # Validate date range
            min_date = df[date_column].min()
            max_date = df[date_column].max()
            self._log('info', f"Date range: {min_date} to {max_date}")

            # Check for future dates (potential data issue)
            future_dates = df[df[date_column] > datetime.now()].shape[0]
            if future_dates > 0:
                self._log('warning', f"Found {future_dates} future dates - possible data issue")

        except Exception as e:
            self._log('error', f"Error processing dates: {e}")

        return df

    def _clean_and_standardize(self, df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
        """Clean and standardize the data"""

        # Rename columns to standard names
        reverse_map = {v: k for k, v in column_map.items()}
        df = df.rename(columns=reverse_map)

        # Process dates
        if 'transaction_date' in df.columns:
            df = self._process_dates(df, 'transaction_date')

        # Clean amount column and convert to Decimal
        if 'amount' in df.columns:
            # First ensure numeric for conversion
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

            # Convert to Decimal for precise financial calculations
            if DECIMAL_SETTINGS.get('use_decimal', True):
                df['amount'] = decimal_series(df['amount'])
                self._log('info', "Converted amount column to Decimal for precise calculations")

            # Check for null amounts
            null_amounts = df['amount'].isnull().sum()
            if null_amounts > 0:
                self._log('warning', f"Found {null_amounts} null amounts - will be excluded from analysis")

        # Clean balance column and convert to Decimal
        if 'balance' in df.columns:
            # First ensure numeric for conversion
            df['balance'] = pd.to_numeric(df['balance'], errors='coerce')

            # Convert to Decimal for precise financial calculations
            if DECIMAL_SETTINGS.get('use_decimal', True):
                df['balance'] = decimal_series(df['balance'])
                self._log('info', "Converted balance column to Decimal for precise calculations")

        # Clean description columns
        for desc_col in ['main_description', 'additional_description']:
            if desc_col in df.columns:
                df[desc_col] = df[desc_col].astype(str).str.strip()
                # Replace 'nan' strings with actual NaN
                df[desc_col] = df[desc_col].replace('nan', np.nan)

        # Fill missing values with defaults
        for col, default_value in DEFAULT_VALUES.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_value)

        # Sort by date and account for better performance
        if 'transaction_date' in df.columns and 'account_number' in df.columns:
            df = df.sort_values(['account_number', 'transaction_date']).reset_index(drop=True)

        return df

    def load_file(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load and process a single file
        
        Args:
            file_path: Path to the file
            sheet_name: Excel sheet name (None for first sheet)
            
        Returns:
            Processed DataFrame
        """
        start_time = time.time()
        file_path = Path(file_path)

        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size_mb = file_path.stat().st_size / 1024 / 1024
        if file_size_mb > FILE_SETTINGS['max_file_size_mb']:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB > {FILE_SETTINGS['max_file_size_mb']}MB")

        self._log('info', f"Loading file: {file_path} ({file_size_mb:.1f}MB)")

        # Determine file type and load
        file_extension = file_path.suffix.lower()

        try:
            if file_extension in ['.xlsx', '.xls']:
                df = self._load_excel(file_path, sheet_name)
            elif file_extension in ['.csv', '.tsv']:
                df = self._load_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            self._log('info', f"Loaded {len(df):,} rows, {len(df.columns)} columns")

            # Detect and map columns
            self.column_map = self._detect_columns(df)

            # Validate required columns
            if not self._validate_required_columns(self.column_map):
                raise ValueError("Required columns missing")

            # Clean and standardize data
            df = self._clean_and_standardize(df, self.column_map)

            # Optimize data types
            df = self._optimize_dtypes(df)

            # Store data info
            self.data_info = {
                'file_path': str(file_path),
                'rows': len(df),
                'columns': len(df.columns),
                'date_range': self._get_date_range(df),
                'accounts': self._get_accounts(df),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024 ** 2,
                'load_time_seconds': time.time() - start_time
            }

            self._log('info', f"Processing completed in {self.data_info['load_time_seconds']:.2f}s")
            self._log('info', f"Memory usage: {self.data_info['memory_mb']:.2f}MB")

            return df

        except Exception as e:
            self._log('error', f"Error loading file {file_path}: {e}")
            raise

    def _load_excel(self, file_path: Path, sheet_name: Optional[str]) -> pd.DataFrame:
        """Load Excel file with performance optimizations"""

        read_params = {
            'engine': 'openpyxl',
            'sheet_name': sheet_name or 0,
            'header': FILE_SETTINGS.get('header_row', 0)
        }

        # For large files, use chunking
        if file_path.stat().st_size > 50 * 1024 * 1024:  # 50MB
            self._log('info', "Large file detected - using chunked reading")
            # Note: Excel doesn't support chunking directly, but we can optimize memory
            read_params['dtype'] = 'object'  # Let pandas infer later

        return pd.read_excel(file_path, **read_params)

    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file with performance optimizations"""

        # Detect separator
        separator = '\t' if file_path.suffix.lower() == '.tsv' else ','

        read_params = {
            'sep': separator,
            'header': FILE_SETTINGS.get('header_row', 0),
            'low_memory': PERFORMANCE_SETTINGS.get('low_memory', True),
            'dtype': 'object'  # Let us handle type conversion
        }

        # For very large files, use chunking
        if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
            chunk_size = PERFORMANCE_SETTINGS.get('chunk_size', 50000)
            self._log('info', f"Large CSV detected - using chunked reading (chunk_size={chunk_size})")

            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, **read_params):
                chunks.append(chunk)

            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.read_csv(file_path, **read_params)

    def _get_date_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get date range information"""
        if 'transaction_date' not in df.columns:
            return {}

        return {
            'min_date': df['transaction_date'].min(),
            'max_date': df['transaction_date'].max(),
            'span_days': (df['transaction_date'].max() - df['transaction_date'].min()).days,
            'unique_dates': df['transaction_date'].nunique()
        }

    def _get_accounts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get account information"""
        if 'account_number' not in df.columns:
            return {}

        return {
            'unique_accounts': df['account_number'].nunique(),
            'account_list': df['account_number'].unique().tolist(),
            'transactions_per_account': df['account_number'].value_counts().to_dict()
        }

    def load_multiple_files(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Load and merge multiple files
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            Combined DataFrame
        """
        if not file_paths:
            raise ValueError("No files provided")

        self._log('info', f"Loading {len(file_paths)} files")

        dataframes = []
        total_rows = 0

        for file_path in file_paths:
            try:
                df = self.load_file(file_path)
                dataframes.append(df)
                total_rows += len(df)
                self._log('info', f"Loaded {file_path}: {len(df):,} rows")

            except Exception as e:
                self._log('error', f"Failed to load {file_path}: {e}")
                continue

        if not dataframes:
            raise ValueError("No files were successfully loaded")

        # Combine all dataframes
        self._log('info', f"Combining {len(dataframes)} files with {total_rows:,} total rows")
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Re-sort combined data
        if 'transaction_date' in combined_df.columns and 'account_number' in combined_df.columns:
            combined_df = combined_df.sort_values(['account_number', 'transaction_date']).reset_index(drop=True)

        self._log('info', f"Combined dataset: {len(combined_df):,} rows")

        return combined_df

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data"""
        return self.data_info.copy() if self.data_info else {}


# Utility functions for external use
def quick_load(file_path: str) -> pd.DataFrame:
    """Quick load function for simple use cases"""
    loader = DataLoader()
    return loader.load_file(file_path)


def batch_load(file_paths: List[str]) -> pd.DataFrame:
    """Batch load function for multiple files"""
    loader = DataLoader()
    return loader.load_multiple_files(file_paths)
