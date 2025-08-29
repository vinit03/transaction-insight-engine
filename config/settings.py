"""
Configuration for Transaction Analysis
"""

# Column mapping arrays - easily extendable for new file formats
COLUMN_MAPPINGS = {
    'account_number': ['AccNo'],
    'transaction_date': ['Date'],
    'main_description': ['MainDesc'],
    'additional_description': ['AddDesc'],
    'transaction_type': ['TransactionType'],
    'amount': ['Amount'],
    'balance': ['Balance'],
    'spend_category': ['SpendCategory'],
    'currency': ['Currency']
}

# Required columns (analysis will fail if these are missing)
REQUIRED_COLUMNS = ['account_number', 'transaction_date', 'main_description', 'amount']

# Performance settings for large datasets
PERFORMANCE_SETTINGS = {
    'chunk_size': 50000,  # Process large CSV files in chunks
    'low_memory': True,  # Use low memory mode for pandas
    'use_categorical': True,  # Convert strings to categories to save memory
    'dtype_optimization': True,  # Optimize data types automatically
}

# Transaction categorization settings
CATEGORIZATION_SETTINGS = {
    'min_frequency_for_category': 5,  # Minimum transactions to create category
    'similarity_threshold': 0.85,  # Threshold for grouping similar merchant names
    'enable_smart_grouping': True,  # Group similar merchants automatically
    'max_auto_categories': 100,  # Maximum number of auto-generated categories
    'exclude_one_off_transactions': True,  # Don't categorize unique transactions
}

# Analysis settings
ANALYSIS_SETTINGS = {
    'large_transaction_percentile': 95,  # Percentile threshold for outlier detection
    'active_account_days': 90,  # Days threshold for active account classification
    'top_merchants_count': 15,  # Number of top merchants to show in reports
}

# Decimal precision settings for financial calculations
DECIMAL_SETTINGS = {
    'precision': 28,  # Total number of significant digits
    'rounding': 'ROUND_HALF_UP',  # Rounding method for financial calculations
    'decimal_places': 2,  # Default decimal places for currency display
    'use_decimal': True,  # Enable decimal arithmetic throughout the system
}

# File processing settings
FILE_SETTINGS = {
    'max_file_size_mb': 1000,  # Maximum file size supported
    'header_row': 0,  # Row number containing headers (0-based)
}

# Export settings
EXPORT_SETTINGS = {
    'excel_engine': 'xlsxwriter',  # Excel engine for better performance
    'chart_dpi': 150,  # Chart resolution
    'max_rows_per_sheet': 1000000,  # Maximum rows per Excel sheet
}

# Default values for missing data
DEFAULT_VALUES = {
    'currency': 'GBP',
    'transaction_type': 'UNKNOWN',
    'spend_category': 'UNCATEGORIZED',
    'additional_description': ''
}
