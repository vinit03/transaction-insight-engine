"""
Transaction Analysis Package
============================

A comprehensive transaction analysis toolkit for financial data processing.

Core Modules:
- data_loader: Load and process transaction files
- analysis_engine: Perform comprehensive transaction analysis
- categorizer: Intelligent transaction categorization
- visualizer: Generate charts and visualizations
- exporter: Export analysis results to various formats
"""

from .analysis_engine import TransactionAnalyzer, analyze_transactions
from .categorizer import TransactionCategorizer, categorize_dataframe
from .data_loader import DataLoader, quick_load, batch_load
from .exporter import TransactionExporter, export_simple_excel, export_full_report
from .visualizer import TransactionVisualizer, create_visualizations

__version__ = "1.0.0"
__author__ = "Transaction Analysis Team"
__email__ = "support@transactionanalysis.com"

__all__ = [
    # Data Loading
    'DataLoader',
    'quick_load',
    'batch_load',

    # Analysis
    'TransactionAnalyzer',
    'analyze_transactions',

    # Categorization
    'TransactionCategorizer',
    'categorize_dataframe',

    # Visualization
    'TransactionVisualizer',
    'create_visualizations',

    # Export
    'TransactionExporter',
    'export_simple_excel',
    'export_full_report',
]
