"""
Export Functionality for Transaction Analysis
Supports Excel exports for large datasets
"""

import logging
import os
import sys
import warnings
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any, Optional

import pandas as pd

warnings.filterwarnings('ignore')

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.settings import EXPORT_SETTINGS, DECIMAL_SETTINGS
from src.decimal_utils import decimal_to_float


class TransactionExporter:
    """Handle exports to Excel"""

    def __init__(self, enable_logging: bool = True):
        self.logger = self._setup_logging() if enable_logging else None
        self.export_stats = {}

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

    def _prepare_dataframe_for_export(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe for export (handle large datasets)"""
        df_export = df.copy()

        # Convert datetime columns to strings for better Excel compatibility
        for col in df_export.select_dtypes(include=['datetime64']).columns:
            df_export[col] = df_export[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Handle categorical columns
        for col in df_export.select_dtypes(include=['category']).columns:
            df_export[col] = df_export[col].astype(str)

        # Handle Decimal columns - convert to float for Excel export with proper rounding
        decimal_places = DECIMAL_SETTINGS.get('decimal_places', 2)
        for col in df_export.columns:
            if df_export[col].dtype == 'object':  # Decimal columns are stored as object dtype
                # Check if this column contains Decimal values
                sample_values = df_export[col].dropna().head(5)
                if not sample_values.empty and all(isinstance(val, Decimal) for val in sample_values):
                    # Convert Decimal to float with proper rounding for Excel
                    df_export[col] = df_export[col].apply(
                        lambda x: round(decimal_to_float(x), decimal_places) if isinstance(x, Decimal) else x
                    )
                    self._log('info', f"Converted Decimal column '{col}' to float for Excel export")

        # Limit decimal places for remaining float columns
        for col in df_export.select_dtypes(include=['float']).columns:
            df_export[col] = df_export[col].round(decimal_places)

        return df_export

    def export_to_excel(self, df: pd.DataFrame, analysis_results: Dict[str, Any],
                        output_path: str) -> str:
        """
        Export comprehensive Excel report with multiple sheets
        
        Args:
            df: Main transaction dataframe
            analysis_results: Analysis results dictionary
            output_path: Output file path
            
        Returns:
            Path to generated Excel file
        """
        self._log('info', f"Starting Excel export to {output_path}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Prepare main dataframe
        df_export = self._prepare_dataframe_for_export(df)

        # Create Excel writer with optimal settings for large files
        excel_engine = EXPORT_SETTINGS.get('excel_engine', 'xlsxwriter')

        try:
            with pd.ExcelWriter(output_path, engine=excel_engine) as writer:

                # Sheet 1: Raw Transaction Data
                max_rows = EXPORT_SETTINGS.get('max_rows_per_sheet', 1000000)
                if len(df_export) > max_rows:
                    self._log('warning', f"Large dataset ({len(df_export):,} rows) - splitting into multiple sheets")

                    # Split into multiple sheets
                    for i, start_idx in enumerate(range(0, len(df_export), max_rows)):
                        end_idx = min(start_idx + max_rows, len(df_export))
                        df_chunk = df_export.iloc[start_idx:end_idx]
                        sheet_name = f'Raw_Data_{i + 1}' if i > 0 else 'Raw_Data'
                        df_chunk.to_excel(writer, sheet_name=sheet_name, index=False)
                        self._log('info', f"✓ Exported {len(df_chunk):,} rows to sheet '{sheet_name}'")
                else:
                    df_export.to_excel(writer, sheet_name='Raw_Data', index=False)
                    self._log('info', f"✓ Exported {len(df_export):,} rows to 'Raw_Data' sheet")

                # Additional sheets
                self._export_analysis_sheets(writer, analysis_results, len(df))

            self._log('info', f"Excel export completed: {output_path}")
            return output_path

        except Exception as e:
            self._log('error', f"Error during Excel export: {e}")
            raise

    def _export_analysis_sheets(self, writer: pd.ExcelWriter, analysis_results: Dict[str, Any],
                                total_records: int) -> None:
        """Export all analysis results to separate Excel sheets"""

        # Executive Summary
        try:
            summary_data = [
                {'Metric': 'Analysis Date', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                {'Metric': 'Total Records Analyzed', 'Value': f"{total_records:,}"},
            ]

            # Add key metrics from analysis
            if 'basic_statistics' in analysis_results:
                basic_stats = analysis_results['basic_statistics']
                if 'dataset_overview' in basic_stats:
                    overview = basic_stats['dataset_overview']
                    summary_data.extend([
                        {'Metric': 'Unique Accounts', 'Value': overview.get('unique_accounts', 0)},
                        {'Metric': 'Unique Merchants', 'Value': overview.get('unique_merchants', 0)},
                        {'Metric': 'Date Range (Days)', 'Value': overview.get('date_span_days', 0)},
                    ])

                if 'amount_analysis' in basic_stats:
                    amount_stats = basic_stats['amount_analysis']
                    summary_data.extend([
                        {'Metric': 'Total Credits', 'Value': f"£{amount_stats.get('total_credits', 0):,.2f}"},
                        {'Metric': 'Total Debits', 'Value': f"£{amount_stats.get('total_debits', 0):,.2f}"},
                        {'Metric': 'Net Position', 'Value': f"£{amount_stats.get('net_position', 0):,.2f}"},
                    ])

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False)
            self._log('info', "✓ Executive summary sheet exported")

        except Exception as e:
            self._log('error', f"Error creating executive summary: {e}")

        # Account Analysis
        if 'account_analysis' in analysis_results:
            try:
                account_data = analysis_results['account_analysis']
                if 'account_details' in account_data:
                    account_details = []

                    for account_info in account_data['account_details']:
                        row = {'account_number': account_info.get('account_number')}
                        row.update(account_info.get('financial_summary', {}))
                        row.update(account_info.get('activity_status', {}))
                        account_details.append(row)

                    if account_details:
                        account_df = pd.DataFrame(account_details)
                        account_df.to_excel(writer, sheet_name='Account_Analysis', index=False)
                        self._log('info', "✓ Account analysis sheet exported")

            except Exception as e:
                self._log('error', f"Error exporting account analysis: {e}")

        # Merchant Analysis
        if 'merchant_analysis' in analysis_results:
            try:
                merchant_data = analysis_results['merchant_analysis']

                # Top merchants by frequency
                if 'merchant_frequency' in merchant_data and 'top_merchants' in merchant_data['merchant_frequency']:
                    freq_data = []
                    for merchant, count in merchant_data['merchant_frequency']['top_merchants'].items():
                        freq_data.append({'Merchant': merchant, 'Transaction_Count': count})

                    if freq_data:
                        freq_df = pd.DataFrame(freq_data)
                        freq_df.to_excel(writer, sheet_name='Top_Merchants', index=False)
                        self._log('info', "✓ Merchant analysis sheet exported")

            except Exception as e:
                self._log('error', f"Error exporting merchant analysis: {e}")

        # Category Analysis
        if 'category_analysis' in analysis_results:
            try:
                category_data = analysis_results['category_analysis']
                if 'category_financial_summary' in category_data:
                    category_summary = []

                    for category, metrics in category_data['category_financial_summary'].items():
                        row = {'category': category}
                        row.update(metrics)
                        category_summary.append(row)

                    if category_summary:
                        category_df = pd.DataFrame(category_summary)
                        category_df.to_excel(writer, sheet_name='Category_Analysis', index=False)
                        self._log('info', "✓ Category analysis sheet exported")

            except Exception as e:
                self._log('error', f"Error exporting category analysis: {e}")

        # Active/Inactive Accounts Analysis
        if 'active_inactive_accounts' in analysis_results:
            try:
                accounts_data = analysis_results['active_inactive_accounts']
                if 'account_details' in accounts_data:
                    # Create a clean dataframe for the Excel export
                    accounts_list = []

                    for account_info in accounts_data['account_details']:
                        row = {
                            'account_number': account_info.get('account_number'),
                            'status': account_info.get('status'),
                            'total_transactions': account_info.get('total_transactions'),
                            'recent_transactions_90_days': account_info.get('recent_transactions_90_days'),
                            'days_since_last_transaction': account_info.get('days_since_last_transaction'),
                            'last_transaction_date': account_info.get('last_transaction_date'),
                            'first_transaction_date': account_info.get('first_transaction_date'),
                            'total_amount': account_info.get('total_amount'),
                            'avg_transaction_amount': account_info.get('avg_transaction_amount')
                        }
                        accounts_list.append(row)

                    if accounts_list:
                        accounts_df = pd.DataFrame(accounts_list)

                        # Format dates
                        if 'last_transaction_date' in accounts_df.columns:
                            accounts_df['last_transaction_date'] = pd.to_datetime(
                                accounts_df['last_transaction_date']).dt.strftime('%Y-%m-%d')
                        if 'first_transaction_date' in accounts_df.columns:
                            accounts_df['first_transaction_date'] = pd.to_datetime(
                                accounts_df['first_transaction_date']).dt.strftime('%Y-%m-%d')

                        accounts_df.to_excel(writer, sheet_name='Active_Inactive_Accounts', index=False)
                        self._log('info', "✓ Active/Inactive accounts sheet exported")

            except Exception as e:
                self._log('error', f"Error exporting active/inactive accounts analysis: {e}")

    def export_simple_report(self, df: pd.DataFrame, output_path: str) -> str:
        """
        Export simple Excel report with just the data and basic summary
        
        Args:
            df: Transaction dataframe
            output_path: Output file path
            
        Returns:
            Path to generated Excel file
        """
        self._log('info', f"Creating simple Excel report: {output_path}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Prepare dataframe
        df_export = self._prepare_dataframe_for_export(df)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main data
            df_export.to_excel(writer, sheet_name='Transaction_Data', index=False)

            # Simple summary
            if 'amount' in df.columns:
                summary_data = [
                    {'Metric': 'Total Transactions', 'Value': len(df)},
                    {'Metric': 'Total Credits',
                     'Value': df[df['amount'] > 0]['amount'].sum() if (df['amount'] > 0).any() else 0},
                    {'Metric': 'Total Debits',
                     'Value': abs(df[df['amount'] < 0]['amount'].sum()) if (df['amount'] < 0).any() else 0},
                    {'Metric': 'Net Position', 'Value': df['amount'].sum()},
                ]

                if 'account_number' in df.columns:
                    summary_data.append({'Metric': 'Unique Accounts', 'Value': df['account_number'].nunique()})

                if 'main_description' in df.columns:
                    summary_data.append({'Metric': 'Unique Merchants', 'Value': df['main_description'].nunique()})

                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

        self._log('info', f"Simple Excel report completed: {output_path}")
        return output_path

    def export_pdf_report(self, df: pd.DataFrame, analysis_results: Dict[str, Any],
                         chart_paths: List[str], output_path: str) -> str:
        """
        Export comprehensive PDF report combining all analysis results and charts
        
        Args:
            df: Main transaction dataframe
            analysis_results: Analysis results dictionary
            chart_paths: List of paths to generated charts
            output_path: Output PDF file path
            
        Returns:
            Path to generated PDF file
        """
        try:
            # Import PDF exporter (lazy import to avoid issues if reportlab not installed)
            from src.pdf_exporter import TransactionPDFExporter
            
            self._log('info', f"Creating PDF report: {output_path}")
            
            # Create PDF exporter and generate report
            pdf_exporter = TransactionPDFExporter(enable_logging=False)
            result_path = pdf_exporter.export_consolidated_pdf(
                df=df,
                analysis_results=analysis_results,
                chart_paths=chart_paths,
                output_path=output_path
            )
            
            self._log('info', f"✅ PDF report completed: {result_path}")
            return result_path
            
        except ImportError as e:
            self._log('error', f"PDF export requires reportlab library: {e}")
            raise ImportError("PDF export functionality requires 'reportlab' package. Please install it.")
        except Exception as e:
            self._log('error', f"Error creating PDF report: {e}")
            raise


def export_simple_excel(df: pd.DataFrame, output_path: str) -> str:
    """Convenience function for simple Excel export"""
    exporter = TransactionExporter()
    return exporter.export_simple_report(df, output_path)


def export_full_report(df: pd.DataFrame, analysis_results: Dict[str, Any],
                       base_filename: str = 'transaction_analysis',
                       chart_paths: Optional[List[str]] = None) -> Dict[str, str]:
    """Convenience function for full report export"""
    exporter = TransactionExporter()
    return exporter.export_comprehensive_report(df, analysis_results, base_filename, chart_paths)
