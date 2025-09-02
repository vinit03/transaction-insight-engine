#!/usr/bin/env python3
"""
Transaction Analysis Tool
Generic transaction analysis for any business type with scalable architecture

Usage:
    python transaction_analyzer.py --input file.xlsx [options]
    python transaction_analyzer.py --input file1.xlsx file2.xlsx --merge [options]

Examples:
    # Basic analysis
    python transaction_analyzer.py --input inputs/statements.xlsx
    
    # With date filtering
    python transaction_analyzer.py --input inputs/statements.xlsx --start-date 2024-01-01 --end-date 2024-12-31
    
    # Multiple files
    python transaction_analyzer.py --input file1.xlsx file2.xlsx --merge
    
    # Specific accounts only
    python transaction_analyzer.py --input inputs/statements.xlsx --accounts 3995 5968
    
    # Generate per-account reports
    python transaction_analyzer.py --input inputs/statements.xlsx --separate-accounts
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from src import (
        DataLoader, TransactionCategorizer, TransactionAnalyzer,
        TransactionVisualizer, TransactionExporter
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required packages are installed:")
    print("pip install pandas openpyxl matplotlib seaborn psutil")
    sys.exit(1)


class TransactionAnalysisApp:
    """Main application class for transaction analysis"""

    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

    def setup_logging(self):
        """Setup application logging"""
        # Create logs directory
        os.makedirs('logs', exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/transaction_analysis_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def validate_inputs(self, args):
        """Validate command line arguments"""
        # Check input files exist
        for file_path in args.input:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Input file not found: {file_path}")

            # Check file extension
            if Path(file_path).suffix.lower() not in ['.xlsx', '.xls', '.csv']:
                raise ValueError(f"Unsupported file format: {file_path}")

        # Validate date formats
        if args.start_date:
            try:
                datetime.strptime(args.start_date, '%Y-%m-%d')
            except ValueError:
                raise ValueError("Start date must be in format YYYY-MM-DD")

        if args.end_date:
            try:
                datetime.strptime(args.end_date, '%Y-%m-%d')
            except ValueError:
                raise ValueError("End date must be in format YYYY-MM-DD")

        # Check date logic
        if args.start_date and args.end_date:
            start = datetime.strptime(args.start_date, '%Y-%m-%d')
            end = datetime.strptime(args.end_date, '%Y-%m-%d')
            if start > end:
                raise ValueError("Start date must be before end date")

    def load_data(self, args) -> pd.DataFrame:
        """Load transaction data from input files"""
        self.logger.info(f"Loading data from {len(args.input)} file(s)")

        # Initialize data loader
        loader = DataLoader(enable_logging=True)

        # Load single or multiple files
        if len(args.input) == 1:
            df = loader.load_file(args.input[0])
        else:
            if not args.merge:
                raise ValueError("Multiple files provided but --merge not specified")
            df = loader.load_multiple_files(args.input)

        self.logger.info(f"Successfully loaded {len(df):,} transactions")

        # Filter by accounts if specified
        if args.accounts and 'account_number' in df.columns:
            original_count = len(df)
            account_filter = [int(acc) for acc in args.accounts]
            df = df[df['account_number'].isin(account_filter)]
            self.logger.info(f"Account filtering: {original_count:,} -> {len(df):,} transactions")

        return df

    def run_categorization(self, df: pd.DataFrame, args) -> pd.DataFrame:
        """Run transaction categorization"""
        if args.skip_categorization:
            self.logger.info("Skipping categorization as requested")
            return df

        self.logger.info("Starting transaction categorization")

        # Initialize categorizer
        categorizer = TransactionCategorizer(enable_logging=True)

        # Run categorization
        df_categorized = categorizer.categorize_transactions(df)

        # Export category mappings for review if requested
        if args.export_categories:
            categories_file = 'output/category_mappings_for_review.xlsx'
            os.makedirs(os.path.dirname(categories_file), exist_ok=True)

            # Get category summary
            category_summary = categorizer.get_category_summary()
            if not category_summary.empty:
                with pd.ExcelWriter(categories_file, engine='openpyxl') as writer:
                    category_summary.to_excel(writer, sheet_name='Category_Summary', index=False)
                self.logger.info(f"Category mappings exported to: {categories_file}")

        return df_categorized

    def run_analysis(self, df: pd.DataFrame, args) -> Dict[str, Any]:
        """Run comprehensive analysis"""
        self.logger.info("Starting comprehensive analysis")

        # Initialize analyzer
        analyzer = TransactionAnalyzer(enable_logging=True)

        # Run analysis with date filtering
        analysis_results = analyzer.comprehensive_analysis(
            df,
            start_date=args.start_date,
            end_date=args.end_date
        )

        return analysis_results

    def create_visualizations(self, df: pd.DataFrame, analysis_results: Dict[str, Any], args, output_dir: str,
                              context_label: str = None) -> List[str]:
        """Create visualizations"""
        if args.skip_charts:
            self.logger.info("Skipping chart generation as requested")
            return []

        self.logger.info("Creating visualizations")

        # Create charts directory
        charts_dir = f'{output_dir}/charts'
        os.makedirs(charts_dir, exist_ok=True)

        # Initialize visualizer with custom output directory and context label
        visualizer = TransactionVisualizer(enable_logging=True, output_dir=charts_dir, context_label=context_label)

        # Create charts
        chart_paths = visualizer.create_comprehensive_dashboard(df, analysis_results)

        return chart_paths

    def export_results(self, df: pd.DataFrame, analysis_results: Dict[str, Any],
                       chart_paths: List[str], args, output_dir: str) -> Dict[str, str]:
        """Export analysis results"""
        self.logger.info("Exporting analysis results")

        # Initialize exporter
        exporter = TransactionExporter(enable_logging=True)

        # Use provided output directory
        charts_dir = f'{output_dir}/charts'
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(charts_dir, exist_ok=True)

        # Extract folder name from output directory for file naming
        folder_name = Path(output_dir).name

        self.logger.info(f"Using output directory: {output_dir}")

        exported_files = {}

        # Export Excel (always)
        excel_path = f'{output_dir}/{folder_name}_analysis.xlsx'
        try:
            excel_file = exporter.export_to_excel(df, analysis_results, excel_path)
            exported_files['excel'] = excel_file
            self.logger.info(f"✓ Excel report: {excel_file}")
        except Exception as e:
            self.logger.error(f"Excel export failed: {e}")

        # Export simple version if requested
        if args.simple_export:
            simple_path = f'{output_dir}/{folder_name}_simple.xlsx'
            try:
                simple_file = exporter.export_simple_report(df, simple_path)
                exported_files['simple_excel'] = simple_file
                self.logger.info(f"✓ Simple Excel report: {simple_file}")
            except Exception as e:
                self.logger.error(f"Simple Excel export failed: {e}")

        # Move charts to the specific output folder
        if chart_paths:
            moved_charts = []
            for chart_path in chart_paths:
                if os.path.exists(chart_path):
                    chart_filename = Path(chart_path).name
                    new_chart_path = f'{charts_dir}/{chart_filename}'

                    try:
                        # Move chart to organized folder
                        os.rename(chart_path, new_chart_path)
                        moved_charts.append(new_chart_path)
                    except Exception as e:
                        self.logger.warning(f"Could not move chart {chart_path}: {e}")
                        moved_charts.append(chart_path)  # Keep original path

            exported_files['charts'] = moved_charts
            self.logger.info(f"✓ Moved {len(moved_charts)} charts to {charts_dir}")
            
        # Export PDF report if requested
        if args.pdf_export:
            pdf_path = f'{output_dir}/{folder_name}_report.pdf'
            try:
                # Use the moved charts for PDF (or original if moving failed)
                pdf_charts = moved_charts if 'charts' in exported_files else chart_paths
                pdf_file = exporter.export_pdf_report(df, analysis_results, pdf_charts, pdf_path)
                exported_files['pdf'] = pdf_file
                self.logger.info(f"✅ PDF report: {pdf_file}")
            except ImportError as e:
                self.logger.error(f"PDF export failed: {e}")
                self.logger.error("Please install reportlab: pip install reportlab")
            except Exception as e:
                self.logger.error(f"PDF export failed: {e}")

        return exported_files

    def generate_separate_account_reports(self, df: pd.DataFrame, args, base_output_dir: str) -> List[Dict[str, str]]:
        """Generate separate reports for each account under the main output directory"""
        if 'account_number' not in df.columns:
            self.logger.warning("No account column found - cannot generate separate account reports")
            return []

        accounts = df['account_number'].unique()
        if len(accounts) <= 1:
            self.logger.info("Only one account found - separate reports not necessary")
            return []

        self.logger.info(f"Generating separate reports for {len(accounts)} accounts")

        separate_reports = []

        for account in accounts:
            self.logger.info(f"Processing account {account}...")

            # Filter data for this account
            account_df = df[df['account_number'] == account].copy()

            # Run categorization for this account
            account_df = self.run_categorization(account_df, args)

            # Run analysis for this account
            account_analysis = self.run_analysis(account_df, args)

            # Setup output directory for this account under the main base directory
            account_output_dir = f'{base_output_dir}/account_{account}'

            # Create visualizations for this account with account context
            account_context_label = f"Account {account}"
            account_chart_paths = self.create_visualizations(account_df, account_analysis, args, account_output_dir,
                                                             account_context_label)

            # Export results for this account
            account_exported_files = self.export_results(account_df, account_analysis, account_chart_paths, args,
                                                         account_output_dir)

            # Add account info to the exported files info
            account_exported_files['account_number'] = account
            account_exported_files['transaction_count'] = len(account_df)

            separate_reports.append(account_exported_files)

            self.logger.info(f"✓ Account {account} report completed: {account_exported_files.get('excel', 'N/A')}")

        return separate_reports

    def run(self, args):
        """Main application workflow"""
        try:
            # Validate inputs
            self.validate_inputs(args)

            # Load data
            df = self.load_data(args)

            # Run categorization
            df = self.run_categorization(df, args)

            # Run analysis
            analysis_results = self.run_analysis(df, args)

            # Setup output directory
            input_filename = Path(args.input[0]).stem
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if args.output:
                folder_name = args.output
            elif len(args.input) == 1:
                folder_name = f"{input_filename}_{timestamp}"
            else:
                folder_name = f"merged_analysis_{timestamp}"

            output_dir = f'output/{folder_name}'

            # Create visualizations with filename as context
            main_context_label = Path(args.input[0]).stem if len(args.input) == 1 else "Combined Analysis"
            chart_paths = self.create_visualizations(df, analysis_results, args, output_dir, main_context_label)

            # Export results
            exported_files = self.export_results(df, analysis_results, chart_paths, args, output_dir)

            # Generate separate account reports if requested
            separate_reports = []
            if args.separate_accounts:
                separate_reports = self.generate_separate_account_reports(df, args, output_dir)

            # Print summary
            self.print_summary(df, analysis_results, exported_files, separate_reports)

        except Exception as e:
            self.logger.error(f"Application error: {e}")
            sys.exit(1)

    def print_summary(self, df: pd.DataFrame, analysis_results: Dict[str, Any],
                      exported_files: Dict[str, str], separate_reports: List[Dict[str, str]] = None):
        """Print execution summary"""
        print("\n" + "=" * 60)
        print("🎉 TRANSACTION ANALYSIS COMPLETED")
        print("=" * 60)

        # Dataset summary
        print(f"📊 Dataset Summary:")
        print(f"   • Total Transactions: {len(df):,}")

        if 'basic_statistics' in analysis_results:
            stats = analysis_results['basic_statistics']
            if 'dataset_overview' in stats:
                overview = stats['dataset_overview']
                print(f"   • Unique Accounts: {overview.get('unique_accounts', 0)}")
                print(f"   • Unique Merchants: {overview.get('unique_merchants', 0):,}")
                print(f"   • Date Range: {overview.get('date_span_days', 0)} days")

            if 'amount_analysis' in stats:
                amount = stats['amount_analysis']
                print(f"   • Net Position: £{amount.get('net_position', 0):,.2f}")

        # Categorization summary
        if 'auto_category' in df.columns:
            categorized_count = (df['auto_category'] != 'UNCATEGORIZED').sum()
            categorization_rate = (categorized_count / len(df)) * 100
            print(f"   • Categorization Rate: {categorization_rate:.1f}%")

        # Exported files
        print(f"\n📁 Generated Files:")
        print(f"🗂️  Combined Report:")
        for file_type, file_path in exported_files.items():
            if file_type not in ['account_number', 'transaction_count']:
                print(f"   • {file_type.upper()}: {file_path}")

        # Separate account reports
        if separate_reports:
            print(f"\n🗂️  Per-Account Reports:")
            for report in separate_reports:
                account_num = report.get('account_number', 'Unknown')
                tx_count = report.get('transaction_count', 0)
                excel_file = report.get('excel', 'N/A')
                print(f"   • Account {account_num} ({tx_count:,} transactions): {excel_file}")

        print("\n✨ Analysis complete! Check the output directory for detailed reports.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generic Transaction Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument('--input', '-i',
                        nargs='+',
                        required=True,
                        help='Input Excel/CSV file(s)')

    # Optional arguments
    parser.add_argument('--start-date', '-s',
                        help='Start date for analysis (YYYY-MM-DD)')

    parser.add_argument('--end-date', '-e',
                        help='End date for analysis (YYYY-MM-DD)')

    parser.add_argument('--accounts', '-a',
                        nargs='+',
                        help='Filter by specific account numbers')

    parser.add_argument('--output', '-o',
                        help='Output filename base (without extension)')

    parser.add_argument('--merge',
                        action='store_true',
                        help='Merge multiple input files')

    parser.add_argument('--skip-categorization',
                        action='store_true',
                        help='Skip automatic categorization')

    parser.add_argument('--skip-charts',
                        action='store_true',
                        help='Skip chart generation')

    parser.add_argument('--export-categories',
                        action='store_true',
                        help='Export category mappings for manual review')

    parser.add_argument('--simple-export',
                        action='store_true',
                        help='Also create simplified Excel export')
    
    parser.add_argument('--pdf-export',
                        action='store_true',
                        help='Generate consolidated PDF report with charts')

    parser.add_argument('--separate-accounts',
                        action='store_true',
                        help='Generate separate reports for each account (in addition to combined report)')

    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Enable verbose logging')

    # Parse arguments
    args = parser.parse_args()

    # Initialize and run application
    app = TransactionAnalysisApp()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run the application
    app.run(args)


if __name__ == "__main__":
    main()
