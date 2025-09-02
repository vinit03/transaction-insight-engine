#!/usr/bin/env python3
"""
🏦 TRANSACTION ANALYSIS TOOL - MAIN INTERFACE
=============================================

User-friendly interface for comprehensive transaction analysis.
Simply run: python main.py

Features:
- 📊 Comprehensive transaction analysis
- 🤖 Intelligent auto-categorization  
- 📈 Charts and visualizations
- 📑 Multi-sheet Excel reports
- 📋 Professional PDF reports
- 🗂️ Per-account analysis
- 🔍 Advanced filtering options

"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def print_header():
    """Print application header"""
    print("=" * 70)
    print("🏦 TRANSACTION ANALYSIS TOOL")
    print("=" * 70)
    print("🚀 Generic transaction analysis for any business type")
    print("💡 Supports Excel files with flexible column mapping")
    print("📊 Generates comprehensive reports with visualizations")
    print("=" * 70)


def find_input_files():
    """Find available input files"""
    input_dir = Path("inputs")
    if not input_dir.exists():
        return []

    patterns = ['*.xlsx', '*.xls', '*.csv']
    files = []
    for pattern in patterns:
        files.extend(list(input_dir.glob(pattern)))

    return sorted(files)


def get_user_choice(prompt, options, default=None):
    """Get user choice from a list of options"""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")

    if default:
        print(f"\nPress Enter for default: {default}")

    while True:
        try:
            sys.stdout.flush()  # Ensure output is displayed before input
            choice = input("\nYour choice: ").strip()

            if not choice and default:
                return default

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx]

            print("❌ Invalid choice. Please try again.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)


def get_yes_no(prompt, default='n'):
    """Get yes/no input from user"""
    default_text = "Y/n" if default.lower() == 'y' else "y/N"

    while True:
        try:
            sys.stdout.flush()  # Ensure output is displayed before input
            choice = input(f"\n{prompt} [{default_text}]: ").strip().lower()

            if not choice:
                return default.lower() == 'y'

            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False

            print("❌ Please enter 'y' for yes or 'n' for no.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)


def get_optional_input(prompt, validation_func=None):
    """Get optional input from user"""
    while True:
        try:
            sys.stdout.flush()  # Ensure output is displayed before input
            value = input(f"\n{prompt} (press Enter to skip): ").strip()

            if not value:
                return None

            if validation_func and not validation_func(value):
                print("❌ Invalid input format.")
                continue  # Use continue instead of recursive call

            return value

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)


def validate_date(date_str):
    """Validate date format YYYY-MM-DD"""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def validate_accounts(accounts_str):
    """Validate account numbers (comma or space separated)"""
    try:
        # Allow comma or space separated
        accounts = accounts_str.replace(',', ' ').split()
        return all(acc.isdigit() for acc in accounts)
    except:
        return False


def verify_cli_alignment():
    """Verify that main.py uses the same arguments as CLI"""
    # These should match exactly with scripts/transaction_analyzer.py argument parser
    expected_args = {
        '--input': 'Input files',
        '--start-date': 'Start date filter',
        '--end-date': 'End date filter',
        '--accounts': 'Account numbers filter',
        '--output': 'Output folder name',
        '--merge': 'Merge multiple files',
        '--separate-accounts': 'Generate per-account reports',
        '--skip-categorization': 'Skip auto-categorization',
        '--skip-charts': 'Skip chart generation',
        '--export-categories': 'Export category mappings',
        '--simple-export': 'Create simplified export',
        '--pdf-export': 'Generate comprehensive PDF report',
        '--verbose': 'Enable verbose logging'
    }
    # This function serves as documentation and validation
    return expected_args


def main():
    """Main interactive interface"""
    try:
        print_header()

        # Step 1: Find and select input files
        print("\n📂 STEP 1: SELECT INPUT FILES")
        print("-" * 35)

        input_files = find_input_files()

        if not input_files:
            print("❌ No input files found in 'inputs/' directory.")
            print("💡 Please add your Excel (.xlsx, .xls) or CSV files to the 'inputs/' folder.")
            print("\nExpected columns in your file:")
            print("   • Account Number (AccNo)")
            print("   • Date")
            print("   • Description (MainDesc)")
            print("   • Amount")
            print("   • Balance (optional)")
            sys.exit(1)

        print("📊 Available input files:")
        for i, file in enumerate(input_files, 1):
            file_size = file.stat().st_size / 1024 / 1024  # MB
            print(f"   {i}. {file.name} ({file_size:.1f}MB)")

        if len(input_files) == 1:
            selected_file = input_files[0]
            print(f"\n✅ Auto-selected: {selected_file.name}")
            input_args = [str(selected_file)]
            use_merge = False
        else:
            print(f"\n   {len(input_files) + 1}. Multiple files (merge)")

            choice = get_user_choice("Choose input option:",
                                     [f.name for f in input_files] + ["Multiple files (merge)"])

            if choice == "Multiple files (merge)":
                input_args = [str(f) for f in input_files]
                use_merge = True
                print(f"✅ Selected: {len(input_files)} files for merging")
            else:
                selected_file = next(f for f in input_files if f.name == choice)
                input_args = [str(selected_file)]
                use_merge = False
                print(f"✅ Selected: {choice}")

        # Step 2: Analysis options
        print("\n⚙️  STEP 2: ANALYSIS OPTIONS")
        print("-" * 30)

        # Date filtering
        print("📅 Date Range Filtering:")
        start_date = get_optional_input("Start date (YYYY-MM-DD)", validate_date)
        end_date = get_optional_input("End date (YYYY-MM-DD)", validate_date)

        if start_date:
            print(f"   ✅ Start date: {start_date}")
        if end_date:
            print(f"   ✅ End date: {end_date}")

        # Account filtering
        print("\n🏦 Account Filtering:")
        accounts_input = get_optional_input("Specific account numbers (comma separated)", validate_accounts)
        accounts = None
        if accounts_input:
            accounts = accounts_input.replace(',', ' ').split()
            print(f"   ✅ Accounts: {', '.join(accounts)}")

        # Advanced options
        print("\n🔧 Advanced Options:")

        separate_accounts = get_yes_no("Generate separate reports for each account?")
        skip_categorization = get_yes_no("Skip automatic categorization?")
        skip_charts = get_yes_no("Skip chart generation?")
        export_categories = get_yes_no("Export category mappings for review?")
        simple_export = get_yes_no("Also create simplified Excel export?")
        pdf_export = get_yes_no("Generate comprehensive PDF report?", 'y')

        # Step 3: Output options
        print("\n📁 STEP 3: OUTPUT OPTIONS")
        print("-" * 27)

        custom_output = get_optional_input("Custom output folder name (optional)")

        # Step 4: Confirmation and execution
        print("\n🚀 STEP 4: EXECUTION SUMMARY")
        print("-" * 31)

        print("📋 Analysis Configuration:")
        print(f"   • Input Files: {len(input_args)} file(s)")
        if use_merge:
            print("   • Mode: Multiple file merge")
        if start_date or end_date:
            date_range = f"{start_date or 'start'} to {end_date or 'end'}"
            print(f"   • Date Filter: {date_range}")
        if accounts:
            print(f"   • Account Filter: {', '.join(accounts)}")
        if separate_accounts:
            print("   • Per-Account Reports: YES")
        if skip_categorization:
            print("   • Categorization: SKIPPED")
        if skip_charts:
            print("   • Charts: SKIPPED")
        if export_categories:
            print("   • Category Export: YES")
        if simple_export:
            print("   • Simple Export: YES")
        if pdf_export:
            print("   • PDF Report: YES")
        if custom_output:
            print(f"   • Output Folder: {custom_output}")

        if not get_yes_no("🔥 Start analysis?", 'y'):
            print("❌ Analysis cancelled.")
            sys.exit(0)

        # Build command with an absolute path to script
        analyzer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', 'transaction_analyzer.py')
        cmd = ['python', analyzer_path]

        # Add input files
        cmd.extend(['--input'] + input_args)

        # Add options
        if use_merge:
            cmd.append('--merge')

        if start_date:
            cmd.extend(['--start-date', start_date])

        if end_date:
            cmd.extend(['--end-date', end_date])

        if accounts:
            cmd.extend(['--accounts'] + accounts)

        if custom_output:
            cmd.extend(['--output', custom_output])

        if separate_accounts:
            cmd.append('--separate-accounts')

        if skip_categorization:
            cmd.append('--skip-categorization')

        if skip_charts:
            cmd.append('--skip-charts')

        if export_categories:
            cmd.append('--export-categories')

        if simple_export:
            cmd.append('--simple-export')

        if pdf_export:
            cmd.append('--pdf-export')

        cmd.append('--verbose')  # Always use verbose for main.py

        # Show equivalent CLI command
        print("\n💡 Equivalent CLI command:")
        cli_cmd = ' '.join(cmd[1:])  # Remove 'python' for cleaner display
        print(f"   python {cli_cmd}")

        # Execute analysis
        print("\n" + "=" * 70)
        print("🔄 STARTING ANALYSIS...")
        print("=" * 70)

        try:
            result = subprocess.run(cmd, check=True)

            print("\n" + "=" * 70)
            print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 70)

            print("\n💡 Next Steps:")
            print("   1. Check the 'output/' directory for your reports")
            print("   2. Open Excel files for detailed analysis")
            print("   3. Review charts for visual insights")
            if separate_accounts:
                print("   4. Check individual account folders for per-account reports")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Analysis failed with error code {e.returncode}")
            print("📋 Check the logs for detailed error information")
            sys.exit(1)

        except KeyboardInterrupt:
            print("\n\n⚠️  Analysis interrupted by user")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("📋 Please check your input files and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()
