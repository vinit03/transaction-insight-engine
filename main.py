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
    # Look for input files in the standard 'inputs/' directory
    input_dir = Path("inputs")
    if not input_dir.exists():
        return []  # Return empty list if no inputs directory exists

    # Support common financial data formats: Excel and CSV
    patterns = ['*.xlsx', '*.xls', '*.csv']
    files = []
    
    # Scan for all supported file types in the inputs directory
    for pattern in patterns:
        files.extend(list(input_dir.glob(pattern)))

    # Return alphabetically sorted list for consistent user experience
    return sorted(files)


def get_user_choice(prompt, options, default=None):
    """Get user choice from a list of options"""
    # Display the prompt and numbered options to the user
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")

    # Show default option if provided
    if default:
        print(f"\nPress Enter for default: {default}")

    # Input validation loop - keep asking until valid choice is made
    while True:
        try:
            sys.stdout.flush()  # Ensure output is displayed before input prompt
            choice = input("\nYour choice: ").strip()

            # Handle default selection (empty input)
            if not choice and default:
                return default

            # Validate numeric choice and range
            if choice.isdigit():
                idx = int(choice) - 1  # Convert to 0-based index
                if 0 <= idx < len(options):
                    return options[idx]  # Return the selected option

            print("❌ Invalid choice. Please try again.")

        except KeyboardInterrupt:
            # Graceful exit on Ctrl+C
            print("\n\n👋 Goodbye!")
            sys.exit(0)


def get_yes_no(prompt, default='n'):
    """Get yes/no input from user"""
    # Format the prompt to show which option is default (uppercase letter)
    default_text = "Y/n" if default.lower() == 'y' else "y/N"

    # Input validation loop for yes/no responses
    while True:
        try:
            sys.stdout.flush()  # Ensure output is displayed before input prompt
            choice = input(f"\n{prompt} [{default_text}]: ").strip().lower()

            # Handle default selection (empty input)
            if not choice:
                return default.lower() == 'y'  # Return boolean based on default

            # Accept various forms of yes/no input
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False

            print("❌ Please enter 'y' for yes or 'n' for no.")

        except KeyboardInterrupt:
            # Graceful exit on Ctrl+C
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

        # Handle file selection logic based on number of available files
        if len(input_files) == 1:
            # Auto-select single file for convenience
            selected_file = input_files[0]
            print(f"\n✅ Auto-selected: {selected_file.name}")
            input_args = [str(selected_file)]  # Convert Path to string for subprocess
            use_merge = False
        else:
            # Multiple files available - offer merge option
            print(f"\n   {len(input_files) + 1}. Multiple files (merge)")

            # Get user choice for file selection or merge
            choice = get_user_choice("Choose input option:",
                                     [f.name for f in input_files] + ["Multiple files (merge)"])

            if choice == "Multiple files (merge)":
                # User chose to merge all files - prepare file list for CLI
                input_args = [str(f) for f in input_files]
                use_merge = True
                print(f"✅ Selected: {len(input_files)} files for merging")
            else:
                # User chose a specific file - find and select it
                selected_file = next(f for f in input_files if f.name == choice)
                input_args = [str(selected_file)]
                use_merge = False
                print(f"✅ Selected: {choice}")

        # Step 2: Analysis options
        print("\n⚙️  STEP 2: ANALYSIS OPTIONS")
        print("-" * 30)

        # DATE FILTERING: Allow users to focus on specific time periods
        print("📅 Date Range Filtering:")
        start_date = get_optional_input("Start date (YYYY-MM-DD)", validate_date)
        end_date = get_optional_input("End date (YYYY-MM-DD)", validate_date)

        # Confirm date selections to user
        if start_date:
            print(f"   ✅ Start date: {start_date}")
        if end_date:
            print(f"   ✅ End date: {end_date}")

        # ACCOUNT FILTERING: Allow analysis of specific accounts only
        print("\n🏦 Account Filtering:")
        accounts_input = get_optional_input("Specific account numbers (comma separated)", validate_accounts)
        accounts = None
        if accounts_input:
            # Parse comma or space-separated account numbers
            accounts = accounts_input.replace(',', ' ').split()
            print(f"   ✅ Accounts: {', '.join(accounts)}")

        # ADVANCED OPTIONS: Feature toggles for different analysis aspects
        print("\n🔧 Advanced Options:")

        # Each option controls a different aspect of analysis behavior
        separate_accounts = get_yes_no("Generate separate reports for each account?")  # Per-account breakdowns
        skip_categorization = get_yes_no("Skip automatic categorization?")  # Faster processing, less insights
        skip_charts = get_yes_no("Skip chart generation?")  # Faster processing, no visualizations
        export_categories = get_yes_no("Export category mappings for review?")  # For manual category refinement
        simple_export = get_yes_no("Also create simplified Excel export?")  # Executive summary version
        pdf_export = get_yes_no("Generate comprehensive PDF report?", 'y')  # Professional report format

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

        # BUILD CLI COMMAND: Convert UI selections into command-line arguments
        # Use absolute path to ensure script can be found regardless of working directory
        analyzer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', 'transaction_analyzer.py')
        cmd = ['python', analyzer_path]

        # Add input files as required arguments
        cmd.extend(['--input'] + input_args)

        # Add optional arguments based on user selections
        if use_merge:
            cmd.append('--merge')  # Combine multiple files into single analysis

        if start_date:
            cmd.extend(['--start-date', start_date])  # Filter transactions from this date

        if end_date:
            cmd.extend(['--end-date', end_date])  # Filter transactions until this date

        if accounts:
            cmd.extend(['--accounts'] + accounts)  # Analyze only specified account numbers

        if custom_output:
            cmd.extend(['--output', custom_output])  # Use custom output folder name

        if separate_accounts:
            cmd.append('--separate-accounts')  # Generate individual account reports

        if skip_categorization:
            cmd.append('--skip-categorization')  # Skip ML categorization for speed

        if skip_charts:
            cmd.append('--skip-charts')  # Skip visualization generation

        if export_categories:
            cmd.append('--export-categories')  # Export categorization mappings

        if simple_export:
            cmd.append('--simple-export')  # Generate executive summary version

        if pdf_export:
            cmd.append('--pdf-export')  # Generate professional PDF report

        cmd.append('--verbose')  # Always enable detailed logging for UI users

        # EXECUTION PHASE: Show command and execute analysis
        # Display equivalent CLI command for power users and troubleshooting
        print("\n💡 Equivalent CLI command:")
        cli_cmd = ' '.join(cmd[1:])  # Remove 'python' for cleaner display
        print(f"   python {cli_cmd}")

        # Execute the analysis with visual feedback
        print("\n" + "=" * 70)
        print("🔄 STARTING ANALYSIS...")
        print("=" * 70)

        try:
            # Run the CLI analyzer subprocess with all user-selected options
            # check=True ensures we catch any subprocess errors immediately
            result = subprocess.run(cmd, check=True)

            # SUCCESS: Analysis completed successfully
            print("\n" + "=" * 70)
            print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 70)

            # Provide clear next steps for the user based on their selections
            print("\n💡 Next Steps:")
            print("   1. Check the 'output/' directory for your reports")
            print("   2. Open Excel files for detailed analysis")
            print("   3. Review charts for visual insights")
            if separate_accounts:
                print("   4. Check individual account folders for per-account reports")

        except subprocess.CalledProcessError as e:
            # ERROR HANDLING: CLI analyzer returned non-zero exit code
            print(f"\n❌ Analysis failed with error code {e.returncode}")
            print("📋 Check the logs for detailed error information")
            sys.exit(1)  # Exit with error status to indicate failure

        except KeyboardInterrupt:
            # USER INTERRUPTION: Handle Ctrl+C gracefully during analysis
            print("\n\n⚠️  Analysis interrupted by user")
            sys.exit(1)  # Exit with error status to indicate interruption

    except Exception as e:
        # UNEXPECTED ERRORS: Catch-all for any other issues in main interface
        print(f"\n❌ Unexpected error: {e}")
        print("📋 Please check your input files and try again")
        sys.exit(1)  # Exit with error status to indicate failure


if __name__ == "__main__":
    main()
