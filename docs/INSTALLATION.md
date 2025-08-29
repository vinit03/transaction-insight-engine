# Quick Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Setup Steps

### 1. Clone or Download Project

```bash
# If using git
git clone <repository-url>

# Or download and extract the ZIP file
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 4. Test Installation

```bash
# Test with help command
python scripts/transaction_analyzer.py --help

# Should display all available options
```

## Quick Start

### Basic Analysis

```bash
# Analyze a single file
python scripts/transaction_analyzer.py --input your_file.xlsx

# With date filtering
python scripts/transaction_analyzer.py --input your_file.xlsx \
    --start-date 2024-01-01 --end-date 2024-12-31

# Generate separate reports per account
python scripts/transaction_analyzer.py --input your_file.xlsx \
    --separate-accounts --simple-export
```

## Expected File Format

Your Excel/CSV file should have columns like:

- **AccNo**: Account number
- **Date**: Transaction date
- **MainDesc**: Transaction description
- **Amount**: Transaction amount
- **Balance**: Running balance (optional)

The tool automatically detects column names - see README.md for adding new formats.

## Output Location

All reports and charts are saved to:

```
output/
├── [filename]_[timestamp]/
│   ├── analysis.xlsx        # Main comprehensive report
│   ├── simple.xlsx         # Simplified report (if requested)
│   ├── charts/             # Generated visualizations
│   └── account_[num]/      # Per-account reports (if requested)
```

## Troubleshooting

### Common Issues

1. **"Module not found" error**:
   ```bash
   # Make sure virtual environment is activated
   source venv/bin/activate  # macOS/Linux
   # or
   venv\Scripts\activate     # Windows
   
   # Reinstall dependencies
   pip install -r requirements.txt
   ```

2. **"File not found" error**:
   ```bash
   # Use full path to your file
   python scripts/transaction_analyzer.py --input /full/path/to/your/file.xlsx
   ```

3. **Permission errors**:
   ```bash
   # Make sure you have write permissions in the output directory
   # Try running from the project root directory
   ```

### Getting Help

```bash
# Show all available options
python scripts/transaction_analyzer.py --help

# Run with verbose logging to see detailed information
python scripts/transaction_analyzer.py --input your_file.xlsx --verbose
```

## Next Steps

- Read the full README.md for advanced usage
- Check PROJECT_SUMMARY.md for feature overview
- Review generated logs in logs/ directory if issues occur
