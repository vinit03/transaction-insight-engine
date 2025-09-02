#!/bin/bash

# Build Documentation Script
# Generates PDF documentation from README.md

echo "🔄 Building Documentation..."

# Check if development dependencies are installed
python -c "import markdown, weasyprint" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing documentation dependencies..."
    pip install -r requirements-dev.txt
fi

# Generate PDF documentation
echo "📄 Generating PDF documentation from README.md..."
python scripts/readme_to_pdf.py

echo "✅ Documentation build complete!"
echo "📋 Generated files:"
echo "   • docs/Generic_Transaction_Analysis_Tool_Documentation.pdf"
