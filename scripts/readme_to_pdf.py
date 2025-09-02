#!/usr/bin/env python3
"""
README to PDF Conversion Script
Converts README.md to a professional PDF document for distribution
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import markdown
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
except ImportError as e:
    print(f"❌ Required dependencies not installed: {e}")
    print("\nTo install dependencies:")
    print("pip install markdown weasyprint")
    print("\nNote: WeasyPrint may require additional system dependencies.")
    print("On macOS: brew install pango")
    print("On Ubuntu: sudo apt-get install libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0")
    sys.exit(1)


class ReadmeToPDFConverter:
    """Professional README to PDF converter with custom styling"""
    
    def __init__(self, enable_logging: bool = True):
        self.logger = self._setup_logging() if enable_logging else None
        self.project_root = Path(__file__).parent.parent
        self.readme_path = self.project_root / "README.md"
        self.output_dir = self.project_root / "docs"
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
    
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
            
    def _get_custom_css(self) -> str:
        """Generate custom CSS for professional PDF styling"""
        return """
        @page {
            size: A4;
            margin: 2.5cm 2cm;
            
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-family: 'DejaVu Sans', sans-serif;
                font-size: 10pt;
                color: #666;
            }
            
            @bottom-right {
                content: "Generated on """ + datetime.now().strftime('%B %d, %Y') + """";
                font-family: 'DejaVu Sans', sans-serif;
                font-size: 9pt;
                color: #888;
            }
        }
        
        body {
            font-family: 'DejaVu Sans', 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 100%;
            margin: 0;
            padding: 0;
        }
        
        h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 0.5em;
            padding-bottom: 0.3em;
            border-bottom: 3px solid #3498db;
            page-break-after: avoid;
        }
        
        h2 {
            color: #34495e;
            font-size: 1.8em;
            margin-top: 1.5em;
            margin-bottom: 0.8em;
            page-break-after: avoid;
            border-left: 4px solid #3498db;
            padding-left: 0.5em;
        }
        
        h3 {
            color: #2c3e50;
            font-size: 1.4em;
            margin-top: 1.5em;
            margin-bottom: 0.6em;
            page-break-after: avoid;
        }
        
        h4 {
            color: #34495e;
            font-size: 1.2em;
            margin-top: 1.2em;
            margin-bottom: 0.5em;
            page-break-after: avoid;
        }
        
        p {
            margin-bottom: 1em;
            text-align: justify;
            orphans: 3;
            widows: 3;
        }
        
        code {
            background-color: #f8f9fa;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-family: 'DejaVu Sans Mono', 'Courier New', monospace;
            font-size: 0.9em;
            color: #e83e8c;
        }
        
        pre {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 0.8em;
            page-break-inside: avoid;
            margin: 0.8em 0;
            font-size: 0.85em;
            line-height: 1.3;
            white-space: pre-wrap;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        
        pre code {
            background-color: transparent;
            padding: 0;
            color: #333;
            font-size: inherit;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        blockquote {
            margin: 1.5em 0;
            padding: 0.5em 1em;
            border-left: 4px solid #ddd;
            background-color: #f9f9f9;
            font-style: italic;
        }
        
        ul, ol {
            margin-bottom: 1em;
            padding-left: 2em;
        }
        
        li {
            margin-bottom: 0.5em;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.2em 0;
            page-break-inside: avoid;
            page-break-before: avoid;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 0.75em;
            text-align: left;
        }
        
        th {
            background-color: #f8f9fa;
            font-weight: bold;
            color: #2c3e50;
        }
        
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        /* Emoji styling */
        .emoji {
            font-size: 1.2em;
            vertical-align: middle;
        }
        
        /* Link styling */
        a {
            color: #3498db;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        /* Page breaks */
        .page-break {
            page-break-before: always;
            margin: 0;
            padding: 0;
            height: 0;
        }
        
        /* Reduce spacing after page breaks */
        .page-break + h2 {
            margin-top: 0.3em;
        }
        
        /* Also handle if there are other elements after page break */
        .page-break + * {
            margin-top: 0.3em;
        }
        
        .page-break + h1 {
            margin-top: 0.2em;
        }
        
        /* Header styling for sections */
        .toc {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 1em;
            margin: 2em 0;
        }
        
        /* Prevent orphaned headers and improve table spacing */
        h1, h2, h3, h4, h5, h6 {
            page-break-after: avoid;
        }
        
        h1 + *, h2 + *, h3 + *, h4 + *, h5 + *, h6 + * {
            page-break-before: avoid;
        }
        
        /* Better table and section spacing */
        h2 + table {
            margin-top: 0.5em;
            page-break-before: avoid !important;
        }
        
        table + h2 {
            margin-top: 1.8em;
            page-break-before: auto;
        }
        
        /* Keep headers with their immediate content - stronger rules */
        h2 + p, h2 + ul, h2 + ol, h2 + pre, h2 + blockquote, h2 + table {
            page-break-before: avoid !important;
        }
        
        /* Prevent tables from being orphaned */
        table {
            page-break-before: avoid !important;
        }
        
        /* Avoid empty pages by managing breaks better */
        .avoid-break {
            page-break-inside: avoid;
            page-break-after: avoid;
        }
        
        /* Compact table formatting for better fit */
        table {
            font-size: 0.9em !important;
        }
        
        th, td {
            padding: 0.6em !important;
            font-size: 0.9em;
        }
        """
    
    def _preprocess_markdown(self, content: str) -> str:
        """Preprocess markdown content for better PDF rendering"""
        self._log('info', 'Preprocessing markdown content...')
        
        lines = content.split('\n')
        processed_lines = []
        
        for line in lines:
            
            # Add page breaks before major sections, but be more selective
            if line.startswith('## ') and not line.startswith('## 🌟'):
                # Only add page breaks for certain sections to avoid empty pages
                section_title = line.lower()
                if any(keyword in section_title for keyword in [
                    'project structure', 'quick start', 'generated analysis', 
                    'generated visualizations', 'export formats', 'configuration', 
                    'advanced usage', 'architecture design', 'requirements', 
                    'testing', 'contributing', 'documentation', 'support'
                ]) and 'command line options' not in section_title:
                    processed_lines.append('<div class="page-break"></div>')
                # Don't add empty line after page break to reduce spacing
            
            # Handle emoji in headers better
            if line.startswith('#'):
                # Ensure emojis render properly
                line = line.replace('🌟', '<span class="emoji">🌟</span>')
                line = line.replace('📁', '<span class="emoji">📁</span>')
                line = line.replace('🚀', '<span class="emoji">🚀</span>')
                line = line.replace('📊', '<span class="emoji">📊</span>')
                line = line.replace('🎯', '<span class="emoji">🎯</span>')
                line = line.replace('📈', '<span class="emoji">📈</span>')
                line = line.replace('📄', '<span class="emoji">📄</span>')
                line = line.replace('⚙️', '<span class="emoji">⚙️</span>')
                line = line.replace('🔧', '<span class="emoji">🔧</span>')
                line = line.replace('🎛️', '<span class="emoji">🎛️</span>')
                line = line.replace('🏗️', '<span class="emoji">🏗️</span>')
                line = line.replace('📋', '<span class="emoji">📋</span>')
                line = line.replace('✅', '<span class="emoji">✅</span>')
                line = line.replace('🤝', '<span class="emoji">🤝</span>')
                line = line.replace('📚', '<span class="emoji">📚</span>')
                line = line.replace('📞', '<span class="emoji">📞</span>')
            
            processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def convert_to_pdf(self, output_filename: Optional[str] = None) -> Path:
        """
        Convert README.md to PDF
        
        Args:
            output_filename: Optional custom output filename
            
        Returns:
            Path to generated PDF file
        """
        if not self.readme_path.exists():
            raise FileNotFoundError(f"README.md not found at {self.readme_path}")
        
        # Set output filename
        if not output_filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"Transaction_Analysis_Tool_Documentation_{timestamp}.pdf"
        
        output_path = self.output_dir / output_filename
        
        self._log('info', f'Converting README.md to PDF: {output_path}')
        
        try:
            # Read and preprocess README content
            with open(self.readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            content = self._preprocess_markdown(content)
            
            # Convert markdown to HTML
            self._log('info', 'Converting markdown to HTML...')
            md = markdown.Markdown(extensions=[
                'markdown.extensions.tables',
                'markdown.extensions.fenced_code',
                'markdown.extensions.toc',
                'markdown.extensions.codehilite'
            ])
            
            html_content = md.convert(content)
            
            # Wrap in proper HTML document
            html_document = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Transaction Analysis Tool - Documentation</title>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            # Convert HTML to PDF
            self._log('info', 'Converting HTML to PDF...')
            font_config = FontConfiguration()
            
            html_doc = HTML(string=html_document, base_url=str(self.project_root))
            css_doc = CSS(string=self._get_custom_css(), font_config=font_config)
            
            html_doc.write_pdf(
                str(output_path),
                stylesheets=[css_doc],
                font_config=font_config
            )
            
            self._log('info', f'✅ PDF generated successfully: {output_path}')
            return output_path
            
        except Exception as e:
            self._log('error', f'Error converting README to PDF: {e}')
            raise
    
    def generate_with_metadata(self) -> Path:
        """Generate PDF with enhanced metadata and professional naming"""
        self._log('info', 'Generating professional documentation PDF...')
        
        # Create a descriptive filename
        output_filename = "Generic_Transaction_Analysis_Tool_Documentation.pdf"
        
        return self.convert_to_pdf(output_filename)


def main():
    """Main entry point"""
    try:
        converter = ReadmeToPDFConverter()
        
        print("🔄 Converting README.md to professional PDF documentation...")
        pdf_path = converter.generate_with_metadata()
        
        print(f"✅ Success! PDF generated at: {pdf_path}")
        print(f"📄 File size: {pdf_path.stat().st_size / 1024:.1f} KB")
        print("\n📋 Generated PDF includes:")
        print("   • Professional formatting and layout")
        print("   • Proper page breaks and sections")
        print("   • Table of contents navigation")
        print("   • Code syntax highlighting")
        print("   • Emoji rendering")
        print("   • Page numbers and metadata")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
