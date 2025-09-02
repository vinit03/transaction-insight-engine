"""
PDF Export Functionality for Transaction Analysis
Creates comprehensive PDF reports combining tables and charts for decision makers
"""

import logging
import os
import sys
import warnings
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any, Optional
import tempfile

import pandas as pd

# Suppress PIL debug logs that are too verbose
logging.getLogger('PIL').setLevel(logging.WARNING)
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, FrameBreak
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.linecharts import HorizontalLineChart

warnings.filterwarnings('ignore')

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.settings import EXPORT_SETTINGS, DECIMAL_SETTINGS
from src.decimal_utils import decimal_to_float


class TransactionPDFExporter:
    """Create comprehensive PDF reports for transaction analysis"""

    def __init__(self, enable_logging: bool = True):
        self.logger = self._setup_logging() if enable_logging else None
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
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

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the PDF"""
        
        # Define styles only if they don't exist
        style_definitions = [
            ('CustomTitle', {
                'parent': self.styles['Heading1'],
                'fontSize': 24,
                'spaceAfter': 30,
                'alignment': TA_CENTER,
                'textColor': colors.darkblue,
                'fontName': 'Helvetica-Bold'
            }),
            ('SectionHeader', {
                'parent': self.styles['Heading2'],
                'fontSize': 16,
                'spaceBefore': 20,
                'spaceAfter': 12,
                'textColor': colors.darkblue,
                'fontName': 'Helvetica-Bold'
            }),
            ('SubsectionHeader', {
                'parent': self.styles['Heading3'],
                'fontSize': 14,
                'spaceBefore': 15,
                'spaceAfter': 8,
                'textColor': colors.darkgreen,
                'fontName': 'Helvetica-Bold'
            }),
            ('CustomBodyText', {  # Renamed to avoid conflict with default BodyText
                'parent': self.styles['Normal'],
                'fontSize': 11,
                'spaceBefore': 6,
                'spaceAfter': 6,
                'alignment': TA_JUSTIFY,
                'fontName': 'Helvetica'
            }),
            ('Executive', {
                'parent': self.styles['Normal'],
                'fontSize': 12,
                'spaceBefore': 8,
                'spaceAfter': 8,
                'leftIndent': 20,
                'rightIndent': 20,
                'borderWidth': 1,
                'borderColor': colors.darkblue,
                'backColor': colors.lightblue,
                'fontName': 'Helvetica'
            })
        ]
        
        # Add styles if they don't exist
        for style_name, style_attrs in style_definitions:
            if style_name not in self.styles:
                self.styles.add(ParagraphStyle(name=style_name, **style_attrs))

    def export_consolidated_pdf(self, df: pd.DataFrame, analysis_results: Dict[str, Any],
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
        self._log('info', f"Starting consolidated PDF export to {output_path}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create the PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )

        # Build the story (content)
        story = []
        
        # 1. Cover Page
        story.extend(self._create_cover_page(df, analysis_results))
        story.append(PageBreak())
        
        # 2. Executive Summary
        story.extend(self._create_executive_summary(df, analysis_results))
        story.append(PageBreak())
        
        # 3. Financial Overview
        story.extend(self._create_financial_overview(df, analysis_results, chart_paths))
        story.append(PageBreak())
        
        # 4. Account Analysis
        if 'account_analysis' in analysis_results:
            story.extend(self._create_account_analysis(analysis_results['account_analysis'], chart_paths))
            story.append(PageBreak())
        
        # 5. Transaction Analysis
        story.extend(self._create_transaction_analysis(df, analysis_results, chart_paths))
        story.append(PageBreak())
        
        # 6. Visual Analytics Dashboard
        story.extend(self._create_visual_analytics(chart_paths))
        story.append(PageBreak())
        
        # 7. Data Quality Report
        if 'data_quality_analysis' in analysis_results:
            story.extend(self._create_data_quality_report(analysis_results['data_quality_analysis']))

        # Build the PDF
        try:
            doc.build(story)
            self._log('info', f"✅ PDF report generated successfully: {output_path}")
            return output_path
        except Exception as e:
            self._log('error', f"Error generating PDF report: {e}")
            raise

    def _create_cover_page(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List:
        """Create professional cover page"""
        content = []
        
        # Title
        content.append(Paragraph("Transaction Analysis Report", self.styles['CustomTitle']))
        content.append(Spacer(1, 0.5*inch))
        
        # Report period and basic stats
        if not df.empty and 'transaction_date' in df.columns:
            start_date = df['transaction_date'].min().strftime('%B %d, %Y')
            end_date = df['transaction_date'].max().strftime('%B %d, %Y')
            period_text = f"<b>Analysis Period:</b> {start_date} to {end_date}"
        else:
            period_text = "<b>Analysis Period:</b> Full Dataset"
            
        content.append(Paragraph(period_text, self.styles['CustomBodyText']))
        content.append(Spacer(1, 0.3*inch))
        
        # Key statistics summary
        total_transactions = len(df)
        if 'account_number' in df.columns:
            total_accounts = df['account_number'].nunique()
        else:
            total_accounts = 1
            
        if 'amount' in df.columns:
            total_volume = df['amount'].sum()
            total_volume_str = f"£{total_volume:,.2f}" if isinstance(total_volume, (int, float, Decimal)) else "N/A"
        else:
            total_volume_str = "N/A"
            
        summary_data = [
            ['Metric', 'Value'],
            ['Total Transactions', f"{total_transactions:,}"],
            ['Total Accounts', f"{total_accounts:,}"],
            ['Total Volume', total_volume_str],
            ['Report Generated', datetime.now().strftime('%B %d, %Y at %I:%M %p')]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(summary_table)
        content.append(Spacer(1, 0.5*inch))
        
        # Confidentiality notice
        confidential_text = """
        <b>CONFIDENTIAL BUSINESS DOCUMENT</b><br/>
        This report contains sensitive financial information and is intended solely for authorized 
        decision-makers. Distribution should be limited to individuals with legitimate business needs.
        """
        content.append(Paragraph(confidential_text, self.styles['Executive']))
        
        return content

    def _create_executive_summary(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List:
        """Create executive summary with key insights"""
        content = []
        
        content.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Key insights from analysis
        insights = []
        
        # Basic statistics insights
        if 'basic_statistics' in analysis_results:
            basic_stats = analysis_results['basic_statistics']
            
            if 'dataset_overview' in basic_stats:
                overview = basic_stats['dataset_overview']
                insights.append(f"Analysis covers {overview.get('unique_accounts', 'N/A')} accounts over {overview.get('date_span_days', 'N/A')} days")
                
            if 'amount_analysis' in basic_stats:
                amount_analysis = basic_stats['amount_analysis']
                total_credits = amount_analysis.get('total_credits', 0)
                total_debits = amount_analysis.get('total_debits', 0) 
                net_position = amount_analysis.get('net_position', 0)
                
                insights.append(f"Total income: £{total_credits:,.2f}")
                insights.append(f"Total expenses: £{abs(total_debits):,.2f}")
                
                if net_position > 0:
                    insights.append(f"Net positive position of £{net_position:,.2f}")
                else:
                    insights.append(f"Net negative position of £{abs(net_position):,.2f}")
        
        # Categorization insights
        if 'category_analysis' in analysis_results:
            category_data = analysis_results['category_analysis']
            if 'categorization_summary' in category_data:
                cat_summary = category_data['categorization_summary']
                categorized_pct = cat_summary.get('categorization_rate', 0) * 100
                insights.append(f"Successfully categorized {categorized_pct:.1f}% of transactions automatically")
        
        # Create insights list
        if insights:
            content.append(Paragraph("Key Findings:", self.styles['SubsectionHeader']))
            for insight in insights[:6]:  # Limit to top 6 insights
                content.append(Paragraph(f"• {insight}", self.styles['CustomBodyText']))
            content.append(Spacer(1, 0.2*inch))
        
        # Recommendations section
        content.append(Paragraph("Recommendations:", self.styles['SubsectionHeader']))
        
        recommendations = [
            "Review uncategorized transactions for potential automation improvements",
            "Monitor accounts with irregular activity patterns",
            "Consider implementing spending limits for high-activity categories",
            "Establish regular reporting cycles for financial oversight"
        ]
        
        for rec in recommendations:
            content.append(Paragraph(f"• {rec}", self.styles['CustomBodyText']))
        
        return content

    def _create_financial_overview(self, df: pd.DataFrame, analysis_results: Dict[str, Any], 
                                 chart_paths: List[str]) -> List:
        """Create financial overview section with key charts"""
        content = []
        
        content.append(Paragraph("Financial Overview", self.styles['SectionHeader']))
        
        # Income vs Expenses chart
        income_expense_chart = self._find_chart(chart_paths, 'income_vs_expenses')
        if income_expense_chart and os.path.exists(income_expense_chart):
            content.append(Paragraph("Income vs Expenses Analysis", self.styles['SubsectionHeader']))
            content.extend(self._create_image_with_caption(
                income_expense_chart, 
                "Monthly comparison of income and expenses showing cash flow trends"
            ))
            content.append(Spacer(1, 0.3*inch))
        
        # Monthly trends
        monthly_chart = self._find_chart(chart_paths, 'monthly_trends')
        if monthly_chart and os.path.exists(monthly_chart):
            content.append(Paragraph("Monthly Transaction Trends", self.styles['SubsectionHeader']))
            content.extend(self._create_image_with_caption(
                monthly_chart,
                "Transaction volume and value trends over the analysis period"
            ))
        
        return content

    def _create_account_analysis(self, account_analysis: Dict[str, Any], chart_paths: List[str]) -> List:
        """Create account analysis section"""
        content = []
        
        content.append(Paragraph("Account Analysis", self.styles['SectionHeader']))
        
        if 'account_details' in account_analysis:
            content.append(Paragraph("Account Performance Summary", self.styles['SubsectionHeader']))
            
            # Create account summary table
            accounts = account_analysis['account_details']
            if accounts:
                table_data = [['Account', 'Status', 'Transactions', 'Net Position', 'Avg Transaction']]
                
                for account in accounts[:10]:  # Limit to top 10 accounts
                    account_num = str(account.get('account_number', 'N/A'))
                    status = 'Active' if account.get('is_active', False) else 'Inactive'
                    transactions = account.get('total_transactions', 0)
                    net_pos = account.get('net_position', 0)
                    avg_txn = account.get('avg_transaction', 0)
                    
                    table_data.append([
                        account_num,
                        status,
                        f"{transactions:,}",
                        f"£{net_pos:,.2f}",
                        f"£{avg_txn:,.2f}"
                    ])
                
                account_table = Table(table_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1.2*inch, 1.2*inch])
                account_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 9)
                ]))
                
                content.append(account_table)
        
        return content

    def _create_transaction_analysis(self, df: pd.DataFrame, analysis_results: Dict[str, Any],
                                   chart_paths: List[str]) -> List:
        """Create transaction analysis section"""
        content = []
        
        content.append(Paragraph("Transaction Analysis", self.styles['SectionHeader']))
        
        # Category distribution
        category_chart = self._find_chart(chart_paths, 'category_distribution')
        if category_chart and os.path.exists(category_chart):
            content.append(Paragraph("Transaction Categories", self.styles['SubsectionHeader']))
            content.extend(self._create_image_with_caption(
                category_chart,
                "Distribution of transactions by automatically assigned categories"
            ))
            content.append(Spacer(1, 0.3*inch))
        
        # Merchant frequency analysis
        merchant_chart = self._find_chart(chart_paths, 'merchant_frequency')
        if merchant_chart and os.path.exists(merchant_chart):
            content.append(Paragraph("Top Merchants", self.styles['SubsectionHeader']))
            content.extend(self._create_image_with_caption(
                merchant_chart,
                "Most frequent transaction sources and their activity levels"
            ))
            content.append(Spacer(1, 0.3*inch))
        
        # Transaction patterns
        weekday_chart = self._find_chart(chart_paths, 'weekday_patterns')
        if weekday_chart and os.path.exists(weekday_chart):
            content.append(Paragraph("Transaction Patterns", self.styles['SubsectionHeader']))
            content.extend(self._create_image_with_caption(
                weekday_chart,
                "Weekly transaction patterns showing business activity cycles"
            ))
        
        return content

    def _create_visual_analytics(self, chart_paths: List[str]) -> List:
        """Create visual analytics dashboard"""
        content = []
        
        content.append(Paragraph("Visual Analytics Dashboard", self.styles['SectionHeader']))
        
        # Amount distribution
        amount_chart = self._find_chart(chart_paths, 'amount_distribution')
        if amount_chart and os.path.exists(amount_chart):
            content.append(Paragraph("Transaction Amount Distribution", self.styles['SubsectionHeader']))
            content.extend(self._create_image_with_caption(
                amount_chart,
                "Distribution of transaction amounts showing spending patterns"
            ))
            content.append(Spacer(1, 0.3*inch))
        
        # Balance trends
        balance_chart = self._find_chart(chart_paths, 'balance_trends')
        if balance_chart and os.path.exists(balance_chart):
            content.append(Paragraph("Account Balance Trends", self.styles['SubsectionHeader']))
            content.extend(self._create_image_with_caption(
                balance_chart,
                "Account balance trends over time showing financial position changes"
            ))
        
        return content

    def _create_data_quality_report(self, data_quality: Dict[str, Any]) -> List:
        """Create data quality and confidence report"""
        content = []
        
        content.append(Paragraph("Data Quality & Confidence Report", self.styles['SectionHeader']))
        
        content.append(Paragraph(
            "This section provides insights into data completeness and the confidence "
            "level of automated analysis results.",
            self.styles['CustomBodyText']
        ))
        
        # Missing data analysis
        if 'column_completeness' in data_quality:
            content.append(Paragraph("Data Completeness", self.styles['SubsectionHeader']))
            
            completeness = data_quality['column_completeness']
            table_data = [['Column', 'Complete %', 'Missing Count']]
            
            for col_info in completeness[:10]:  # Top 10 columns
                col_name = col_info.get('column', 'Unknown')
                complete_pct = col_info.get('completeness_percentage', 0)
                missing_count = col_info.get('missing_count', 0)
                
                table_data.append([
                    col_name,
                    f"{complete_pct:.1f}%",
                    f"{missing_count:,}"
                ])
            
            completeness_table = Table(table_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            completeness_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9)
            ]))
            
            content.append(completeness_table)
        
        content.append(Spacer(1, 0.3*inch))
        
        # Confidence summary
        content.append(Paragraph("Analysis Confidence Summary", self.styles['SubsectionHeader']))
        confidence_text = """
        The automated analysis has been performed with high confidence on the available data. 
        Key areas of high confidence include transaction categorization, amount calculations, 
        and temporal analysis. Areas requiring manual review include uncategorized transactions 
        and accounts with unusual activity patterns.
        """
        content.append(Paragraph(confidence_text, self.styles['CustomBodyText']))
        
        return content

    def _find_chart(self, chart_paths: List[str], chart_type: str) -> Optional[str]:
        """Find specific chart by type in the chart paths list"""
        for path in chart_paths:
            if chart_type.lower() in path.lower():
                return path
        return None

    def _create_image_with_caption(self, image_path: str, caption: str, 
                                 max_width: float = 6*inch, max_height: float = 4*inch) -> List:
        """Create an image with caption for the PDF, preserving aspect ratio"""
        content = []
        
        try:
            from PIL import Image as PILImage
            
            # Open image to get original dimensions
            with PILImage.open(image_path) as pil_img:
                original_width, original_height = pil_img.size
                original_ratio = original_width / original_height
                
            # Calculate proper dimensions preserving aspect ratio
            if max_width / max_height > original_ratio:
                # Height is the limiting factor
                draw_height = max_height
                draw_width = draw_height * original_ratio
            else:
                # Width is the limiting factor
                draw_width = max_width
                draw_height = draw_width / original_ratio
                
            # Add the image with preserved aspect ratio
            img = Image(image_path)
            img.drawHeight = draw_height
            img.drawWidth = draw_width
            content.append(img)
            
            # Add caption
            content.append(Spacer(1, 0.1*inch))
            caption_style = ParagraphStyle(
                'Caption',
                parent=self.styles['Normal'],
                fontSize=9,
                alignment=TA_CENTER,
                textColor=colors.grey,
                fontName='Helvetica-Oblique'
            )
            content.append(Paragraph(f"Figure: {caption}", caption_style))
            content.append(Spacer(1, 0.2*inch))
            
        except Exception as e:
            self._log('warning', f"Could not add image {image_path}: {e}")
            # Add placeholder text instead
            content.append(Paragraph(f"[Chart: {caption}]", self.styles['CustomBodyText']))
            content.append(Spacer(1, 0.1*inch))
        
        return content


def export_pdf_report(df: pd.DataFrame, analysis_results: Dict[str, Any],
                     chart_paths: List[str], output_path: str) -> str:
    """Convenience function for PDF report export"""
    exporter = TransactionPDFExporter()
    return exporter.export_consolidated_pdf(df, analysis_results, chart_paths, output_path)
