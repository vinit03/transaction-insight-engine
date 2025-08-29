"""
Dynamic Visualization System for Transaction Analysis
Creates charts that adapt to the actual data found
"""

import logging
import os
import sys
import warnings
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.settings import EXPORT_SETTINGS, ANALYSIS_SETTINGS
from src.decimal_utils import decimal_to_float

# Set matplotlib style for better-looking charts
plt.style.use('default')
sns.set_palette("husl")


class TransactionVisualizer:
    """Dynamic chart generator that adapts to any transaction data"""

    def __init__(self, enable_logging: bool = True, output_dir: str = 'output/charts', context_label: str = None):
        self.logger = self._setup_logging() if enable_logging else None
        self.figure_size = (12, 8)
        self.dpi = EXPORT_SETTINGS.get('chart_dpi', 150)
        self.color_palette = sns.color_palette("husl", 20)
        self.charts_generated = []
        self.output_dir = output_dir
        self.context_label = context_label  # e.g., "Account 1234" or "sample_data"

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

    def _prepare_numeric_for_plotting(self, series: pd.Series) -> pd.Series:
        """Convert Decimal values to float for matplotlib compatibility"""
        # Check if series contains Decimal values
        sample_values = series.dropna().head(5)
        if not sample_values.empty and any(isinstance(val, Decimal) for val in sample_values):
            # Convert Decimal to float for plotting
            return series.apply(lambda x: decimal_to_float(x) if isinstance(x, Decimal) else x)
        return series

    def _save_chart(self, base_filename: str, title: str) -> str:
        """Save chart and return filepath"""
        os.makedirs(self.output_dir, exist_ok=True)

        # Add context prefix to filename if provided
        if self.context_label:
            # Clean context label for filename (remove spaces, special chars)
            clean_context = ''.join(c for c in self.context_label if c.isalnum() or c in ('_', '-')).lower()
            filename = f"{clean_context}_{base_filename}"
        else:
            filename = base_filename

        filepath = f'{self.output_dir}/{filename}'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        self.charts_generated.append({
            'filename': filename,
            'filepath': filepath,
            'title': title,
            'generated_at': datetime.now().isoformat()
        })

        self._log('info', f"Chart saved: {filepath}")
        return filepath

    def create_amount_distribution_chart(self, df: pd.DataFrame) -> Optional[str]:
        """Create amount distribution histogram"""
        if 'amount' not in df.columns:
            return None

        plt.figure(figsize=self.figure_size, dpi=self.dpi)

        amounts = df['amount'].dropna()

        # Create subplot for positive and negative amounts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Positive amounts (income)
        positive_amounts = amounts[amounts > 0]
        if not positive_amounts.empty:
            ax1.hist(positive_amounts, bins=50, color='green', alpha=0.7, edgecolor='black')
            ax1.set_title('Income Distribution (Positive Amounts)')
            ax1.set_xlabel('Amount')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)

        # Negative amounts (expenses)
        negative_amounts = amounts[amounts < 0].abs()
        if not negative_amounts.empty:
            ax2.hist(negative_amounts, bins=50, color='red', alpha=0.7, edgecolor='black')
            ax2.set_title('Expense Distribution (Absolute Values)')
            ax2.set_xlabel('Amount')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._save_chart('amount_distribution.png', 'Transaction Amount Distribution')

    def create_merchant_frequency_chart(self, df: pd.DataFrame, top_n: int = None) -> Optional[str]:
        """Create top merchants by frequency chart"""
        if 'main_description' not in df.columns:
            return None

        plt.figure(figsize=self.figure_size, dpi=self.dpi)

        # Get top merchants using configured setting
        if top_n is None:
            top_n = ANALYSIS_SETTINGS.get('top_merchants_count', 15)
        merchant_counts = df['main_description'].value_counts().head(top_n)

        if merchant_counts.empty:
            return None

        # Truncate long merchant names for display
        merchant_names = [name[:40] + '...' if len(name) > 40 else name for name in merchant_counts.index]

        # Create horizontal bar chart
        plt.barh(range(len(merchant_counts)), merchant_counts.values,
                 color=self.color_palette[:len(merchant_counts)])
        plt.yticks(range(len(merchant_counts)), merchant_names)
        plt.xlabel('Number of Transactions')
        plt.title(f'Top {top_n} Merchants by Transaction Frequency')
        plt.grid(True, alpha=0.3, axis='x')

        # Invert y-axis to show highest at top
        plt.gca().invert_yaxis()
        plt.tight_layout()

        return self._save_chart('merchant_frequency.png', f'Top {top_n} Merchants by Frequency')

    def create_monthly_trends_chart(self, df: pd.DataFrame) -> Optional[str]:
        """Create monthly transaction trends chart"""
        if 'transaction_date' not in df.columns or 'amount' not in df.columns:
            return None

        plt.figure(figsize=self.figure_size, dpi=self.dpi)

        # Prepare monthly data
        df_temp = df.copy()
        df_temp['year_month'] = df_temp['transaction_date'].dt.to_period('M')

        # Calculate monthly metrics
        monthly_data = df_temp.groupby('year_month').agg({
            'amount': ['count', 'sum']
        }).round(2)

        monthly_data.columns = ['transaction_count', 'total_amount']

        if monthly_data.empty:
            return None

        # Create subplot for count and amount
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Transaction count trend
        ax1.plot(monthly_data.index.astype(str), monthly_data['transaction_count'],
                 marker='o', linewidth=2, markersize=6, color='blue')

        title_suffix = f" - {self.context_label}" if self.context_label else ""
        ax1.set_title(f'Monthly Transaction Count Trend{title_suffix}')
        ax1.set_ylabel('Number of Transactions')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # Amount trend
        ax2.plot(monthly_data.index.astype(str), monthly_data['total_amount'],
                 marker='s', linewidth=2, markersize=6, color='green')
        ax2.set_title(f'Monthly Transaction Amount Trend{title_suffix}')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Total Amount')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return self._save_chart('monthly_trends.png', 'Monthly Transaction Trends')

    def create_category_distribution_chart(self, df: pd.DataFrame) -> Optional[str]:
        """Create category distribution pie chart (if categories exist)"""
        if 'auto_category' not in df.columns:
            return None

        category_counts = df['auto_category'].value_counts()

        if category_counts.empty or len(category_counts) <= 1:
            return None

        plt.figure(figsize=self.figure_size, dpi=self.dpi)

        # Limit to top categories for readability using configured setting
        max_categories = ANALYSIS_SETTINGS.get('top_merchants_count', 15)
        if len(category_counts) > max_categories:
            top_categories = category_counts.head(max_categories - 1)
            others_count = category_counts.tail(len(category_counts) - (max_categories - 1)).sum()
            top_categories['OTHERS'] = others_count
            category_counts = top_categories

        # Create pie chart
        wedges, texts, autotexts = plt.pie(category_counts.values,
                                           labels=category_counts.index,
                                           autopct='%1.1f%%',
                                           startangle=90,
                                           colors=self.color_palette[:len(category_counts)])

        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.title('Transaction Distribution by Category')
        plt.axis('equal')

        return self._save_chart('category_distribution.png', 'Transaction Category Distribution')

    def create_cash_flow_chart(self, df: pd.DataFrame) -> Optional[str]:
        """Create cash flow analysis chart"""
        if 'amount' not in df.columns or 'transaction_date' not in df.columns:
            return None

        plt.figure(figsize=self.figure_size, dpi=self.dpi)

        # Prepare data
        df_temp = df.copy()
        df_temp['year_month'] = df_temp['transaction_date'].dt.to_period('M')
        df_temp['income'] = df_temp['amount'].where(df_temp['amount'] > 0, 0)
        df_temp['expenses'] = df_temp['amount'].where(df_temp['amount'] < 0, 0).abs()

        monthly_flow = df_temp.groupby('year_month').agg({
            'income': 'sum',
            'expenses': 'sum'
        }).round(2)

        if monthly_flow.empty:
            return None

        monthly_flow['net_flow'] = monthly_flow['income'] - monthly_flow['expenses']

        # Create the chart
        x_labels = [str(period) for period in monthly_flow.index]
        x_pos = np.arange(len(x_labels))

        plt.bar(x_pos, monthly_flow['income'], label='Income', color='green', alpha=0.7)
        plt.bar(x_pos, -monthly_flow['expenses'], label='Expenses', color='red', alpha=0.7)

        # Net flow line
        plt.plot(x_pos, monthly_flow['net_flow'], label='Net Flow',
                 color='blue', linewidth=2, marker='o')

        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Month')
        plt.ylabel('Amount')
        plt.title('Monthly Cash Flow Analysis')
        plt.xticks(x_pos, x_labels, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._save_chart('cash_flow_analysis.png', 'Monthly Cash Flow Analysis')

    def create_account_comparison_chart(self, df: pd.DataFrame) -> Optional[str]:
        """Create account comparison chart (if multiple accounts)"""
        if 'account_number' not in df.columns or df['account_number'].nunique() <= 1:
            return None

        plt.figure(figsize=self.figure_size, dpi=self.dpi)

        # Account metrics
        account_stats = df.groupby('account_number').agg({
            'amount': ['count', 'sum', 'mean']
        }).round(2)

        account_stats.columns = ['transaction_count', 'total_amount', 'avg_amount']

        if account_stats.empty:
            return None

        # Create subplot for different metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        accounts = account_stats.index.astype(str)
        colors = self.color_palette[:len(accounts)]

        # Transaction count by account
        ax1.bar(accounts, account_stats['transaction_count'], color=colors)
        ax1.set_title('Transaction Count by Account')
        ax1.set_ylabel('Number of Transactions')

        # Total amount by account
        ax2.bar(accounts, account_stats['total_amount'], color=colors)
        ax2.set_title('Total Amount by Account')
        ax2.set_ylabel('Total Amount')

        # Average transaction size by account
        ax3.bar(accounts, account_stats['avg_amount'], color=colors)
        ax3.set_title('Average Transaction Size by Account')
        ax3.set_ylabel('Average Amount')

        # Account activity over time (if date column exists)
        if 'transaction_date' in df.columns:
            for i, account in enumerate(df['account_number'].unique()):
                account_data = df[df['account_number'] == account]
                daily_counts = account_data.groupby('transaction_date').size()
                ax4.plot(daily_counts.index, daily_counts.values,
                         label=f'Account {account}', color=colors[i], alpha=0.7)

            ax4.set_title('Daily Activity by Account')
            ax4.set_ylabel('Daily Transactions')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._save_chart('account_comparison.png', 'Account Comparison Analysis')

    def create_weekday_patterns_chart(self, df: pd.DataFrame) -> Optional[str]:
        """Create weekday transaction patterns chart"""
        if 'transaction_date' not in df.columns:
            return None

        plt.figure(figsize=self.figure_size, dpi=self.dpi)

        # Calculate weekday patterns
        df_temp = df.copy()
        df_temp['day_of_week'] = df_temp['transaction_date'].dt.day_of_week
        df_temp['weekday_name'] = df_temp['transaction_date'].dt.day_name()

        weekday_data = df_temp.groupby(['day_of_week', 'weekday_name']).agg({
            'amount': ['count', 'sum', 'mean']
        }).round(2)

        weekday_data.columns = ['transaction_count', 'total_amount', 'avg_amount']
        weekday_data = weekday_data.reset_index()

        if weekday_data.empty:
            return None

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Transaction count by weekday
        ax1.bar(weekday_data['weekday_name'], weekday_data['transaction_count'],
                color=self.color_palette[:7])
        ax1.set_title('Transaction Count by Day of Week')
        ax1.set_ylabel('Number of Transactions')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Total amount by weekday
        ax2.bar(weekday_data['weekday_name'], weekday_data['total_amount'],
                color=self.color_palette[7:14])
        ax2.set_title('Total Amount by Day of Week')
        ax2.set_ylabel('Total Amount')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._save_chart('weekday_patterns.png', 'Weekday Transaction Patterns')

    def create_outliers_chart(self, df: pd.DataFrame, top_n: int = None) -> Optional[str]:
        """Create chart showing transaction outliers"""
        if 'amount' not in df.columns:
            return None

        # Get top outliers by absolute amount using configured setting
        if top_n is None:
            top_n = ANALYSIS_SETTINGS.get('top_merchants_count', 15)
        df_outliers = df.nlargest(top_n, df['amount'].abs())

        if df_outliers.empty:
            return None

        plt.figure(figsize=self.figure_size, dpi=self.dpi)

        # Prepare data
        amounts = df_outliers['amount'].values
        descriptions = [desc[:30] + '...' if len(desc) > 30 else desc
                        for desc in df_outliers['main_description'].fillna('Unknown')]

        # Color based on positive/negative
        colors = ['green' if amount > 0 else 'red' for amount in amounts]

        # Create horizontal bar chart
        plt.barh(range(len(amounts)), amounts, color=colors, alpha=0.7)
        plt.yticks(range(len(amounts)), descriptions)
        plt.xlabel('Amount')
        plt.title(f'Top {top_n} Transaction Outliers by Amount')
        plt.grid(True, alpha=0.3, axis='x')

        # Add zero line
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        # Invert y-axis to show largest at top
        plt.gca().invert_yaxis()
        plt.tight_layout()

        return self._save_chart('transaction_outliers.png', f'Top {top_n} Transaction Outliers')

    def create_balance_trend_chart(self, df: pd.DataFrame) -> Optional[str]:
        """Create balance trend chart (if balance column exists)"""
        if 'balance' not in df.columns or 'transaction_date' not in df.columns:
            return None

        plt.figure(figsize=self.figure_size, dpi=self.dpi)

        # Handle multiple accounts
        if 'account_number' in df.columns and df['account_number'].nunique() > 1:
            fig, axes = plt.subplots(df['account_number'].nunique(), 1,
                                     figsize=(12, 6 * df['account_number'].nunique()))

            if df['account_number'].nunique() == 1:
                axes = [axes]

            for i, account in enumerate(df['account_number'].unique()):
                account_data = df[df['account_number'] == account].sort_values('transaction_date')

                if not account_data.empty:
                    axes[i].plot(account_data['transaction_date'], account_data['balance'],
                                 linewidth=2, marker='o', markersize=3)
                    axes[i].set_title(f'Account {account} - Balance Trend')
                    axes[i].set_ylabel('Balance')
                    axes[i].grid(True, alpha=0.3)
                    axes[i].tick_params(axis='x', rotation=45)
        else:
            # Single account
            df_sorted = df.sort_values('transaction_date')
            plt.plot(df_sorted['transaction_date'], df_sorted['balance'],
                     linewidth=2, marker='o', markersize=3, color='blue')
            plt.title('Account Balance Trend Over Time')
            plt.xlabel('Date')
            plt.ylabel('Balance')
            plt.grid(True, alpha=0.3)
            plt.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return self._save_chart('balance_trends.png', 'Account Balance Trends')

    def create_income_vs_expense_chart(self, df: pd.DataFrame) -> Optional[str]:
        """Create income vs expense comparison chart"""
        if 'amount' not in df.columns:
            return None

        # Calculate totals
        total_income = df[df['amount'] > 0]['amount'].sum() if (df['amount'] > 0).any() else 0
        total_expenses = abs(df[df['amount'] < 0]['amount'].sum()) if (df['amount'] < 0).any() else 0

        if total_income == 0 and total_expenses == 0:
            return None

        plt.figure(figsize=(10, 8), dpi=self.dpi)

        # Create pie chart for overall distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Overall Income vs Expenses
        if total_income > 0 or total_expenses > 0:
            amounts = [total_income, total_expenses]
            labels = ['Income', 'Expenses']
            colors = ['green', 'red']

            ax1.pie(amounts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax1.set_title('Income vs Expenses Distribution')

        # Monthly income vs expenses
        if 'transaction_date' in df.columns:
            df_temp = df.copy()
            df_temp['year_month'] = df_temp['transaction_date'].dt.to_period('M')
            df_temp['income'] = df_temp['amount'].where(df_temp['amount'] > 0, 0)
            df_temp['expenses'] = df_temp['amount'].where(df_temp['amount'] < 0, 0).abs()

            monthly_summary = df_temp.groupby('year_month').agg({
                'income': 'sum',
                'expenses': 'sum'
            }).round(2)

            if not monthly_summary.empty:
                x_pos = np.arange(len(monthly_summary))
                width = 0.35

                ax2.bar(x_pos - width / 2, monthly_summary['income'], width,
                        label='Income', color='green', alpha=0.7)
                ax2.bar(x_pos + width / 2, monthly_summary['expenses'], width,
                        label='Expenses', color='red', alpha=0.7)

                ax2.set_xlabel('Month')
                ax2.set_ylabel('Amount')
                ax2.set_title('Monthly Income vs Expenses')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels([str(period) for period in monthly_summary.index], rotation=45)
                ax2.legend()
                ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._save_chart('income_vs_expenses.png', 'Income vs Expenses Analysis')

    def create_comprehensive_dashboard(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Create a comprehensive dashboard with multiple charts
        
        Args:
            df: Transaction dataframe
            analysis_results: Results from analysis engine
            
        Returns:
            List of generated chart filepaths
        """
        self._log('info', f"Creating comprehensive dashboard for {len(df):,} transactions")

        generated_charts = []

        try:
            # Generate all possible charts
            chart_methods = [
                ('amount_distribution', self.create_amount_distribution_chart),
                ('merchant_frequency', self.create_merchant_frequency_chart),
                ('monthly_trends', self.create_monthly_trends_chart),
                ('weekday_patterns', self.create_weekday_patterns_chart),
                ('income_vs_expenses', self.create_income_vs_expense_chart),
                ('balance_trends', self.create_balance_trend_chart),
                ('category_distribution', self.create_category_distribution_chart),
                ('outliers', self.create_outliers_chart)
            ]

            for chart_name, chart_method in chart_methods:
                try:
                    chart_path = chart_method(df)
                    if chart_path:
                        generated_charts.append(chart_path)
                        self._log('info', f"✓ Generated {chart_name} chart")
                    else:
                        self._log('info', f"⚠ Skipped {chart_name} chart (insufficient data)")

                    # Clear matplotlib state
                    plt.clf()
                    plt.close('all')

                except Exception as e:
                    self._log('error', f"Error creating {chart_name} chart: {e}")
                    continue

        except Exception as e:
            self._log('error', f"Error in dashboard creation: {e}")

        self._log('info', f"Dashboard creation completed. Generated {len(generated_charts)} charts")
        return generated_charts

    def get_chart_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all generated charts"""
        return self.charts_generated.copy()


def create_visualizations(df: pd.DataFrame, analysis_results: Optional[Dict[str, Any]] = None) -> List[str]:
    """Convenience function to create all visualizations"""
    visualizer = TransactionVisualizer()
    return visualizer.create_comprehensive_dashboard(df, analysis_results or {})
