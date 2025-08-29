"""
Scalable Analysis Engine for Transaction Data
Handles 1M+ records with adaptive analysis for any business type
"""

import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.settings import ANALYSIS_SETTINGS
from src.decimal_utils import (
    decimal_sum, decimal_mean, decimal_min, decimal_max, decimal_abs,
    decimal_to_float, decimal_comparison_ops
)


class TransactionAnalyzer:
    """Scalable analyzer for transaction data of any business type"""

    def __init__(self, enable_logging: bool = True):
        self.logger = self._setup_logging() if enable_logging else None
        self.analysis_results = {}
        self.performance_metrics = {}

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

    def _apply_date_filter(self, df: pd.DataFrame, start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """Apply date filtering to dataframe"""
        if 'transaction_date' not in df.columns:
            self._log('warning', "No date column found for filtering")
            return df

        original_count = len(df)
        filtered_df = df.copy()

        if start_date:
            start_date_parsed = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df['transaction_date'] >= start_date_parsed]
            self._log('info', f"Applied start date filter: {start_date}")

        if end_date:
            end_date_parsed = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df['transaction_date'] <= end_date_parsed]
            self._log('info', f"Applied end date filter: {end_date}")

        filtered_count = len(filtered_df)
        self._log('info', f"Date filtering: {original_count:,} -> {filtered_count:,} rows "
                          f"({((filtered_count / original_count) * 100):.1f}% retained)")

        return filtered_df

    def basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic statistics for any dataset"""
        stats = {
            'dataset_overview': {
                'total_transactions': len(df),
                'unique_accounts': df['account_number'].nunique() if 'account_number' in df.columns else 0,
                'date_span_days': 0,
                'unique_merchants': df['main_description'].nunique() if 'main_description' in df.columns else 0,
                'currencies': df['currency'].unique().tolist() if 'currency' in df.columns else []
            }
        }

        # Date analysis
        if 'transaction_date' in df.columns:
            date_stats = {
                'earliest_date': df['transaction_date'].min(),
                'latest_date': df['transaction_date'].max(),
                'date_span_days': (df['transaction_date'].max() - df['transaction_date'].min()).days,
                'unique_dates': df['transaction_date'].nunique()
            }
            stats['dataset_overview'].update(date_stats)

        # Amount analysis
        if 'amount' in df.columns:
            amounts = df['amount'].dropna()

            # Use Decimal operations for precise calculations
            positive_mask = decimal_comparison_ops(df, 'amount', 0, '>')
            negative_mask = decimal_comparison_ops(df, 'amount', 0, '<')
            zero_mask = decimal_comparison_ops(df, 'amount', 0, '==')

            total_value = decimal_sum(amounts)
            positive_amounts = amounts[positive_mask[amounts.index]] if positive_mask.any() else pd.Series([],
                                                                                                           dtype=object)
            negative_amounts = amounts[negative_mask[amounts.index]] if negative_mask.any() else pd.Series([],
                                                                                                           dtype=object)

            amount_stats = {
                'total_value': decimal_to_float(total_value) if total_value else 0,
                'positive_transactions': positive_mask.sum(),
                'negative_transactions': negative_mask.sum(),
                'zero_transactions': zero_mask.sum(),
                'total_credits': decimal_to_float(decimal_sum(positive_amounts)) if not positive_amounts.empty else 0,
                'total_debits': decimal_to_float(decimal_sum(negative_amounts)) if not negative_amounts.empty else 0,
                'net_position': decimal_to_float(total_value) if total_value else 0,
                'avg_transaction_size': decimal_to_float(decimal_mean(amounts.apply(decimal_abs))) if len(
                    amounts) > 0 else 0,
                'largest_credit': decimal_to_float(decimal_max(positive_amounts)) if not positive_amounts.empty else 0,
                'largest_debit': decimal_to_float(decimal_min(negative_amounts)) if not negative_amounts.empty else 0
            }
            stats['amount_analysis'] = amount_stats

        return stats

    def account_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze accounts present in the dataset"""
        if 'account_number' not in df.columns:
            return {'error': 'No account column found'}

        account_stats = {}

        # Overall account metrics
        accounts = df['account_number'].unique()
        account_stats['overview'] = {
            'total_accounts': len(accounts),
            'account_numbers': accounts.tolist()
        }

        # Per-account analysis
        account_details = []
        for account in accounts:
            account_data = df[df['account_number'] == account]

            account_detail = {
                'account_number': account,
                'transaction_count': len(account_data),
                'date_range': {
                    'first_transaction': account_data[
                        'transaction_date'].min() if 'transaction_date' in df.columns else None,
                    'last_transaction': account_data[
                        'transaction_date'].max() if 'transaction_date' in df.columns else None
                }
            }

            # Financial metrics per account
            if 'amount' in account_data.columns:
                amounts = account_data['amount'].dropna()
                account_detail['financial_summary'] = {
                    'total_credits': amounts[amounts > 0].sum() if (amounts > 0).any() else 0,
                    'total_debits': amounts[amounts < 0].sum() if (amounts < 0).any() else 0,
                    'net_position': amounts.sum(),
                    'avg_transaction': amounts.mean() if len(amounts) > 0 else 0,
                    'current_balance': account_data['balance'].iloc[-1] if 'balance' in account_data.columns and len(
                        account_data) > 0 else None
                }

            # Activity status with smart date reference
            if 'transaction_date' in account_data.columns:
                # Use smart reference date logic (same as active_inactive_accounts_analysis)
                data_end = account_data['transaction_date'].max()
                current_date = datetime.now()
                days_since_data_end = (current_date - data_end).days

                # Use data end as reference if data is historical (> 30 days old)
                reference_date = data_end if days_since_data_end > 30 else current_date

                days_since_last = (reference_date - account_data['transaction_date'].max()).days
                account_detail['activity_status'] = {
                    'days_since_last_transaction': days_since_last,
                    'is_active': days_since_last <= ANALYSIS_SETTINGS.get('active_account_days', 90),
                    'reference_date_used': 'data_end' if days_since_data_end > 30 else 'current_date',
                    'is_historical_data': days_since_data_end > 30,
                    'transactions_last_30_days': len(account_data[
                                                         account_data['transaction_date'] > (
                                                                 reference_date - timedelta(days=30))
                                                         ]) if 'transaction_date' in account_data.columns else 0
                }

            account_details.append(account_detail)

        account_stats['account_details'] = account_details
        return account_stats

    def merchant_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze merchant/payee patterns"""
        if 'main_description' not in df.columns:
            return {'error': 'No description column found'}

        # Top merchants by transaction count
        merchant_counts = df['main_description'].value_counts()
        top_merchants = merchant_counts.head(ANALYSIS_SETTINGS.get('top_merchants_count', 25))

        # Top merchants by total amount (for expenses)
        merchant_amounts = {}
        if 'amount' in df.columns:
            # Expenses (negative amounts)
            expense_data = df[df['amount'] < 0].copy()
            if not expense_data.empty:
                expense_by_merchant = expense_data.groupby('main_description')['amount'].sum().abs()
                top_expense_merchants = expense_by_merchant.nlargest(ANALYSIS_SETTINGS.get('top_merchants_count', 15))
                merchant_amounts['top_expense_merchants'] = top_expense_merchants.to_dict()

            # Revenue sources (positive amounts)
            revenue_data = df[df['amount'] > 0].copy()
            if not revenue_data.empty:
                revenue_by_merchant = revenue_data.groupby('main_description')['amount'].sum()
                top_revenue_merchants = revenue_by_merchant.nlargest(ANALYSIS_SETTINGS.get('top_merchants_count', 15))
                merchant_amounts['top_revenue_merchants'] = top_revenue_merchants.to_dict()

        return {
            'merchant_frequency': {
                'total_unique_merchants': merchant_counts.shape[0],
                'single_occurrence_merchants': (merchant_counts == 1).sum(),
                'frequent_merchants': (merchant_counts >= 5).sum(),
                'top_merchants': top_merchants.to_dict()
            },
            'merchant_amounts': merchant_amounts
        }

    def temporal_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze transaction patterns over time"""
        if 'transaction_date' not in df.columns:
            return {'error': 'No date column found'}

        df_temp = df.copy()

        # Add time components
        df_temp['year'] = df_temp['transaction_date'].dt.year
        df_temp['month'] = df_temp['transaction_date'].dt.month
        df_temp['day_of_week'] = df_temp['transaction_date'].dt.day_of_week
        df_temp['day_of_month'] = df_temp['transaction_date'].dt.day
        df_temp['week_of_year'] = df_temp['transaction_date'].dt.isocalendar().week

        temporal_stats = {}

        # Monthly analysis
        if 'amount' in df_temp.columns:
            monthly_summary = df_temp.groupby(['year', 'month']).agg({
                'amount': ['count', 'sum', 'mean'],
                'transaction_date': 'nunique'
            }).round(2)

            monthly_summary.columns = ['transaction_count', 'total_amount', 'avg_amount', 'active_days']
            temporal_stats['monthly_summary'] = monthly_summary.to_dict('index')

            # Income vs expenses by month
            df_temp['income'] = df_temp['amount'].where(df_temp['amount'] > 0, 0)
            df_temp['expenses'] = df_temp['amount'].where(df_temp['amount'] < 0, 0)

            monthly_flow = df_temp.groupby(['year', 'month']).agg({
                'income': 'sum',
                'expenses': 'sum'
            }).round(2)
            monthly_flow['net_flow'] = monthly_flow['income'] + monthly_flow['expenses']

            temporal_stats['monthly_cash_flow'] = monthly_flow.to_dict('index')

        # Weekly patterns
        weekday_patterns = df_temp['day_of_week'].value_counts().sort_index()
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        temporal_stats['weekday_patterns'] = dict(zip(weekday_names, weekday_patterns.values))

        # Transaction frequency over time
        daily_counts = df_temp.groupby('transaction_date').size()
        temporal_stats['transaction_frequency'] = {
            'avg_transactions_per_day': daily_counts.mean(),
            'max_transactions_per_day': daily_counts.max(),
            'min_transactions_per_day': daily_counts.min(),
            'busiest_day': daily_counts.idxmax(),
            'quietest_day': daily_counts.idxmin()
        }

        return temporal_stats

    def category_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze transaction categories (if auto_category column exists)"""
        if 'auto_category' not in df.columns:
            return {'error': 'No category column found - run categorization first'}

        category_stats = {}

        # Category distribution
        category_counts = df['auto_category'].value_counts()
        category_stats['category_distribution'] = {
            'total_categories': len(category_counts),
            'category_counts': category_counts.to_dict(),
            'uncategorized_percentage': (category_counts.get('UNCATEGORIZED', 0) / len(df)) * 100
        }

        # Financial analysis by category
        if 'amount' in df.columns:
            category_amounts = df.groupby('auto_category')['amount'].agg([
                'count', 'sum', 'mean', 'std', 'min', 'max'
            ]).round(2)

            category_stats['category_financial_summary'] = category_amounts.to_dict('index')

            # Expense categories (negative amounts only)
            expense_data = df[df['amount'] < 0]
            if not expense_data.empty:
                expense_by_category = expense_data.groupby('auto_category')['amount'].sum().abs()
                category_stats['expense_breakdown'] = expense_by_category.to_dict()

            # Revenue categories (positive amounts only)
            revenue_data = df[df['amount'] > 0]
            if not revenue_data.empty:
                revenue_by_category = revenue_data.groupby('auto_category')['amount'].sum()
                category_stats['revenue_breakdown'] = revenue_by_category.to_dict()

        return category_stats

    def outlier_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers and unusual transactions"""
        if 'amount' not in df.columns:
            return {'error': 'No amount column found'}

        amounts = df['amount'].dropna().abs()  # Use absolute values for outlier detection

        # Statistical outlier detection
        q75, q25 = np.percentile(amounts, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - (1.5 * iqr)
        upper_bound = q75 + (1.5 * iqr)

        # Percentile-based thresholds
        percentile_threshold = np.percentile(amounts, ANALYSIS_SETTINGS.get('large_transaction_percentile', 95))

        outlier_stats = {
            'statistical_outliers': {
                'count': len(df[df['amount'].abs() > upper_bound]),
                'threshold': upper_bound,
                'percentage': (len(df[df['amount'].abs() > upper_bound]) / len(df)) * 100
            },
            'large_transactions': {
                'count': len(df[df['amount'].abs() > percentile_threshold]),
                'threshold': percentile_threshold,
                'percentage': (len(df[df['amount'].abs() > percentile_threshold]) / len(df)) * 100
            }
        }

        # Find actual outlier transactions (top 10 by absolute amount)
        outlier_transactions = df.nlargest(10, df['amount'].abs().name if 'amount' in df.columns else 'amount')

        if not outlier_transactions.empty:
            outlier_list = []
            for _, row in outlier_transactions.iterrows():
                outlier_list.append({
                    'date': row.get('transaction_date'),
                    'account': row.get('account_number'),
                    'description': row.get('main_description', '')[:50],  # Truncate for readability
                    'amount': row.get('amount'),
                    'abs_amount': abs(row.get('amount', 0))
                })

            outlier_stats['top_outliers'] = outlier_list

        # Duplicate transaction detection
        if len(df.columns) >= 3:
            # Look for potential duplicates (same date, amount, and similar description)
            duplicate_candidates = df[
                df.duplicated(subset=['transaction_date', 'amount'], keep=False)
            ].copy() if 'transaction_date' in df.columns else pd.DataFrame()

            outlier_stats['potential_duplicates'] = {
                'count': len(duplicate_candidates),
                'percentage': (len(duplicate_candidates) / len(df)) * 100 if len(df) > 0 else 0
            }

        return outlier_stats

    def cash_flow_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cash flow patterns"""
        if 'amount' not in df.columns:
            return {'error': 'No amount column found'}

        # Separate income and expenses
        income_transactions = df[df['amount'] > 0]
        expense_transactions = df[df['amount'] < 0]

        cash_flow_stats = {
            'summary': {
                'total_income': income_transactions['amount'].sum() if not income_transactions.empty else 0,
                'total_expenses': expense_transactions['amount'].sum() if not expense_transactions.empty else 0,
                'net_cash_flow': df['amount'].sum(),
                'income_transaction_count': len(income_transactions),
                'expense_transaction_count': len(expense_transactions)
            }
        }

        # Cash flow by time period
        if 'transaction_date' in df.columns:
            df_temp = df.copy()
            df_temp['year_month'] = df_temp['transaction_date'].dt.to_period('M')

            monthly_flow = df_temp.groupby('year_month')['amount'].agg([
                lambda x: x[x > 0].sum(),  # Income
                lambda x: x[x < 0].sum(),  # Expenses  
                'sum'  # Net
            ]).round(2)

            monthly_flow.columns = ['monthly_income', 'monthly_expenses', 'monthly_net']
            cash_flow_stats['monthly_flow'] = monthly_flow.to_dict('index')

            # Calculate running balance if balance column exists
            if 'balance' in df.columns:
                balance_validation = self._validate_running_balance(df_temp)
                cash_flow_stats['balance_validation'] = balance_validation

        return cash_flow_stats

    def _validate_running_balance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that running balance calculations are correct"""
        validation_results = {}

        if 'account_number' in df.columns:
            # Validate per account
            for account in df['account_number'].unique():
                account_data = df[df['account_number'] == account].sort_values('transaction_date')

                if len(account_data) > 1 and 'balance' in account_data.columns:
                    # Calculate expected balance
                    account_data = account_data.copy()
                    account_data['calculated_balance'] = account_data['balance'].iloc[0] - account_data['amount'].iloc[
                        0] + account_data['amount'].cumsum()

                    # Compare with actual balance
                    balance_differences = abs(account_data['balance'] - account_data['calculated_balance'])
                    max_difference = balance_differences.max()

                    validation_results[f'account_{account}'] = {
                        'max_difference': max_difference,
                        'avg_difference': balance_differences.mean(),
                        'is_valid': max_difference < 0.01,  # Allow for rounding differences
                        'error_count': (balance_differences > 0.01).sum()
                    }

        return validation_results

    def trend_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in transaction data"""
        if 'transaction_date' not in df.columns or 'amount' not in df.columns:
            return {'error': 'Missing required columns for trend analysis'}

        df_temp = df.copy()

        # Daily trends
        daily_trends = df_temp.groupby('transaction_date').agg({
            'amount': ['count', 'sum', 'mean']
        }).round(2)
        daily_trends.columns = ['daily_transaction_count', 'daily_total_amount', 'daily_avg_amount']

        # Weekly trends
        df_temp['week'] = df_temp['transaction_date'].dt.to_period('W')
        weekly_trends = df_temp.groupby('week').agg({
            'amount': ['count', 'sum', 'mean']
        }).round(2)
        weekly_trends.columns = ['weekly_transaction_count', 'weekly_total_amount', 'weekly_avg_amount']

        # Monthly trends  
        df_temp['month'] = df_temp['transaction_date'].dt.to_period('M')
        monthly_trends = df_temp.groupby('month').agg({
            'amount': ['count', 'sum', 'mean']
        }).round(2)
        monthly_trends.columns = ['monthly_transaction_count', 'monthly_total_amount', 'monthly_avg_amount']

        # Calculate growth rates
        trend_stats = {
            'daily_trends': daily_trends.tail(30).to_dict('index'),  # Last 30 days
            'weekly_trends': weekly_trends.tail(12).to_dict('index'),  # Last 12 weeks
            'monthly_trends': monthly_trends.to_dict('index')  # All months
        }

        # Growth calculations
        if len(monthly_trends) >= 2:
            latest_month = monthly_trends['monthly_total_amount'].iloc[-1]
            previous_month = monthly_trends['monthly_total_amount'].iloc[-2]

            if previous_month != 0:
                month_over_month = ((latest_month - previous_month) / abs(previous_month)) * 100
                trend_stats['growth_metrics'] = {
                    'month_over_month_change': round(month_over_month, 2),
                    'latest_month_amount': latest_month,
                    'previous_month_amount': previous_month
                }

        return trend_stats

    def active_inactive_accounts_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Smart context-aware analysis for active vs inactive accounts"""
        if 'account_number' not in df.columns or 'transaction_date' not in df.columns:
            return {'error': 'Missing required columns for account activity analysis'}

        # Get data context
        data_start = df['transaction_date'].min()
        data_end = df['transaction_date'].max()
        current_date = datetime.now()

        # Determine if we're analyzing current or historical data
        days_since_data_end = (current_date - data_end).days
        is_historical_data = days_since_data_end > 30  # More than 30 days old

        # Choose appropriate reference date and thresholds
        if is_historical_data:
            # Historical data: use the end of data period as reference
            reference_date = data_end
            analysis_context = "Historical"
            self._log('info', f"Detected historical data (latest: {data_end.date()}, {days_since_data_end} days ago)")
        else:
            # Current data: use today as reference
            reference_date = current_date
            analysis_context = "Current"
            self._log('info', f"Detected current data (latest: {data_end.date()}, {days_since_data_end} days ago)")

        # Get configurable thresholds
        active_threshold_days = ANALYSIS_SETTINGS.get('active_account_days', 90)
        data_period_days = (data_end - data_start).days

        # Adaptive threshold: for short data periods, use a percentage of the period
        if data_period_days < active_threshold_days:
            adaptive_threshold = max(30, data_period_days * 0.33)  # At least 30 days or 33% of period
            self._log('info',
                      f"Short data period ({data_period_days} days), using adaptive threshold: {adaptive_threshold:.0f} days")
        else:
            adaptive_threshold = active_threshold_days

        # Calculate cutoff dates
        activity_cutoff = reference_date - timedelta(days=adaptive_threshold)
        recent_activity_cutoff = reference_date - timedelta(days=30)  # Last month activity

        account_activity = []
        accounts = df['account_number'].unique()

        for account in accounts:
            account_data = df[df['account_number'] == account]

            # Basic metrics
            last_transaction_date = account_data['transaction_date'].max()
            first_transaction_date = account_data['transaction_date'].min()
            total_transactions = len(account_data)

            # Time-based calculations
            days_since_last = (reference_date - last_transaction_date).days
            days_since_first = (reference_date - first_transaction_date).days
            account_active_period = (last_transaction_date - first_transaction_date).days + 1

            # Activity metrics
            recent_transactions = len(account_data[
                                          account_data['transaction_date'] > activity_cutoff
                                          ])
            very_recent_transactions = len(account_data[
                                               account_data['transaction_date'] > recent_activity_cutoff
                                               ])

            # Multi-criteria activity determination
            criteria_met = 0
            activity_criteria = []

            # Criterion 1: Recent transactions within threshold
            if days_since_last <= adaptive_threshold:
                criteria_met += 1
                activity_criteria.append("Recent transactions")

            # Criterion 2: Has transactions in the active period
            if recent_transactions > 0:
                criteria_met += 1
                activity_criteria.append("Activity in threshold period")

            # Criterion 3: Consistent activity (for accounts with longer history)
            if account_active_period > 30:
                transaction_frequency = total_transactions / max(1, account_active_period)
                if transaction_frequency >= 0.1:  # At least 1 transaction per 10 days average
                    criteria_met += 1
                    activity_criteria.append("Consistent activity pattern")

            # Determine status based on criteria met
            is_active = criteria_met >= 2 or (criteria_met >= 1 and total_transactions >= 10)
            confidence_score = min(100, (criteria_met / 3) * 100 + (min(total_transactions, 50) / 50) * 20)

            # Financial metrics
            total_amount = account_data['amount'].sum() if 'amount' in account_data.columns else 0
            avg_transaction = account_data['amount'].mean() if 'amount' in account_data.columns else 0

            # Activity intensity score (transactions per active day)
            activity_intensity = total_transactions / max(1, account_active_period)

            account_activity.append({
                'account_number': account,
                'status': 'Active' if is_active else 'Inactive',
                'confidence_score': round(confidence_score, 1),
                'total_transactions': total_transactions,
                'recent_transactions_90_days': recent_transactions,
                'very_recent_transactions_30_days': very_recent_transactions,
                'days_since_last_transaction': days_since_last,
                'days_since_first_transaction': days_since_first,
                'account_active_period_days': account_active_period,
                'activity_intensity': round(activity_intensity, 3),
                'last_transaction_date': last_transaction_date,
                'first_transaction_date': first_transaction_date,
                'total_amount': round(total_amount, 2) if total_amount else 0,
                'avg_transaction_amount': round(avg_transaction, 2) if avg_transaction else 0,
                'is_active': is_active,
                'activity_criteria_met': activity_criteria,
                'criteria_score': f"{criteria_met}/3"
            })

        # Enhanced summary statistics
        active_accounts = [acc for acc in account_activity if acc['is_active']]
        inactive_accounts = [acc for acc in account_activity if not acc['is_active']]

        # Calculate activity distribution
        high_confidence_active = [acc for acc in active_accounts if acc['confidence_score'] >= 80]
        moderate_confidence_active = [acc for acc in active_accounts if 50 <= acc['confidence_score'] < 80]
        low_confidence_active = [acc for acc in active_accounts if acc['confidence_score'] < 50]

        summary = {
            'analysis_context': analysis_context,
            'reference_date': reference_date,
            'data_period': {
                'start_date': data_start,
                'end_date': data_end,
                'period_days': data_period_days,
                'days_since_data_end': days_since_data_end
            },
            'thresholds': {
                'configured_threshold_days': active_threshold_days,
                'applied_threshold_days': int(adaptive_threshold),
                'is_adaptive': data_period_days < active_threshold_days
            },
            'total_accounts': len(accounts),
            'active_accounts': len(active_accounts),
            'inactive_accounts': len(inactive_accounts),
            'active_percentage': round((len(active_accounts) / len(accounts)) * 100, 2) if len(accounts) > 0 else 0,
            'inactive_percentage': round((len(inactive_accounts) / len(accounts)) * 100, 2) if len(accounts) > 0 else 0,
            'confidence_distribution': {
                'high_confidence_active': len(high_confidence_active),
                'moderate_confidence_active': len(moderate_confidence_active),
                'low_confidence_active': len(low_confidence_active)
            }
        }

        return {
            'summary': summary,
            'account_details': account_activity,
            'active_accounts': active_accounts,
            'inactive_accounts': inactive_accounts,
            'methodology': {
                'description': 'Smart context-aware analysis using multiple activity criteria',
                'criteria': [
                    'Recent transactions within threshold period',
                    'Activity in recent period',
                    'Consistent transaction pattern over time'
                ],
                'adaptive_threshold': 'Threshold adjusts based on data period length',
                'confidence_scoring': 'Based on criteria met and transaction volume'
            }
        }

    def data_quality_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality issues"""
        quality_stats = {
            'missing_data': {},
            'data_consistency': {},
            'potential_issues': []
        }

        # Missing data analysis
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100

            quality_stats['missing_data'][column] = {
                'missing_count': missing_count,
                'missing_percentage': round(missing_percentage, 2),
                'has_data': missing_percentage < 100
            }

        # Data consistency checks
        if 'amount' in df.columns:
            # Check for zero amounts
            zero_amounts = (df['amount'] == 0).sum()
            quality_stats['data_consistency']['zero_amounts'] = {
                'count': zero_amounts,
                'percentage': (zero_amounts / len(df)) * 100
            }

            # Check for extremely large amounts (potential data entry errors)
            amount_std = df['amount'].std()
            amount_mean = df['amount'].mean()
            extreme_amounts = df[abs(df['amount'] - amount_mean) > 5 * amount_std]

            quality_stats['data_consistency']['extreme_amounts'] = {
                'count': len(extreme_amounts),
                'threshold': 5 * amount_std,
                'percentage': (len(extreme_amounts) / len(df)) * 100
            }

        # Currency consistency
        if 'currency' in df.columns:
            currency_distribution = df['currency'].value_counts()
            quality_stats['data_consistency']['currency_distribution'] = currency_distribution.to_dict()

            if len(currency_distribution) > 1:
                quality_stats['potential_issues'].append('Multiple currencies detected')

        # Account number consistency
        if 'account_number' in df.columns:
            account_distribution = df['account_number'].value_counts()
            quality_stats['data_consistency']['account_distribution'] = account_distribution.to_dict()

        return quality_stats

    def comprehensive_analysis(self, df: pd.DataFrame, start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive analysis on the dataset
        
        Args:
            df: Transaction dataframe
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            
        Returns:
            Complete analysis results
        """
        self._log('info', f"Starting comprehensive analysis of {len(df):,} transactions")

        # Apply date filtering if requested
        if start_date or end_date:
            df = self._apply_date_filter(df, start_date, end_date)

        # Run all analyses
        results = {
            'analysis_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'dataset_size': len(df),
                'date_filter_applied': bool(start_date or end_date),
                'filtered_date_range': {
                    'start_date': start_date,
                    'end_date': end_date
                }
            }
        }

        # Execute analyses
        try:
            results['basic_statistics'] = self.basic_statistics(df)
            self._log('info', "✓ Basic statistics completed")

            results['account_analysis'] = self.account_analysis(df)
            self._log('info', "✓ Account analysis completed")

            results['merchant_analysis'] = self.merchant_analysis(df)
            self._log('info', "✓ Merchant analysis completed")

            results['temporal_analysis'] = self.temporal_analysis(df)
            self._log('info', "✓ Temporal analysis completed")

            results['outlier_detection'] = self.outlier_detection(df)
            self._log('info', "✓ Outlier detection completed")

            results['cash_flow_analysis'] = self.cash_flow_analysis(df)
            self._log('info', "✓ Cash flow analysis completed")

            results['data_quality'] = self.data_quality_analysis(df)
            self._log('info', "✓ Data quality analysis completed")

            # Category analysis (if categories exist)
            if 'auto_category' in df.columns:
                results['category_analysis'] = self.category_analysis(df)
                self._log('info', "✓ Category analysis completed")

            # Active/Inactive accounts analysis
            results['active_inactive_accounts'] = self.active_inactive_accounts_analysis(df)
            self._log('info', "✓ Active/Inactive accounts analysis completed")

        except Exception as e:
            self._log('error', f"Error during analysis: {e}")
            results['analysis_error'] = str(e)

        self._log('info', "Comprehensive analysis completed")
        return results


def analyze_transactions(df: pd.DataFrame, start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function for transaction analysis"""
    analyzer = TransactionAnalyzer()
    return analyzer.comprehensive_analysis(df, start_date, end_date)
