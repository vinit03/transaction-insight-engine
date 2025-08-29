"""
Intelligent Transaction Categorizer
Automatically learns patterns from transaction data without business assumptions
"""

import logging
import os
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, List, Set, Any, Optional

import pandas as pd

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.settings import CATEGORIZATION_SETTINGS
from src.decimal_utils import decimal_mean, decimal_to_float, decimal_comparison_ops


class TransactionCategorizer:
    """Intelligent categorizer that learns from transaction patterns"""

    def __init__(self, enable_logging: bool = True):
        self.logger = self._setup_logging() if enable_logging else None
        self.categories = {}
        self.merchant_patterns = {}
        self.amount_patterns = {}
        self.learned_rules = []
        self.similarity_cache = {}

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

    def _clean_merchant_name(self, name: str) -> str:
        """Clean and normalize merchant names"""
        if pd.isna(name) or name == '':
            return 'UNKNOWN'

        # Convert to string and uppercase
        name = str(name).upper().strip()

        # Remove common noise words/characters
        noise_patterns = [
            r'\\n', r'\\t', r'\\r',  # Escape sequences
            r'\s+CD\s+\d+',  # CD followed by numbers
            r'\s+T/A\s+',  # Trading as
            r'\s+LTD$',  # Ltd at end
            r'\s+LIMITED$',  # Limited at end
            r'\s+PLC$',  # PLC at end
            r'\s+-\s+',  # Dash with spaces
        ]

        for pattern in noise_patterns:
            name = re.sub(pattern, ' ', name)

        # Clean multiple spaces
        name = re.sub(r'\s+', ' ', name).strip()

        return name

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings with caching"""
        cache_key = f"{str1}|{str2}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        similarity = SequenceMatcher(None, str1, str2).ratio()
        self.similarity_cache[cache_key] = similarity
        return similarity

    def _find_similar_merchants(self, merchant: str, existing_merchants: List[str],
                                threshold: float = None) -> Optional[str]:
        """Find similar merchant names above threshold"""
        if threshold is None:
            threshold = CATEGORIZATION_SETTINGS.get('similarity_threshold', 0.85)

        best_match = None
        best_score = threshold

        for existing in existing_merchants:
            similarity = self._calculate_similarity(merchant, existing)
            if similarity > best_score:
                best_match = existing
                best_score = similarity

        return best_match

    def _extract_keywords(self, description: str) -> Set[str]:
        """Extract meaningful keywords from transaction description"""
        if pd.isna(description) or description == '':
            return set()

        # Clean and split
        words = str(description).upper().split()

        # Filter out common noise words
        stop_words = {
            'THE', 'AND', 'OR', 'BUT', 'IN', 'ON', 'AT', 'TO', 'FOR', 'OF', 'WITH',
            'BY', 'FROM', 'UP', 'ABOUT', 'INTO', 'OVER', 'AFTER', 'LTD', 'LIMITED',
            'PLC', 'CORP', 'INC', 'CO', 'COMPANY'
        }

        keywords = set()
        for word in words:
            # Remove special characters
            clean_word = re.sub(r'[^\w]', '', word)

            # Keep meaningful words (3+ chars, not stop words, not pure numbers)
            if (len(clean_word) >= 3 and
                    clean_word not in stop_words and
                    not clean_word.isdigit()):
                keywords.add(clean_word)

        return keywords

    def _analyze_transaction_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze transaction patterns to understand the data"""
        patterns = {
            'total_transactions': len(df),
            'unique_merchants': df['main_description'].nunique(),
            'amount_distribution': {},
            'merchant_frequency': {},
            'temporal_patterns': {},
            'amount_ranges': {}
        }

        # Analyze amount patterns with Decimal support
        if 'amount' in df.columns:
            amounts = df['amount'].dropna()

            # Use Decimal-aware comparisons
            positive_mask = decimal_comparison_ops(df, 'amount', 0, '>')
            negative_mask = decimal_comparison_ops(df, 'amount', 0, '<')
            zero_mask = decimal_comparison_ops(df, 'amount', 0, '==')

            positive_amounts = amounts[positive_mask[amounts.index]] if positive_mask.any() else pd.Series([],
                                                                                                           dtype=object)
            negative_amounts = amounts[negative_mask[amounts.index]] if negative_mask.any() else pd.Series([],
                                                                                                           dtype=object)

            patterns['amount_distribution'] = {
                'positive_count': positive_mask.sum(),
                'negative_count': negative_mask.sum(),
                'zero_count': zero_mask.sum(),
                'mean_positive': decimal_to_float(decimal_mean(positive_amounts)) if not positive_amounts.empty else 0,
                'mean_negative': decimal_to_float(decimal_mean(negative_amounts)) if not negative_amounts.empty else 0,
                # For percentiles, convert to float temporarily for numpy calculation, then back
                'percentiles': amounts.apply(decimal_to_float).quantile(
                    [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict() if not amounts.empty else {}
            }

        # Analyze merchant frequency
        merchant_counts = df['main_description'].value_counts()
        patterns['merchant_frequency'] = {
            'top_merchants': merchant_counts.head(CATEGORIZATION_SETTINGS.get('top_merchants_count', 20)).to_dict(),
            'single_occurrence': (merchant_counts == 1).sum(),
            'frequent_merchants': (
                    merchant_counts >= CATEGORIZATION_SETTINGS.get('min_frequency_for_category', 5)).sum()
        }

        # Temporal patterns
        if 'transaction_date' in df.columns:
            df_temp = df.copy()
            df_temp['month'] = df_temp['transaction_date'].dt.month
            df_temp['day_of_week'] = df_temp['transaction_date'].dt.day_of_week
            patterns['temporal_patterns'] = {
                'monthly_distribution': df_temp['month'].value_counts().to_dict(),
                'weekday_distribution': df_temp['day_of_week'].value_counts().to_dict()
            }

        return patterns

    def _create_automatic_categories(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Create categories automatically based on transaction patterns"""
        categories = {}

        # Clean merchant names
        df = df.copy()
        df['clean_merchant'] = df['main_description'].apply(self._clean_merchant_name)

        # Get frequent merchants
        merchant_counts = df['clean_merchant'].value_counts()
        frequent_merchants = merchant_counts[
            merchant_counts >= CATEGORIZATION_SETTINGS.get('min_frequency_for_category', 5)
            ]

        self._log('info', f"Found {len(frequent_merchants)} frequent merchants")

        # Group similar merchants
        processed_merchants = set()
        category_counter = 1

        for merchant, count in frequent_merchants.items():
            if merchant in processed_merchants:
                continue

            # Find similar merchants
            similar_merchants = [merchant]

            if CATEGORIZATION_SETTINGS.get('enable_smart_grouping', True):
                for other_merchant in frequent_merchants.index:
                    if other_merchant != merchant and other_merchant not in processed_merchants:
                        if self._calculate_similarity(merchant, other_merchant) > CATEGORIZATION_SETTINGS.get(
                                'similarity_threshold', 0.85):
                            similar_merchants.append(other_merchant)
                            processed_merchants.add(other_merchant)

            processed_merchants.add(merchant)

            # Create category
            category_name = self._generate_category_name(merchant, similar_merchants)

            # Analyze transaction patterns for this category
            category_transactions = df[df['clean_merchant'].isin(similar_merchants)]

            categories[category_name] = {
                'merchants': similar_merchants,
                'transaction_count': len(category_transactions),
                'avg_amount': decimal_to_float(decimal_mean(category_transactions[
                                                                'amount'])) if 'amount' in category_transactions and not category_transactions.empty else 0,
                'amount_pattern': self._determine_amount_pattern(category_transactions),
                'keywords': self._extract_common_keywords(similar_merchants),
                'frequency_score': count / len(df),
                'category_id': category_counter
            }

            category_counter += 1

            # Limit number of auto-generated categories
            if len(categories) >= CATEGORIZATION_SETTINGS.get('max_auto_categories', 100):
                break

        self._log('info', f"Created {len(categories)} automatic categories")
        return categories

    def _generate_category_name(self, primary_merchant: str, similar_merchants: List[str]) -> str:
        """Generate a meaningful category name from merchant names"""

        # Extract common words
        all_words = []
        for merchant in similar_merchants:
            words = merchant.split()
            all_words.extend(words)

        # Find most common meaningful words
        word_counts = Counter(all_words)

        # Filter out noise
        meaningful_words = []
        for word, count in word_counts.most_common():
            if len(word) >= 3 and not word.isdigit():
                meaningful_words.append(word)
                if len(meaningful_words) >= 3:  # Limit to top 3 words
                    break

        if meaningful_words:
            category_name = '_'.join(meaningful_words)
        else:
            # Fallback to primary merchant name
            category_name = primary_merchant.replace(' ', '_')

        # Ensure reasonable length
        if len(category_name) > 30:
            category_name = category_name[:30]

        return category_name.upper()

    def _determine_amount_pattern(self, transactions: pd.DataFrame) -> str:
        """Determine the typical amount pattern for transactions"""
        if 'amount' not in transactions.columns or transactions.empty:
            return 'unknown'

        positive_count = (transactions['amount'] > 0).sum()
        negative_count = (transactions['amount'] < 0).sum()
        total_count = len(transactions)

        if positive_count / total_count > 0.8:
            return 'mostly_positive'
        elif negative_count / total_count > 0.8:
            return 'mostly_negative'
        else:
            return 'mixed'

    def _extract_common_keywords(self, merchants: List[str]) -> List[str]:
        """Extract common keywords from merchant names"""
        all_keywords = set()
        for merchant in merchants:
            all_keywords.update(self._extract_keywords(merchant))

        # Return most common keywords
        return list(all_keywords)[:10]  # Limit to top 10

    def categorize_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize all transactions in the dataframe
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with added 'auto_category' column
        """
        self._log('info', f"Starting categorization of {len(df):,} transactions")

        # Analyze patterns
        patterns = self._analyze_transaction_patterns(df)
        self._log('info', f"Found {patterns['unique_merchants']:,} unique merchants")

        # Create automatic categories
        self.categories = self._create_automatic_categories(df)

        # Apply categorization
        df_categorized = df.copy()
        df_categorized['auto_category'] = 'UNCATEGORIZED'
        df_categorized['category_confidence'] = 0.0
        df_categorized['clean_merchant'] = df_categorized['main_description'].apply(self._clean_merchant_name)

        # Apply categories
        for category_name, category_info in self.categories.items():
            mask = df_categorized['clean_merchant'].isin(category_info['merchants'])
            df_categorized.loc[mask, 'auto_category'] = category_name
            df_categorized.loc[mask, 'category_confidence'] = 1.0  # High confidence for exact matches

        # Handle remaining uncategorized transactions with fuzzy matching
        uncategorized_mask = df_categorized['auto_category'] == 'UNCATEGORIZED'
        uncategorized_count = uncategorized_mask.sum()

        if uncategorized_count > 0:
            self._log('info', f"Applying fuzzy matching to {uncategorized_count:,} uncategorized transactions")

            # Get all known merchants
            all_known_merchants = []
            for category_info in self.categories.values():
                all_known_merchants.extend(category_info['merchants'])

            # Apply fuzzy matching
            for idx in df_categorized[uncategorized_mask].index:
                merchant = df_categorized.loc[idx, 'clean_merchant']

                # Skip one-off transactions if configured
                if CATEGORIZATION_SETTINGS.get('exclude_one_off_transactions', True):
                    merchant_count = (df_categorized['clean_merchant'] == merchant).sum()
                    if merchant_count < CATEGORIZATION_SETTINGS.get('min_frequency_for_category', 5):
                        continue

                # Find similar merchant
                similar_merchant = self._find_similar_merchants(merchant, all_known_merchants)

                if similar_merchant:
                    # Find which category this similar merchant belongs to
                    for category_name, category_info in self.categories.items():
                        if similar_merchant in category_info['merchants']:
                            df_categorized.loc[idx, 'auto_category'] = category_name
                            df_categorized.loc[idx, 'category_confidence'] = 0.7  # Lower confidence for fuzzy matches
                            break

        # Generate statistics
        categorization_stats = self._generate_categorization_stats(df_categorized)
        self._log('info', f"Categorization complete: {categorization_stats['categorized_percentage']:.1f}% categorized")

        # Drop temporary column
        df_categorized = df_categorized.drop('clean_merchant', axis=1)

        return df_categorized

    def _generate_categorization_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate categorization statistics"""
        total_transactions = len(df)
        categorized_transactions = (df['auto_category'] != 'UNCATEGORIZED').sum()

        stats = {
            'total_transactions': total_transactions,
            'categorized_transactions': categorized_transactions,
            'uncategorized_transactions': total_transactions - categorized_transactions,
            'categorized_percentage': (
                                              categorized_transactions / total_transactions) * 100 if total_transactions > 0 else 0,
            'categories_created': len(self.categories),
            'category_distribution': df['auto_category'].value_counts().to_dict()
        }

        return stats

    def get_category_summary(self) -> pd.DataFrame:
        """Get summary of created categories"""
        if not self.categories:
            return pd.DataFrame()

        summary_data = []
        for category_name, category_info in self.categories.items():
            summary_data.append({
                'category_name': category_name,
                'merchant_count': len(category_info['merchants']),
                'transaction_count': category_info['transaction_count'],
                'avg_amount': category_info['avg_amount'],
                'amount_pattern': category_info['amount_pattern'],
                'frequency_score': category_info['frequency_score'],
                'sample_merchants': ', '.join(category_info['merchants'][:3])  # Show first 3
            })

        return pd.DataFrame(summary_data).sort_values('transaction_count', ascending=False)


def categorize_dataframe(df: pd.DataFrame, export_categories: bool = False,
                         categories_file: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to categorize a dataframe"""
    categorizer = TransactionCategorizer()

    # Load manual categories if provided
    if categories_file and os.path.exists(categories_file):
        # Note: manual category loading would be implemented if needed
        pass

    # Categorize transactions
    df_categorized = categorizer.categorize_transactions(df)

    # Export categories for review if requested
    if export_categories:
        export_path = 'output/auto_categories_review.xlsx'
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        # Note: export functionality would be implemented if needed

    return df_categorized
