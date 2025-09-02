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
        # Initialize logger for categorization progress tracking and debugging
        self.logger = self._setup_logging() if enable_logging else None
        # Storage for discovered categories and their characteristics
        self.categories = {}
        # Pattern recognition for merchant name variations and groupings
        self.merchant_patterns = {}
        # Amount-based patterns for identifying recurring payments and subscriptions
        self.amount_patterns = {}
        # Machine learning-like rules discovered during categorization process
        self.learned_rules = []
        # Performance cache for string similarity calculations (expensive operations)
        self.similarity_cache = {}

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration for categorization operations"""
        # Configure logging for pattern recognition progress and category discovery
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _log(self, level: str, message: str) -> None:
        """Safe logging method that handles disabled logging gracefully"""
        if self.logger:
            # Dynamically call the appropriate logging level method
            getattr(self.logger, level.lower())(message)

    def _clean_merchant_name(self, name: str) -> str:
        """MERCHANT NAME NORMALIZATION: Clean and standardize merchant names for grouping"""
        # MISSING DATA HANDLING: Provide default for missing merchant information
        if pd.isna(name) or name == '':
            return 'UNKNOWN'

        # CASE NORMALIZATION: Convert to uppercase for consistent comparison
        name = str(name).upper().strip()

        # NOISE REMOVAL: Remove common noise patterns that obscure merchant identity
        # These patterns are common in transaction descriptions but don't help identify the merchant
        noise_patterns = [
            r'\\n', r'\\t', r'\\r',    # Escape sequences from data corruption
            r'\s+CD\s+\d+',            # Card/Check digits (CD 123)
            r'\s+T/A\s+',              # Trading as indicators
            r'\s+LTD$',                # Company type suffixes
            r'\s+LIMITED$',
            r'\s+PLC$',
            r'\s+INC$',
            r'\s+CORP$',
            r'\s+-\s+',                # Dashes with spaces (often separators)
            r'\s+\d{2}/\d{2}\s+',      # Date patterns (MM/DD)
            r'\s+\d{4}\s+',            # Year patterns
        ]

        # PATTERN CLEANING: Apply each noise removal pattern
        for pattern in noise_patterns:
            name = re.sub(pattern, ' ', name)

        # WHITESPACE NORMALIZATION: Clean up multiple spaces and trim
        name = re.sub(r'\s+', ' ', name).strip()

        # MINIMUM LENGTH CHECK: Ensure meaningful merchant names
        if len(name) < 2:
            return 'UNKNOWN'

        return name

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """SIMILARITY CALCULATION: Compute string similarity with performance caching"""
        # CACHE OPTIMIZATION: Avoid recalculating expensive similarity operations
        # String similarity calculation is O(n*m) complexity, so caching provides major performance gains
        cache_key = f"{str1}|{str2}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        # SEQUENCE MATCHING: Use difflib's ratio for accurate string similarity
        # This algorithm considers both common subsequences and overall string structure
        # Returns value from 0.0 (completely different) to 1.0 (identical)
        similarity = SequenceMatcher(None, str1, str2).ratio()
        
        # CACHE STORAGE: Store result for future use
        self.similarity_cache[cache_key] = similarity
        return similarity

    def _find_similar_merchants(self, merchant: str, existing_merchants: List[str],
                                threshold: float = None) -> Optional[str]:
        """SIMILARITY MATCHING: Find existing merchants similar to the current one"""
        # THRESHOLD CONFIGURATION: Use configurable similarity threshold for flexibility
        if threshold is None:
            threshold = CATEGORIZATION_SETTINGS.get('similarity_threshold', 0.85)

        best_match = None
        best_score = threshold  # Start with threshold as minimum acceptable score

        # SIMILARITY SEARCH: Compare against all existing merchants to find best match
        for existing in existing_merchants:
            similarity = self._calculate_similarity(merchant, existing)
            
            # BEST MATCH TRACKING: Keep track of highest similarity score above threshold
            if similarity > best_score:
                best_match = existing
                best_score = similarity

        # MATCH RESULT: Return best match if found, otherwise None
        # This enables merchant grouping: "AMAZON.COM" and "AMAZON SERVICES" -> "AMAZON.COM"
        return best_match

    def _extract_keywords(self, description: str) -> Set[str]:
        """KEYWORD EXTRACTION: Extract meaningful keywords for pattern recognition"""
        # MISSING DATA HANDLING: Return empty set for invalid descriptions
        if pd.isna(description) or description == '':
            return set()

        # TEXT PREPROCESSING: Convert to uppercase and split into words
        words = str(description).upper().split()

        # STOP WORDS FILTERING: Remove common words that don't add categorization value
        # These words appear frequently but don't help distinguish transaction types
        stop_words = {
            'THE', 'AND', 'OR', 'BUT', 'IN', 'ON', 'AT', 'TO', 'FOR', 'OF', 'WITH',
            'BY', 'FROM', 'UP', 'ABOUT', 'INTO', 'OVER', 'AFTER', 'LTD', 'LIMITED',
            'PLC', 'CORP', 'INC', 'CO', 'COMPANY', 'PAYMENT', 'TRANSFER', 'ONLINE',
            'CARD', 'DEBIT', 'CREDIT', 'TRANSACTION', 'PURCHASE'
        }

        keywords = set()
        for word in words:
            # CHARACTER CLEANING: Remove special characters and punctuation
            clean_word = re.sub(r'[^\w]', '', word)

            # QUALITY FILTERING: Keep only meaningful words for categorization
            # Criteria: minimum length, not a stop word, not pure numbers
            if (len(clean_word) >= 3 and              # Minimum 3 characters (avoid noise)
                    clean_word not in stop_words and   # Not a common stop word
                    not clean_word.isdigit() and       # Not a pure number
                    not re.match(r'^\d+[A-Z]*$', clean_word)):  # Not alphanumeric codes
                keywords.add(clean_word)

        return keywords

    def _analyze_transaction_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """PATTERN ANALYSIS: Discover transaction patterns for intelligent categorization"""
        # PATTERN STORAGE: Initialize structure to hold discovered patterns
        patterns = {
            'total_transactions': len(df),
            'unique_merchants': df['main_description'].nunique(),
            'amount_distribution': {},
            'merchant_frequency': {},
            'temporal_patterns': {},
            'amount_ranges': {}
        }

        # AMOUNT PATTERN ANALYSIS: Understand financial flow characteristics
        # This helps identify categories based on transaction directions and magnitudes
        if 'amount' in df.columns:
            amounts = df['amount'].dropna()

            # DECIMAL-AWARE COMPARISONS: Use precise decimal operations for financial data
            positive_mask = decimal_comparison_ops(df, 'amount', 0, '>')
            negative_mask = decimal_comparison_ops(df, 'amount', 0, '<')
            zero_mask = decimal_comparison_ops(df, 'amount', 0, '==')

            # AMOUNT SEGMENTATION: Separate positive and negative amounts for analysis
            positive_amounts = amounts[positive_mask[amounts.index]] if positive_mask.any() else pd.Series([], dtype=object)
            negative_amounts = amounts[negative_mask[amounts.index]] if negative_mask.any() else pd.Series([], dtype=object)

            # DISTRIBUTION ANALYSIS: Calculate key statistics for categorization decisions
            patterns['amount_distribution'] = {
                'positive_count': positive_mask.sum(),          # Income transactions
                'negative_count': negative_mask.sum(),          # Expense transactions
                'zero_count': zero_mask.sum(),                  # Zero-amount transactions
                'mean_positive': decimal_to_float(decimal_mean(positive_amounts)) if not positive_amounts.empty else 0,
                'mean_negative': decimal_to_float(decimal_mean(negative_amounts)) if not negative_amounts.empty else 0,
                # PERCENTILE ANALYSIS: Understand amount distributions for outlier detection
                'percentiles': amounts.apply(decimal_to_float).quantile(
                    [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict() if not amounts.empty else {}
            }

        # MERCHANT FREQUENCY ANALYSIS: Identify patterns in merchant interactions
        # High-frequency merchants are candidates for automatic categorization
        merchant_counts = df['main_description'].value_counts()
        patterns['merchant_frequency'] = {
            'top_merchants': merchant_counts.head(CATEGORIZATION_SETTINGS.get('top_merchants_count', 20)).to_dict(),
            'single_occurrence': (merchant_counts == 1).sum(),  # One-off transactions
            'frequent_merchants': (merchant_counts >= CATEGORIZATION_SETTINGS.get('min_frequency_for_category', 5)).sum()  # Recurring merchants
        }

        # TEMPORAL PATTERN ANALYSIS: Discover time-based transaction patterns
        # This can help identify recurring subscriptions, payroll, etc.
        if 'transaction_date' in df.columns:
            df_temp = df.copy()
            df_temp['month'] = df_temp['transaction_date'].dt.month
            df_temp['day_of_week'] = df_temp['transaction_date'].dt.day_of_week
            patterns['temporal_patterns'] = {
                'monthly_distribution': df_temp['month'].value_counts().to_dict(),      # Seasonal patterns
                'weekday_distribution': df_temp['day_of_week'].value_counts().to_dict() # Weekly patterns
            }

        return patterns

    def _create_automatic_categories(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """AUTOMATIC CATEGORY CREATION: Generate categories from transaction patterns"""
        categories = {}

        # MERCHANT NORMALIZATION: Clean merchant names for consistent grouping
        df = df.copy()
        df['clean_merchant'] = df['main_description'].apply(self._clean_merchant_name)

        # FREQUENCY ANALYSIS: Identify merchants with sufficient transaction volume
        # Only frequent merchants become category candidates (reduces noise)
        merchant_counts = df['clean_merchant'].value_counts()
        min_frequency = CATEGORIZATION_SETTINGS.get('min_frequency_for_category', 5)
        frequent_merchants = merchant_counts[merchant_counts >= min_frequency]

        self._log('info', f"Found {len(frequent_merchants)} frequent merchants (>= {min_frequency} transactions)")

        # SIMILARITY GROUPING: Group similar merchants into cohesive categories
        processed_merchants = set()  # Track merchants already assigned to categories
        category_counter = 1

        for merchant, count in frequent_merchants.items():
            # SKIP PROCESSED: Avoid duplicate processing of already-categorized merchants
            if merchant in processed_merchants:
                continue

            # INITIALIZE GROUP: Start with the current merchant
            similar_merchants = [merchant]

            # SMART GROUPING: Find similar merchants to group together
            if CATEGORIZATION_SETTINGS.get('enable_smart_grouping', True):
                similarity_threshold = CATEGORIZATION_SETTINGS.get('similarity_threshold', 0.85)
                
                for other_merchant in frequent_merchants.index:
                    # Check unprocessed merchants that aren't the current one
                    if other_merchant != merchant and other_merchant not in processed_merchants:
                        similarity_score = self._calculate_similarity(merchant, other_merchant)
                        
                        # GROUP SIMILAR MERCHANTS: Add if similarity exceeds threshold
                        if similarity_score > similarity_threshold:
                            similar_merchants.append(other_merchant)
                            processed_merchants.add(other_merchant)
                            self._log('debug', f"Grouped '{other_merchant}' with '{merchant}' (similarity: {similarity_score:.3f})")

            # MARK AS PROCESSED: Prevent reprocessing the primary merchant
            processed_merchants.add(merchant)

            # CATEGORY NAME GENERATION: Create meaningful category name from grouped merchants
            category_name = self._generate_category_name(merchant, similar_merchants)

            # TRANSACTION ANALYSIS: Analyze patterns for the grouped merchants
            category_transactions = df[df['clean_merchant'].isin(similar_merchants)]

            # CATEGORY CREATION: Build comprehensive category information
            categories[category_name] = {
                'merchants': similar_merchants,                          # All merchants in this category
                'transaction_count': len(category_transactions),         # Volume of transactions
                'avg_amount': decimal_to_float(decimal_mean(            # Average transaction amount
                    category_transactions['amount'])) if 'amount' in category_transactions and not category_transactions.empty else 0,
                'amount_pattern': self._determine_amount_pattern(category_transactions),  # Income/expense pattern
                'keywords': self._extract_common_keywords(similar_merchants),            # Key identifying terms
                'frequency_score': count / len(df),                     # Relative frequency in dataset
                'category_id': category_counter,                        # Unique identifier
                'confidence_score': self._calculate_category_confidence(similar_merchants, count)  # Quality metric
            }

            category_counter += 1

            # SCALABILITY LIMIT: Prevent excessive category generation
            max_categories = CATEGORIZATION_SETTINGS.get('max_auto_categories', 100)
            if len(categories) >= max_categories:
                self._log('info', f"Reached maximum category limit ({max_categories})")
                break

        self._log('info', f"Created {len(categories)} automatic categories from {len(frequent_merchants)} frequent merchants")
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

    def _calculate_category_confidence(self, merchants: List[str], transaction_count: int) -> float:
        """CONFIDENCE SCORING: Calculate confidence score for category quality"""
        # BASE CONFIDENCE: Start with transaction volume factor
        volume_score = min(1.0, transaction_count / 50)  # Scale up to 50 transactions
        
        # MERCHANT CONSISTENCY: Higher confidence for fewer, more consistent merchants
        merchant_consistency = 1.0 / len(merchants) if merchants else 0.0
        
        # NAME QUALITY: Check if merchant names are meaningful (not UNKNOWN)
        meaningful_merchants = [m for m in merchants if m != 'UNKNOWN' and len(m) > 2]
        name_quality = len(meaningful_merchants) / len(merchants) if merchants else 0.0
        
        # COMBINE FACTORS: Weight different aspects of category quality
        confidence = (volume_score * 0.4 + merchant_consistency * 0.3 + name_quality * 0.3)
        
        return round(min(1.0, confidence), 3)

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
