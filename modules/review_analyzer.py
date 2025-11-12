"""
Review Analyzer Module
Analyzes reviews for authenticity and detects fake reviews
"""

import re
from typing import Dict, List, Optional
from collections import Counter
import logging


class ReviewAnalyzer:
    """Analyzes review authenticity and detects suspicious patterns"""

    def __init__(self):
        """Initialize the review analyzer"""
        self.logger = logging.getLogger(__name__)

        # Patterns that indicate fake reviews
        self.fake_review_indicators = [
            r'(?:five|5)\s+stars?.*(?:highly recommend|best ever)',
            r'amazing.*perfect.*incredible.*wonderful',  # Too many superlatives
            r'this product changed my life',
            r'best.*ever.*my life',
            r'you\s+(?:will not|won\'t)\s+regret',
            r'trust me',
            r'(?:buy|purchase|order)\s+(?:now|today|immediately)',
        ]

        # Generic filler phrases often in fake reviews
        self.generic_phrases = [
            'highly recommend',
            'best ever',
            'changed my life',
            'five stars',
            'excellent service',
            'very professional',
            'amazing experience',
        ]

    def analyze_reviews(self, reviews: List[Dict]) -> Dict:
        """
        Analyze a set of reviews for authenticity

        Args:
            reviews: List of review dictionaries

        Returns:
            Dictionary with authenticity analysis
        """
        if not reviews:
            return {
                'authenticity_score': None,
                'suspicious': False,
                'reason': 'No reviews to analyze',
                'flags': []
            }

        analysis = {
            'authenticity_score': 1.0,  # Start with full authenticity
            'suspicious': False,
            'reason': None,
            'flags': [],
            'total_reviews': len(reviews),
            'suspicious_reviews': 0
        }

        # Check each review
        suspicious_count = 0
        for review in reviews:
            review_flags = self._check_single_review(review)
            if review_flags:
                suspicious_count += 1
                analysis['flags'].extend(review_flags)

        analysis['suspicious_reviews'] = suspicious_count

        # Calculate authenticity score
        if len(reviews) > 0:
            analysis['authenticity_score'] = 1.0 - (suspicious_count / len(reviews))

        # Check for suspicious patterns across all reviews
        collective_flags = self._check_collective_patterns(reviews)
        if collective_flags:
            analysis['flags'].extend(collective_flags)
            analysis['authenticity_score'] *= 0.7  # Reduce score

        # Mark as suspicious if score is low
        if analysis['authenticity_score'] < 0.5:
            analysis['suspicious'] = True
            analysis['reason'] = f"Low authenticity score: {analysis['authenticity_score']:.2%}"

        return analysis

    def _check_single_review(self, review: Dict) -> List[str]:
        """
        Check a single review for fake indicators

        Args:
            review: Review dictionary

        Returns:
            List of flags/issues found
        """
        flags = []
        review_text = str(review.get('text', '')).lower()

        if not review_text:
            return flags

        # Check 1: Fake review patterns
        for pattern in self.fake_review_indicators:
            if re.search(pattern, review_text, re.IGNORECASE):
                flags.append(f"Contains suspicious phrase pattern: {pattern}")

        # Check 2: Too many superlatives
        superlatives = [
            'amazing', 'incredible', 'awesome', 'fantastic',
            'perfect', 'best', 'excellent', 'outstanding',
            'wonderful', 'brilliant', 'spectacular'
        ]
        superlative_count = sum(
            1 for word in superlatives if word in review_text
        )
        if superlative_count > 3:
            flags.append(f"Excessive superlatives ({superlative_count} found)")

        # Check 3: Very short reviews with high ratings
        rating = review.get('rating', 0)
        word_count = len(review_text.split())
        if rating >= 4 and word_count < 10:
            flags.append("Very short review with high rating")

        # Check 4: All caps or excessive punctuation
        if review_text.isupper() and len(review_text) > 20:
            flags.append("Review is all caps")

        exclamation_count = review_text.count('!')
        if exclamation_count > 3:
            flags.append(f"Excessive exclamation marks ({exclamation_count})")

        # Check 5: Generic filler phrases
        generic_count = sum(
            1 for phrase in self.generic_phrases
            if phrase in review_text
        )
        if generic_count > 2:
            flags.append(f"Contains multiple generic phrases ({generic_count})")

        # Check 6: Suspicious reviewer name patterns
        reviewer = review.get('reviewer_name', '')
        if self._is_suspicious_reviewer_name(reviewer):
            flags.append(f"Suspicious reviewer name: {reviewer}")

        return flags

    def _check_collective_patterns(self, reviews: List[Dict]) -> List[str]:
        """
        Check for suspicious patterns across multiple reviews

        Args:
            reviews: List of review dictionaries

        Returns:
            List of collective flags
        """
        flags = []

        if len(reviews) < 3:
            return flags  # Not enough reviews to analyze patterns

        # Check 1: Too many reviews in short time period
        dates = [r.get('date') for r in reviews if r.get('date')]
        if len(dates) > 0:
            # This would require actual date parsing
            # Placeholder for date clustering analysis
            pass

        # Check 2: Similar review content (copy-paste)
        review_texts = [str(r.get('text', '')).lower() for r in reviews]
        if len(review_texts) >= 3:
            similarity_count = 0
            for i in range(len(review_texts)):
                for j in range(i + 1, len(review_texts)):
                    if self._text_similarity(review_texts[i], review_texts[j]) > 0.7:
                        similarity_count += 1

            if similarity_count > len(reviews) * 0.3:
                flags.append("Many reviews have very similar content")

        # Check 3: Unnatural rating distribution
        ratings = [r.get('rating', 0) for r in reviews if r.get('rating')]
        if len(ratings) >= 5:
            rating_dist = Counter(ratings)
            # Check if too many 5-star reviews
            five_star_ratio = rating_dist.get(5, 0) / len(ratings)
            if five_star_ratio > 0.8:
                flags.append(f"Suspiciously high ratio of 5-star reviews ({five_star_ratio:.0%})")

            # Check if distribution is too uniform
            if len(rating_dist) == 1:
                flags.append("All reviews have identical ratings")

        # Check 4: Suspicious reviewer names pattern
        reviewer_names = [r.get('reviewer_name', '') for r in reviews if r.get('reviewer_name')]
        if len(reviewer_names) >= 5:
            suspicious_names = sum(
                1 for name in reviewer_names
                if self._is_suspicious_reviewer_name(name)
            )
            if suspicious_names / len(reviewer_names) > 0.5:
                flags.append("Many reviews from suspicious accounts")

        return flags

    def _is_suspicious_reviewer_name(self, name: str) -> bool:
        """
        Check if reviewer name appears suspicious

        Args:
            name: Reviewer name

        Returns:
            True if suspicious
        """
        if not name:
            return True

        name_lower = name.lower()

        # Check for generic names
        generic_names = [
            'user', 'customer', 'guest', 'anonymous',
            'reviewer', 'verified buyer'
        ]
        if any(generic in name_lower for generic in generic_names):
            return True

        # Check for random-looking names (e.g., "user12345")
        if re.match(r'^[a-z]+\d{3,}$', name_lower):
            return True

        # Check for very short names
        if len(name) < 3:
            return True

        return False

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple similarity between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0

        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def generate_review_summary(self, reviews: List[Dict]) -> str:
        """
        Generate a summary of reviews

        Args:
            reviews: List of review dictionaries

        Returns:
            Summary string
        """
        if not reviews:
            return "No reviews available"

        total = len(reviews)
        ratings = [r.get('rating', 0) for r in reviews if r.get('rating')]

        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            rating_summary = f"Average: {avg_rating:.1f}/5.0 from {total} reviews"
        else:
            rating_summary = f"{total} reviews (no ratings available)"

        return rating_summary
