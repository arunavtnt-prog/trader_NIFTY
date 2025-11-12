"""
Ranking Engine Module
Ranks and filters venue listings based on quality and relevance
"""

from typing import Dict, List
import logging
from difflib import SequenceMatcher
from urllib.parse import urlparse


class RankingEngine:
    """Ranks listings based on multiple criteria"""

    def __init__(self):
        """Initialize the ranking engine"""
        self.logger = logging.getLogger(__name__)

        # Weights for different ranking factors
        self.weights = {
            'legitimacy': 0.30,
            'data_completeness': 0.20,
            'review_authenticity': 0.20,
            'price_transparency': 0.15,
            'domain_trust': 0.15
        }

    def rank_listings(self, listings: List[Dict], criteria: Dict) -> List[Dict]:
        """
        Rank listings by quality and relevance

        Args:
            listings: List of listing dictionaries
            criteria: Search criteria for relevance scoring

        Returns:
            Sorted list of listings with ranking scores
        """
        self.logger.info(f"Ranking {len(listings)} listings...")

        # Calculate scores for each listing
        for listing in listings:
            score = self._calculate_ranking_score(listing, criteria)
            listing['ranking_score'] = score

        # Sort by score (highest first)
        ranked_listings = sorted(
            listings,
            key=lambda x: x.get('ranking_score', 0),
            reverse=True
        )

        # Add rank position
        for idx, listing in enumerate(ranked_listings, 1):
            listing['rank'] = idx

        return ranked_listings

    def _calculate_ranking_score(self, listing: Dict, criteria: Dict) -> float:
        """
        Calculate overall ranking score for a listing

        Args:
            listing: Listing dictionary
            criteria: Search criteria

        Returns:
            Ranking score (0-100)
        """
        scores = {}

        # Factor 1: Legitimacy score
        validation = listing.get('validation', {})
        legitimacy_score = self._calculate_legitimacy_score(validation)
        scores['legitimacy'] = legitimacy_score

        # Factor 2: Data completeness score
        completeness = validation.get('checks', {}).get('data_completeness', {})
        completeness_score = completeness.get('score', 0.5) * 100
        scores['data_completeness'] = completeness_score

        # Factor 3: Review authenticity score
        review_auth = validation.get('review_authenticity', {})
        review_score = review_auth.get('authenticity_score', 0.5) * 100 if review_auth else 50
        scores['review_authenticity'] = review_score

        # Factor 4: Price transparency score
        pricing = validation.get('checks', {}).get('vague_pricing', {})
        price_score = 0 if pricing.get('is_vague') else 100
        scores['price_transparency'] = price_score

        # Factor 5: Domain trust score
        domain_age = validation.get('checks', {}).get('domain_age', {})
        domain_score = self._calculate_domain_trust_score(domain_age)
        scores['domain_trust'] = domain_score

        # Calculate weighted average
        total_score = sum(
            scores[factor] * self.weights[factor]
            for factor in self.weights.keys()
        )

        # Bonus points for matching criteria
        relevance_bonus = self._calculate_relevance_bonus(listing, criteria)
        total_score += relevance_bonus

        # Ensure score is between 0 and 100
        total_score = max(0, min(100, total_score))

        return round(total_score, 2)

    def _calculate_legitimacy_score(self, validation: Dict) -> float:
        """
        Calculate legitimacy score from validation results

        Args:
            validation: Validation dictionary

        Returns:
            Score from 0 to 100
        """
        if validation.get('is_legitimate'):
            base_score = 100
        else:
            base_score = 0

        # Deduct for warnings
        warnings = validation.get('warnings', [])
        warning_penalty = len(warnings) * 10

        score = base_score - warning_penalty
        return max(0, score)

    def _calculate_domain_trust_score(self, domain_info: Dict) -> float:
        """
        Calculate domain trust score

        Args:
            domain_info: Domain age information

        Returns:
            Score from 0 to 100
        """
        age_days = domain_info.get('age_days')

        if age_days is None:
            return 50  # Unknown, neutral score

        # Score based on age
        if age_days >= 365:  # 1+ years
            return 100
        elif age_days >= 180:  # 6+ months
            return 80
        elif age_days >= 90:  # 3+ months
            return 60
        elif age_days >= 30:  # 1+ month
            return 40
        else:  # Very new
            return 20

    def _calculate_relevance_bonus(self, listing: Dict, criteria: Dict) -> float:
        """
        Calculate bonus points for matching search criteria

        Args:
            listing: Listing dictionary
            criteria: Search criteria

        Returns:
            Bonus points (0-10)
        """
        bonus = 0.0

        # Location match
        search_location = str(criteria.get('location', '')).lower()
        listing_location = str(listing.get('location', '')).lower()
        if search_location and search_location != 'any location':
            if search_location in listing_location:
                bonus += 3.0

        # Venue type match
        search_type = str(criteria.get('venue_type', '')).lower()
        listing_title = str(listing.get('title', '')).lower()
        listing_desc = str(listing.get('description', '')).lower()
        if search_type and search_type != 'any venue type':
            if search_type in listing_title or search_type in listing_desc:
                bonus += 3.0

        # Capacity match
        search_capacity = criteria.get('capacity')
        listing_capacity = listing.get('capacity')
        if search_capacity and listing_capacity:
            if listing_capacity >= search_capacity:
                bonus += 2.0

        # Price range match
        if listing.get('price'):
            price_match = self._check_price_in_range(
                listing.get('price'),
                criteria.get('price_min'),
                criteria.get('price_max')
            )
            if price_match:
                bonus += 2.0

        return bonus

    def _check_price_in_range(self, price_str: str,
                              min_price: float = None,
                              max_price: float = None) -> bool:
        """
        Check if price falls within specified range

        Args:
            price_str: Price string
            min_price: Minimum price
            max_price: Maximum price

        Returns:
            True if in range, False otherwise
        """
        if not min_price and not max_price:
            return True

        # Extract numeric value from price string
        import re
        numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d{2})?', price_str)
        if not numbers:
            return False

        try:
            price = float(numbers[0].replace(',', ''))

            if min_price and price < min_price:
                return False
            if max_price and price > max_price:
                return False

            return True

        except ValueError:
            return False

    def remove_duplicates(self, listings: List[Dict]) -> List[Dict]:
        """
        Remove duplicate listings

        Args:
            listings: List of listing dictionaries

        Returns:
            List with duplicates removed
        """
        self.logger.info("Removing duplicates...")

        if not listings:
            return listings

        unique_listings = []
        seen_urls = set()
        seen_signatures = set()

        for listing in listings:
            # Check 1: Exact URL match
            url = listing.get('url', '')
            normalized_url = self._normalize_url(url)

            if normalized_url in seen_urls:
                self.logger.debug(f"Duplicate URL found: {url}")
                continue

            # Check 2: Similar content signature
            signature = self._create_listing_signature(listing)
            if self._is_duplicate_signature(signature, seen_signatures):
                self.logger.debug(f"Duplicate content found: {listing.get('title')}")
                continue

            # Not a duplicate, add to unique list
            seen_urls.add(normalized_url)
            seen_signatures.add(signature)
            unique_listings.append(listing)

        self.logger.info(
            f"Removed {len(listings) - len(unique_listings)} duplicates"
        )

        return unique_listings

    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL for comparison

        Args:
            url: URL to normalize

        Returns:
            Normalized URL
        """
        if not url:
            return ''

        # Parse and reconstruct without query parameters
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        # Remove trailing slash
        normalized = normalized.rstrip('/')

        # Convert to lowercase
        normalized = normalized.lower()

        return normalized

    def _create_listing_signature(self, listing: Dict) -> str:
        """
        Create a signature for duplicate detection

        Args:
            listing: Listing dictionary

        Returns:
            Signature string
        """
        # Combine key fields
        title = str(listing.get('title', '')).lower().strip()
        location = str(listing.get('location', '')).lower().strip()
        price = str(listing.get('price', '')).lower().strip()

        # Create signature
        signature = f"{title}|{location}|{price}"

        return signature

    def _is_duplicate_signature(self, signature: str,
                               seen_signatures: set,
                               threshold: float = 0.85) -> bool:
        """
        Check if signature is similar to any seen signatures

        Args:
            signature: Signature to check
            seen_signatures: Set of previously seen signatures
            threshold: Similarity threshold

        Returns:
            True if duplicate found
        """
        for seen in seen_signatures:
            similarity = SequenceMatcher(None, signature, seen).ratio()
            if similarity >= threshold:
                return True

        return False

    def filter_by_criteria(self, listings: List[Dict], criteria: Dict) -> List[Dict]:
        """
        Filter listings based on search criteria

        Args:
            listings: List of listings
            criteria: Search criteria

        Returns:
            Filtered list of listings
        """
        filtered = []

        for listing in listings:
            if self._matches_criteria(listing, criteria):
                filtered.append(listing)

        self.logger.info(
            f"Filtered to {len(filtered)} listings from {len(listings)}"
        )

        return filtered

    def _matches_criteria(self, listing: Dict, criteria: Dict) -> bool:
        """
        Check if listing matches search criteria

        Args:
            listing: Listing dictionary
            criteria: Search criteria

        Returns:
            True if matches
        """
        # Location filter
        search_location = str(criteria.get('location', '')).lower()
        if search_location and search_location != 'any location':
            listing_location = str(listing.get('location', '')).lower()
            if search_location not in listing_location:
                return False

        # Price filter
        if listing.get('price'):
            if not self._check_price_in_range(
                listing.get('price'),
                criteria.get('price_min'),
                criteria.get('price_max')
            ):
                return False

        # Capacity filter
        search_capacity = criteria.get('capacity')
        listing_capacity = listing.get('capacity')
        if search_capacity and listing_capacity:
            if listing_capacity < search_capacity:
                return False

        return True
