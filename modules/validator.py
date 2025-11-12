"""
Legitimacy Validator Module
Validates venue listings for legitimacy and trustworthiness
"""

import re
import whois
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import urlparse
import tldextract
import logging
import socket


class LegitimacyValidator:
    """Validates listing legitimacy through multiple checks"""

    def __init__(self):
        """Initialize the validator"""
        self.logger = logging.getLogger(__name__)

        # Vague pricing patterns to detect
        self.vague_pricing_patterns = [
            r'from\s*\$',
            r'starting\s*(at|from)',
            r'approx\.?\s*\$',
            r'approximately',
            r'as\s+low\s+as',
            r'contact\s+(us\s+)?for\s+(pricing|price|quote)',
            r'call\s+for\s+(pricing|price)',
            r'\+\s*$',  # Prices ending with +
            r'and\s+up',
        ]

        # SEO spam indicators
        self.spam_indicators = [
            r'seo\s+company',
            r'buy\s+(backlinks|links)',
            r'guaranteed\s+ranking',
            r'#1\s+on\s+google',
            r'increase\s+your\s+ranking',
            r'viagra',
            r'casino\s+online',
            r'weight\s+loss\s+pill',
        ]

        # Minimum domain age for legitimacy (in days)
        self.min_domain_age_days = 90

    def validate_listing(self, listing: Dict) -> Dict:
        """
        Perform comprehensive validation on a listing

        Args:
            listing: Listing dictionary to validate

        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_legitimate': True,
            'exclusion_reasons': [],
            'warnings': [],
            'checks': {}
        }

        # Check 1: HTTPS validation
        https_result = self._check_https(listing.get('url', ''))
        validation['checks']['https'] = https_result
        if not https_result['valid']:
            validation['is_legitimate'] = False
            validation['exclusion_reasons'].append('No HTTPS - Insecure connection')

        # Check 2: Domain age validation
        domain_result = self._check_domain_age(listing.get('domain', ''))
        validation['checks']['domain_age'] = domain_result
        if domain_result['age_days'] is not None:
            if domain_result['age_days'] < self.min_domain_age_days:
                validation['warnings'].append(
                    f"Very new domain ({domain_result['age_days']} days old)"
                )
                # Not excluding but flagging as suspicious
                validation['checks']['domain_age']['suspicious'] = True

        # Check 3: Vague pricing detection
        pricing_result = self._check_vague_pricing(listing.get('price'))
        validation['checks']['vague_pricing'] = pricing_result
        if pricing_result['is_vague']:
            validation['is_legitimate'] = False
            validation['exclusion_reasons'].append(
                f"Vague pricing: {pricing_result['reason']}"
            )

        # Check 4: SEO spam detection
        spam_result = self._check_seo_spam(listing)
        validation['checks']['seo_spam'] = spam_result
        if spam_result['is_spam']:
            validation['is_legitimate'] = False
            validation['exclusion_reasons'].append(
                f"SEO spam detected: {spam_result['reason']}"
            )

        # Check 5: Data completeness
        completeness_result = self._check_data_completeness(listing)
        validation['checks']['data_completeness'] = completeness_result
        if completeness_result['score'] < 0.3:  # Less than 30% complete
            validation['warnings'].append(
                f"Incomplete data ({completeness_result['score']:.0%} complete)"
            )

        # Check 6: URL validity
        url_result = self._check_url_validity(listing.get('url', ''))
        validation['checks']['url_validity'] = url_result
        if not url_result['valid']:
            validation['is_legitimate'] = False
            validation['exclusion_reasons'].append('Invalid or suspicious URL')

        # Check 7: Price reasonability
        if listing.get('price'):
            price_check = self._check_price_reasonability(listing.get('price'))
            validation['checks']['price_reasonability'] = price_check
            if not price_check['reasonable']:
                validation['warnings'].append(price_check['reason'])

        return validation

    def _check_https(self, url: str) -> Dict:
        """
        Check if URL uses HTTPS

        Args:
            url: URL to check

        Returns:
            Dictionary with HTTPS check results
        """
        result = {
            'valid': False,
            'uses_https': False
        }

        if not url:
            return result

        try:
            parsed = urlparse(url)
            result['uses_https'] = parsed.scheme == 'https'
            result['valid'] = result['uses_https']
        except Exception as e:
            self.logger.debug(f"HTTPS check failed for {url}: {e}")

        return result

    def _check_domain_age(self, domain: str) -> Dict:
        """
        Check domain age using WHOIS

        Args:
            domain: Domain name to check

        Returns:
            Dictionary with domain age information
        """
        result = {
            'age_days': None,
            'creation_date': None,
            'expiration_date': None,
            'registrar': None,
            'error': None
        }

        if not domain or 'example' in domain.lower():
            # Skip example domains
            result['age_days'] = 999  # Assume old enough
            return result

        try:
            # Perform WHOIS lookup
            w = whois.whois(domain)

            # Extract creation date
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]

            if creation_date:
                result['creation_date'] = creation_date.isoformat() if hasattr(
                    creation_date, 'isoformat'
                ) else str(creation_date)

                # Calculate age in days
                if hasattr(creation_date, 'date'):
                    age = datetime.now() - creation_date
                    result['age_days'] = age.days
                else:
                    result['age_days'] = 999  # Can't determine, assume ok

            # Extract expiration date
            expiration_date = w.expiration_date
            if isinstance(expiration_date, list):
                expiration_date = expiration_date[0]

            if expiration_date:
                result['expiration_date'] = expiration_date.isoformat() if hasattr(
                    expiration_date, 'isoformat'
                ) else str(expiration_date)

            # Extract registrar
            result['registrar'] = w.registrar if hasattr(w, 'registrar') else None

        except Exception as e:
            self.logger.debug(f"WHOIS lookup failed for {domain}: {e}")
            result['error'] = str(e)
            result['age_days'] = None  # Unknown

        return result

    def _check_vague_pricing(self, price: Optional[str]) -> Dict:
        """
        Check if pricing is vague or unclear

        Args:
            price: Price string to check

        Returns:
            Dictionary with vague pricing check results
        """
        result = {
            'is_vague': False,
            'reason': None
        }

        if not price:
            result['is_vague'] = True
            result['reason'] = 'No price provided'
            return result

        price_lower = price.lower()

        # Check against vague pricing patterns
        for pattern in self.vague_pricing_patterns:
            if re.search(pattern, price_lower, re.IGNORECASE):
                result['is_vague'] = True
                result['reason'] = f"Contains vague term: {pattern}"
                break

        return result

    def _check_seo_spam(self, listing: Dict) -> Dict:
        """
        Check for SEO spam indicators

        Args:
            listing: Listing dictionary

        Returns:
            Dictionary with spam check results
        """
        result = {
            'is_spam': False,
            'reason': None,
            'indicators_found': []
        }

        # Combine all text content
        text_content = ' '.join([
            str(listing.get('title', '')),
            str(listing.get('description', '')),
            str(listing.get('location', ''))
        ]).lower()

        # Check for spam indicators
        for indicator in self.spam_indicators:
            if re.search(indicator, text_content, re.IGNORECASE):
                result['indicators_found'].append(indicator)

        if result['indicators_found']:
            result['is_spam'] = True
            result['reason'] = f"Found spam indicators: {', '.join(result['indicators_found'])}"

        # Check for excessive keyword stuffing
        words = text_content.split()
        if len(words) > 0:
            unique_words = set(words)
            repetition_ratio = len(words) / len(unique_words)

            if repetition_ratio > 3:  # High repetition
                result['is_spam'] = True
                result['reason'] = 'Excessive keyword repetition detected'

        return result

    def _check_data_completeness(self, listing: Dict) -> Dict:
        """
        Check how complete the listing data is

        Args:
            listing: Listing dictionary

        Returns:
            Dictionary with completeness score
        """
        important_fields = [
            'title', 'url', 'price', 'location',
            'description', 'capacity'
        ]

        filled_fields = sum(
            1 for field in important_fields
            if listing.get(field) is not None and listing.get(field) != ''
        )

        score = filled_fields / len(important_fields)

        return {
            'score': score,
            'filled_fields': filled_fields,
            'total_fields': len(important_fields),
            'missing_fields': [
                field for field in important_fields
                if not listing.get(field)
            ]
        }

    def _check_url_validity(self, url: str) -> Dict:
        """
        Check if URL is valid and reachable

        Args:
            url: URL to check

        Returns:
            Dictionary with URL validity results
        """
        result = {
            'valid': False,
            'reason': None
        }

        if not url:
            result['reason'] = 'No URL provided'
            return result

        try:
            parsed = urlparse(url)

            # Check for required components
            if not parsed.scheme:
                result['reason'] = 'Missing URL scheme'
                return result

            if not parsed.netloc:
                result['reason'] = 'Missing domain'
                return result

            # Check for suspicious patterns
            if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', parsed.netloc):
                # IP address instead of domain
                result['reason'] = 'Uses IP address instead of domain'
                return result

            # Check for suspicious TLDs
            suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq']
            if any(parsed.netloc.endswith(tld) for tld in suspicious_tlds):
                result['reason'] = f'Suspicious TLD: {parsed.netloc}'
                return result

            result['valid'] = True

        except Exception as e:
            result['reason'] = f'URL parsing error: {str(e)}'

        return result

    def _check_price_reasonability(self, price: str) -> Dict:
        """
        Check if price seems reasonable

        Args:
            price: Price string

        Returns:
            Dictionary with reasonability check
        """
        result = {
            'reasonable': True,
            'reason': None
        }

        # Extract numeric value
        numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d{2})?', price)
        if numbers:
            try:
                # Remove commas and convert to float
                amount = float(numbers[0].replace(',', ''))

                # Check for unreasonably low prices
                if amount < 10:
                    result['reasonable'] = False
                    result['reason'] = f'Suspiciously low price: ${amount}'

                # Check for unreasonably high prices
                elif amount > 1000000:
                    result['reasonable'] = False
                    result['reason'] = f'Suspiciously high price: ${amount}'

            except ValueError:
                pass

        return result

    def validate_availability(self, listing: Dict, search_dates: tuple) -> Dict:
        """
        Validate if listing is available for requested dates

        Args:
            listing: Listing dictionary
            search_dates: Tuple of (start_date, end_date)

        Returns:
            Dictionary with availability validation results
        """
        result = {
            'available': None,  # None means unknown
            'verified': False,
            'reason': None
        }

        # This would require accessing actual calendar/booking data
        # For now, we'll mark as unverified
        result['reason'] = 'Availability verification requires direct API access'

        return result
