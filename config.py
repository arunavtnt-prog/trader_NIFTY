"""
Configuration file for Venue Listing Validator
"""

# Scraping configuration
SCRAPING_CONFIG = {
    'timeout': 30,  # Request timeout in seconds
    'max_retries': 3,  # Maximum retry attempts
    'delay_between_requests': 1,  # Delay in seconds to avoid rate limiting
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Validation configuration
VALIDATION_CONFIG = {
    'min_domain_age_days': 90,  # Minimum domain age for legitimacy
    'require_https': True,  # Require HTTPS for valid listings
    'min_data_completeness': 0.3,  # Minimum data completeness score (0-1)
}

# Ranking weights (must sum to 1.0)
RANKING_WEIGHTS = {
    'legitimacy': 0.30,
    'data_completeness': 0.20,
    'review_authenticity': 0.20,
    'price_transparency': 0.15,
    'domain_trust': 0.15
}

# Export configuration
EXPORT_CONFIG = {
    'excel_engine': 'openpyxl',
    'include_metadata': True,
    'google_credentials_file': 'google_credentials.json'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'log_directory': 'logs',
    'console_level': 'WARNING'  # Only show warnings and above in console
}

# Common venue listing websites (add more as needed)
VENUE_SOURCES = [
    {
        'name': 'EventUp',
        'base_url': 'https://www.eventup.com',
        'search_pattern': '/venues/search'
    },
    {
        'name': 'Peerspace',
        'base_url': 'https://www.peerspace.com',
        'search_pattern': '/venues'
    },
    {
        'name': 'VenueScanner',
        'base_url': 'https://www.venuescanner.com',
        'search_pattern': '/search'
    }
    # Add more venue sources here
]

# Vague pricing patterns to detect and flag
VAGUE_PRICING_PATTERNS = [
    r'from\s*\$',
    r'starting\s*(at|from)',
    r'approx\.?\s*\$',
    r'approximately',
    r'as\s+low\s+as',
    r'contact\s+(us\s+)?for\s+(pricing|price|quote)',
    r'call\s+for\s+(pricing|price)',
    r'\+\s*$',
    r'and\s+up',
]

# SEO spam indicators
SEO_SPAM_INDICATORS = [
    r'seo\s+company',
    r'buy\s+(backlinks|links)',
    r'guaranteed\s+ranking',
    r'#1\s+on\s+google',
    r'increase\s+your\s+ranking',
    r'viagra',
    r'casino\s+online',
    r'weight\s+loss\s+pill',
]

# Suspicious TLDs to flag
SUSPICIOUS_TLDS = [
    '.tk', '.ml', '.ga', '.cf', '.gq'
]
