#!/usr/bin/env python3
"""
Example usage script for Venue Listing Validator
This demonstrates how to use the application programmatically
"""

from modules.cli_interface import CLIInterface
from modules.scraper import VenueScraper
from modules.validator import LegitimacyValidator
from modules.review_analyzer import ReviewAnalyzer
from modules.ranking_engine import RankingEngine
from modules.export_manager import ExportManager


def example_programmatic_usage():
    """Example of using the modules programmatically"""

    print("=== Venue Listing Validator - Programmatic Example ===\n")

    # Example search criteria (instead of interactive input)
    search_criteria = {
        'location': 'New York City',
        'venue_type': 'Wedding venue',
        'price_min': 5000,
        'price_max': 15000,
        'date_from': '2025-06-01',
        'date_to': '2025-09-30',
        'capacity': 150,
        'amenities': ['Parking', 'Catering'],
        'custom_urls': [],
        'additional_criteria': 'Outdoor space preferred'
    }

    print("Search Criteria:")
    for key, value in search_criteria.items():
        print(f"  {key}: {value}")
    print()

    # Initialize modules
    scraper = VenueScraper()
    validator = LegitimacyValidator()
    review_analyzer = ReviewAnalyzer()
    ranking_engine = RankingEngine()
    export_manager = ExportManager()

    # Example: Create a sample listing for validation
    sample_listing = {
        'title': 'Grand Ballroom Event Space',
        'url': 'https://example-venue.com/grand-ballroom',
        'domain': 'example-venue.com',
        'price': '$8,500 per event',
        'location': 'Manhattan, New York',
        'description': 'Beautiful event space perfect for weddings and corporate events.',
        'capacity': 200,
        'rating': 4.8,
        'review_count': 45,
        'reviews': [],
        'scraped_at': '2025-01-01',
        'source_type': 'sample'
    }

    print("Sample Listing:")
    print(f"  Title: {sample_listing['title']}")
    print(f"  URL: {sample_listing['url']}")
    print(f"  Price: {sample_listing['price']}")
    print()

    # Validate the listing
    print("Validating listing...")
    validation_result = validator.validate_listing(sample_listing)

    print(f"  Is Legitimate: {validation_result['is_legitimate']}")
    print(f"  HTTPS: {validation_result['checks']['https']['uses_https']}")
    print(f"  Vague Pricing: {validation_result['checks']['vague_pricing']['is_vague']}")
    print(f"  Data Completeness: {validation_result['checks']['data_completeness']['score']:.0%}")
    print()

    # Add validation results
    sample_listing['validation'] = validation_result

    # Rank the listing
    listings = [sample_listing]
    ranked_listings = ranking_engine.rank_listings(listings, search_criteria)

    print(f"Ranking Score: {ranked_listings[0]['ranking_score']:.2f}/100")
    print()

    print("✓ Example completed successfully!")
    print("\nTo run the full interactive application, use:")
    print("  python venue_validator.py")


if __name__ == "__main__":
    example_programmatic_usage()
