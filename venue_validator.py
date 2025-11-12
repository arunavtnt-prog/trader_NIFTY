#!/usr/bin/env python3
"""
Venue Listing Validator - Main Application
A comprehensive tool for scraping, validating, and ranking venue listings
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Optional
import json

from modules.cli_interface import CLIInterface
from modules.scraper import VenueScraper
from modules.validator import LegitimacyValidator
from modules.review_analyzer import ReviewAnalyzer
from modules.ranking_engine import RankingEngine
from modules.export_manager import ExportManager
from modules.logger import setup_logger


class VenueListingValidator:
    """Main application class for venue listing validation"""

    def __init__(self):
        """Initialize the application with all necessary modules"""
        self.logger = setup_logger()
        self.cli = CLIInterface()
        self.scraper = VenueScraper()
        self.validator = LegitimacyValidator()
        self.review_analyzer = ReviewAnalyzer()
        self.ranking_engine = RankingEngine()
        self.export_manager = ExportManager()

        self.search_criteria = {}
        self.raw_listings = []
        self.validated_listings = []
        self.excluded_listings = []

    def run(self):
        """Main execution flow"""
        try:
            self.logger.info("=== Venue Listing Validator Started ===")

            # Step 1: Collect search criteria
            self.collect_search_criteria()

            # Step 2: Scrape listings
            self.scrape_listings()

            # Step 3: Validate and filter listings
            self.validate_listings()

            # Step 4: Rank listings
            self.rank_listings()

            # Step 5: Remove duplicates
            self.remove_duplicates()

            # Step 6: Export results
            self.export_results()

            # Step 7: Generate summary report
            self.generate_summary()

            self.logger.info("=== Venue Listing Validator Completed Successfully ===")

        except KeyboardInterrupt:
            self.logger.warning("\n\nProcess interrupted by user. Exiting...")
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"Fatal error: {str(e)}", exc_info=True)
            sys.exit(1)

    def collect_search_criteria(self):
        """Collect search parameters from user via interactive CLI"""
        self.logger.info("Collecting search criteria from user...")
        print("\n" + "="*70)
        print("VENUE LISTING VALIDATOR - Search Criteria Collection")
        print("="*70 + "\n")

        self.search_criteria = self.cli.collect_search_criteria()

        print("\n" + "-"*70)
        print("Search Criteria Summary:")
        print("-"*70)
        for key, value in self.search_criteria.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print("-"*70 + "\n")

        # Confirm before proceeding
        confirm = input("Proceed with these criteria? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            self.logger.info("User cancelled. Restarting criteria collection...")
            self.collect_search_criteria()

    def scrape_listings(self):
        """Scrape venues from multiple sources"""
        self.logger.info("Starting web scraping process...")
        print("\n" + "="*70)
        print("SCRAPING VENUE LISTINGS")
        print("="*70 + "\n")

        target_count = 50
        self.raw_listings = self.scraper.scrape_multiple_sources(
            self.search_criteria,
            target_count=target_count
        )

        print(f"\nTotal listings scraped: {len(self.raw_listings)}")
        self.logger.info(f"Scraped {len(self.raw_listings)} listings")

    def validate_listings(self):
        """Validate each listing for legitimacy"""
        self.logger.info("Validating listings...")
        print("\n" + "="*70)
        print("VALIDATING LISTING LEGITIMACY")
        print("="*70 + "\n")

        for idx, listing in enumerate(self.raw_listings, 1):
            print(f"Validating listing {idx}/{len(self.raw_listings)}...", end='\r')

            # Perform validation checks
            validation_result = self.validator.validate_listing(listing)

            # Analyze reviews if present
            if listing.get('reviews'):
                review_analysis = self.review_analyzer.analyze_reviews(
                    listing['reviews']
                )
                validation_result['review_authenticity'] = review_analysis

            # Add validation results to listing
            listing['validation'] = validation_result

            # Categorize as valid or excluded
            if validation_result['is_legitimate']:
                self.validated_listings.append(listing)
            else:
                self.excluded_listings.append(listing)

        print(f"\nValid listings: {len(self.validated_listings)}")
        print(f"Excluded listings: {len(self.excluded_listings)}")
        self.logger.info(
            f"Validation complete: {len(self.validated_listings)} valid, "
            f"{len(self.excluded_listings)} excluded"
        )

    def rank_listings(self):
        """Rank validated listings by quality and relevance"""
        self.logger.info("Ranking listings...")
        print("\n" + "="*70)
        print("RANKING LISTINGS")
        print("="*70 + "\n")

        self.validated_listings = self.ranking_engine.rank_listings(
            self.validated_listings,
            self.search_criteria
        )

        print(f"Ranked {len(self.validated_listings)} listings")
        self.logger.info(f"Ranked {len(self.validated_listings)} listings")

    def remove_duplicates(self):
        """Remove duplicate listings"""
        self.logger.info("Removing duplicates...")
        print("\n" + "="*70)
        print("REMOVING DUPLICATES")
        print("="*70 + "\n")

        original_count = len(self.validated_listings)
        self.validated_listings = self.ranking_engine.remove_duplicates(
            self.validated_listings
        )

        duplicates_removed = original_count - len(self.validated_listings)
        print(f"Duplicates removed: {duplicates_removed}")
        print(f"Final listing count: {len(self.validated_listings)}")
        self.logger.info(
            f"Removed {duplicates_removed} duplicates. "
            f"Final count: {len(self.validated_listings)}"
        )

    def export_results(self):
        """Export results to Excel and optionally Google Sheets"""
        self.logger.info("Exporting results...")
        print("\n" + "="*70)
        print("EXPORTING RESULTS")
        print("="*70 + "\n")

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export to Excel
        excel_file = f"venue_listings_{timestamp}.xlsx"
        self.export_manager.export_to_excel(
            self.validated_listings,
            self.excluded_listings,
            excel_file
        )
        print(f"✓ Excel file created: {excel_file}")

        # Ask about Google Sheets export
        export_to_sheets = input("\nExport to Google Sheets? (yes/no): ").strip().lower()
        if export_to_sheets in ['yes', 'y']:
            try:
                sheet_url = self.export_manager.export_to_google_sheets(
                    self.validated_listings,
                    self.excluded_listings,
                    f"Venue Listings {timestamp}"
                )
                print(f"✓ Google Sheet created: {sheet_url}")
            except Exception as e:
                self.logger.error(f"Failed to export to Google Sheets: {e}")
                print(f"✗ Google Sheets export failed: {e}")

    def generate_summary(self):
        """Generate and display summary report"""
        self.logger.info("Generating summary report...")
        print("\n" + "="*70)
        print("SUMMARY REPORT")
        print("="*70 + "\n")

        print(f"Total Listings Scraped: {len(self.raw_listings)}")
        print(f"Valid Listings: {len(self.validated_listings)}")
        print(f"Excluded Listings: {len(self.excluded_listings)}")

        if self.excluded_listings:
            print("\n" + "-"*70)
            print("EXCLUSION REASONS:")
            print("-"*70)

            exclusion_reasons = {}
            for listing in self.excluded_listings:
                reasons = listing['validation'].get('exclusion_reasons', ['Unknown'])
                for reason in reasons:
                    exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1

            for reason, count in sorted(exclusion_reasons.items(),
                                       key=lambda x: x[1], reverse=True):
                print(f"  • {reason}: {count} listings")

        # Save detailed summary to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"summary_report_{timestamp}.txt"

        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("VENUE LISTING VALIDATOR - DETAILED SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("SEARCH CRITERIA:\n")
            f.write("-"*70 + "\n")
            for key, value in self.search_criteria.items():
                f.write(f"  {key.replace('_', ' ').title()}: {value}\n")

            f.write("\n" + "="*70 + "\n")
            f.write("RESULTS SUMMARY\n")
            f.write("="*70 + "\n")
            f.write(f"Total Listings Scraped: {len(self.raw_listings)}\n")
            f.write(f"Valid Listings: {len(self.validated_listings)}\n")
            f.write(f"Excluded Listings: {len(self.excluded_listings)}\n\n")

            if self.excluded_listings:
                f.write("EXCLUDED LISTINGS DETAILS:\n")
                f.write("-"*70 + "\n")
                for idx, listing in enumerate(self.excluded_listings, 1):
                    f.write(f"\n{idx}. {listing.get('title', 'Untitled')}\n")
                    f.write(f"   URL: {listing.get('url', 'N/A')}\n")
                    f.write(f"   Domain: {listing.get('domain', 'N/A')}\n")
                    f.write(f"   Reasons: {', '.join(listing['validation'].get('exclusion_reasons', ['Unknown']))}\n")

        print(f"\n✓ Detailed summary saved to: {summary_file}")
        self.logger.info(f"Summary report saved to {summary_file}")


def main():
    """Entry point for the application"""
    app = VenueListingValidator()
    app.run()


if __name__ == "__main__":
    main()
