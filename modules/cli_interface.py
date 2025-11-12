"""
CLI Interface Module
Handles interactive user input collection
"""

from datetime import datetime
from typing import Dict, List, Optional


class CLIInterface:
    """Interactive command-line interface for collecting search criteria"""

    def __init__(self):
        """Initialize CLI interface"""
        pass

    def collect_search_criteria(self) -> Dict:
        """
        Collect search criteria from user through interactive prompts

        Returns:
            Dictionary containing all search parameters
        """
        criteria = {}

        print("Please provide your venue search criteria:\n")

        # Location
        criteria['location'] = self._get_input(
            "What location or venue area do you want?",
            default="Any location",
            help_text="e.g., New York, Manhattan, Downtown LA"
        )

        # Venue type
        criteria['venue_type'] = self._get_input(
            "What type of venue or listing are you looking for?",
            default="Any venue type",
            help_text="e.g., Wedding venue, Conference hall, Restaurant, Event space"
        )

        # Price range
        criteria['price_min'] = self._get_number_input(
            "Minimum price (leave empty for no minimum):",
            required=False
        )
        criteria['price_max'] = self._get_number_input(
            "Maximum price (leave empty for no maximum):",
            required=False
        )

        # Date availability
        criteria['date_from'] = self._get_date_input(
            "Start date for availability (YYYY-MM-DD):",
            required=False
        )
        criteria['date_to'] = self._get_date_input(
            "End date for availability (YYYY-MM-DD):",
            required=False
        )

        # Capacity
        criteria['capacity'] = self._get_number_input(
            "Minimum capacity (number of guests):",
            required=False
        )

        # Additional amenities
        criteria['amenities'] = self._get_list_input(
            "Any specific amenities required? (comma-separated)",
            help_text="e.g., Parking, WiFi, Catering, AV Equipment"
        )

        # Custom URLs to scrape
        criteria['custom_urls'] = self._get_list_input(
            "Any specific URLs to scrape? (comma-separated, optional)"
        )

        # Additional criteria
        criteria['additional_criteria'] = self._get_input(
            "Any other specific criteria or requirements?",
            required=False,
            help_text="Any other details you'd like to specify"
        )

        return criteria

    def _get_input(self, prompt: str, default: str = None,
                   required: bool = True, help_text: str = None) -> str:
        """
        Get text input from user

        Args:
            prompt: The question to ask
            default: Default value if user provides no input
            required: Whether input is required
            help_text: Optional help text to display

        Returns:
            User input string
        """
        if help_text:
            print(f"  ({help_text})")

        full_prompt = f"{prompt} "
        if default:
            full_prompt += f"[{default}] "
        full_prompt += ": "

        while True:
            value = input(full_prompt).strip()

            if not value and default:
                return default
            elif not value and not required:
                return None
            elif not value and required:
                print("  ⚠ This field is required. Please provide a value.")
                continue
            else:
                return value

    def _get_number_input(self, prompt: str, required: bool = True) -> Optional[float]:
        """
        Get numeric input from user

        Args:
            prompt: The question to ask
            required: Whether input is required

        Returns:
            Numeric value or None
        """
        while True:
            value = input(f"{prompt} ").strip()

            if not value and not required:
                return None
            elif not value and required:
                print("  ⚠ This field is required. Please provide a value.")
                continue

            try:
                return float(value)
            except ValueError:
                print("  ⚠ Please enter a valid number.")

    def _get_date_input(self, prompt: str, required: bool = True) -> Optional[str]:
        """
        Get date input from user and validate format

        Args:
            prompt: The question to ask
            required: Whether input is required

        Returns:
            Date string in YYYY-MM-DD format or None
        """
        while True:
            value = input(f"{prompt} ").strip()

            if not value and not required:
                return None
            elif not value and required:
                print("  ⚠ This field is required. Please provide a value.")
                continue

            try:
                # Validate date format
                datetime.strptime(value, "%Y-%m-%d")
                return value
            except ValueError:
                print("  ⚠ Please enter a valid date in YYYY-MM-DD format.")

    def _get_list_input(self, prompt: str, help_text: str = None) -> List[str]:
        """
        Get comma-separated list input from user

        Args:
            prompt: The question to ask
            help_text: Optional help text

        Returns:
            List of strings
        """
        if help_text:
            print(f"  ({help_text})")

        value = input(f"{prompt} ").strip()

        if not value:
            return []

        # Split by comma and clean up
        items = [item.strip() for item in value.split(',') if item.strip()]
        return items

    def confirm_action(self, message: str) -> bool:
        """
        Ask for user confirmation

        Args:
            message: Confirmation message

        Returns:
            True if user confirms, False otherwise
        """
        response = input(f"{message} (yes/no): ").strip().lower()
        return response in ['yes', 'y']
