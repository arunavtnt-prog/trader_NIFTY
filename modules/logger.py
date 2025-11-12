"""
Logger Module
Sets up comprehensive logging for the application
"""

import logging
import sys
from datetime import datetime
import os


def setup_logger(log_level=logging.INFO, log_file=None):
    """
    Set up application-wide logger

    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Generate log filename if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/venue_validator_{timestamp}.log"

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove existing handlers
    logger.handlers = []

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # File handler (detailed logging)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler (simpler logging)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    logger.info("="*70)
    logger.info("Venue Listing Validator - Session Started")
    logger.info(f"Log file: {log_file}")
    logger.info("="*70)

    return logger


class ErrorHandler:
    """Centralized error handling"""

    @staticmethod
    def handle_scraping_error(url: str, error: Exception, logger):
        """Handle scraping errors"""
        logger.error(f"Scraping error for {url}: {type(error).__name__} - {str(error)}")
        return {
            'error': True,
            'url': url,
            'error_type': type(error).__name__,
            'error_message': str(error)
        }

    @staticmethod
    def handle_validation_error(listing: dict, error: Exception, logger):
        """Handle validation errors"""
        logger.error(
            f"Validation error for {listing.get('url', 'unknown')}: "
            f"{type(error).__name__} - {str(error)}"
        )
        return {
            'error': True,
            'listing_url': listing.get('url', 'unknown'),
            'error_type': type(error).__name__,
            'error_message': str(error)
        }

    @staticmethod
    def handle_export_error(filename: str, error: Exception, logger):
        """Handle export errors"""
        logger.error(
            f"Export error for {filename}: "
            f"{type(error).__name__} - {str(error)}"
        )
        return {
            'error': True,
            'filename': filename,
            'error_type': type(error).__name__,
            'error_message': str(error)
        }
