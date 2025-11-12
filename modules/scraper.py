"""
Web Scraper Module
Handles scraping venue listings from multiple sources
"""

import requests
from bs4 import BeautifulSoup
import time
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse
import tldextract
from datetime import datetime
import logging


class VenueScraper:
    """Scrapes venue listings from various sources"""

    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """
        Initialize the scraper

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

        # User agent to avoid being blocked
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                         '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Common venue listing websites (can be expanded)
        self.venue_sources = [
            {
                'name': 'Venue Directory',
                'url_pattern': 'https://www.example-venue-site.com/search',
                'selectors': {
                    'listing': '.venue-card',
                    'title': '.venue-title',
                    'price': '.venue-price',
                    'location': '.venue-location',
                    'link': 'a.venue-link'
                }
            }
            # Add more sources as needed
        ]

    def scrape_multiple_sources(self, criteria: Dict, target_count: int = 50) -> List[Dict]:
        """
        Scrape venues from multiple sources

        Args:
            criteria: Search criteria dictionary
            target_count: Target number of listings to collect

        Returns:
            List of scraped listing dictionaries
        """
        all_listings = []

        self.logger.info("Starting multi-source scraping...")

        # If custom URLs provided, scrape those first
        if criteria.get('custom_urls'):
            for url in criteria['custom_urls']:
                self.logger.info(f"Scraping custom URL: {url}")
                print(f"Scraping: {url}")
                try:
                    listings = self.scrape_generic_page(url, criteria)
                    all_listings.extend(listings)
                    print(f"  Found {len(listings)} listings")
                except Exception as e:
                    self.logger.error(f"Failed to scrape {url}: {e}")
                    print(f"  ✗ Failed: {e}")

        # If we haven't reached target, try generic search
        if len(all_listings) < target_count:
            self.logger.info("Attempting generic venue searches...")
            print("\nSearching common venue listing sites...")

            # Generate search queries based on criteria
            search_queries = self._generate_search_queries(criteria)

            for query in search_queries:
                if len(all_listings) >= target_count:
                    break

                try:
                    listings = self._search_generic_venues(query, criteria)
                    all_listings.extend(listings)
                    print(f"  Found {len(listings)} listings for: {query}")
                except Exception as e:
                    self.logger.error(f"Search failed for '{query}': {e}")

        # If still not enough, generate sample listings for demonstration
        if len(all_listings) < target_count:
            self.logger.warning(
                f"Only found {len(all_listings)} listings. "
                f"Generating sample data to reach target..."
            )
            print(f"\n⚠ Only found {len(all_listings)} real listings.")
            print(f"  Generating {target_count - len(all_listings)} sample listings for demonstration...")

            sample_listings = self._generate_sample_listings(
                criteria,
                target_count - len(all_listings)
            )
            all_listings.extend(sample_listings)

        return all_listings[:target_count]

    def scrape_generic_page(self, url: str, criteria: Dict) -> List[Dict]:
        """
        Scrape a generic venue listing page

        Args:
            url: URL to scrape
            criteria: Search criteria for filtering

        Returns:
            List of listing dictionaries
        """
        listings = []

        try:
            response = self._make_request(url)
            if not response:
                return listings

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract domain info
            parsed_url = urlparse(url)
            domain_info = tldextract.extract(url)
            domain = f"{domain_info.domain}.{domain_info.suffix}"

            # Try to find listings using common patterns
            listing_containers = self._find_listing_containers(soup)

            for container in listing_containers[:20]:  # Limit per page
                try:
                    listing = self._extract_listing_data(
                        container, url, domain
                    )
                    if listing:
                        listings.append(listing)
                except Exception as e:
                    self.logger.debug(f"Failed to extract listing: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}")
            raise

        return listings

    def _make_request(self, url: str) -> Optional[requests.Response]:
        """
        Make HTTP request with retries and error handling

        Args:
            url: URL to request

        Returns:
            Response object or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout,
                    allow_redirects=True
                )
                response.raise_for_status()
                return response

            except requests.RequestException as e:
                self.logger.warning(
                    f"Request attempt {attempt + 1} failed for {url}: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"All retry attempts failed for {url}")
                    return None

        return None

    def _find_listing_containers(self, soup: BeautifulSoup) -> List:
        """
        Find listing containers using common patterns

        Args:
            soup: BeautifulSoup object

        Returns:
            List of container elements
        """
        # Common class/id patterns for venue listings
        patterns = [
            {'class': re.compile(r'.*venue.*card.*', re.I)},
            {'class': re.compile(r'.*listing.*item.*', re.I)},
            {'class': re.compile(r'.*property.*card.*', re.I)},
            {'class': re.compile(r'.*result.*item.*', re.I)},
            {'class': re.compile(r'.*event.*space.*', re.I)},
        ]

        containers = []
        for pattern in patterns:
            found = soup.find_all(['div', 'article', 'li'], pattern, limit=30)
            if found:
                containers.extend(found)

        # Remove duplicates while preserving order
        seen = set()
        unique_containers = []
        for container in containers:
            container_id = id(container)
            if container_id not in seen:
                seen.add(container_id)
                unique_containers.append(container)

        return unique_containers

    def _extract_listing_data(self, container, base_url: str,
                            domain: str) -> Optional[Dict]:
        """
        Extract listing data from container element

        Args:
            container: BeautifulSoup element containing listing
            base_url: Base URL for resolving relative links
            domain: Domain name

        Returns:
            Dictionary with listing data or None
        """
        listing = {
            'domain': domain,
            'scraped_at': datetime.now().isoformat(),
            'source_type': 'scraped'
        }

        # Extract title
        title_elem = container.find(['h1', 'h2', 'h3', 'h4'],
                                    class_=re.compile(r'.*(title|name|heading).*', re.I))
        if not title_elem:
            title_elem = container.find(['h1', 'h2', 'h3', 'h4'])
        listing['title'] = title_elem.get_text(strip=True) if title_elem else 'Unknown'

        # Extract URL
        link_elem = container.find('a', href=True)
        if link_elem:
            listing['url'] = urljoin(base_url, link_elem['href'])
        else:
            listing['url'] = base_url

        # Extract price
        price_elem = container.find(
            ['span', 'div', 'p'],
            class_=re.compile(r'.*(price|cost|rate).*', re.I)
        )
        if not price_elem:
            price_elem = container.find(
                text=re.compile(r'\$|€|£|USD|EUR|GBP', re.I)
            )
        listing['price'] = self._extract_price(price_elem) if price_elem else None

        # Extract location
        location_elem = container.find(
            ['span', 'div', 'p'],
            class_=re.compile(r'.*(location|address|city).*', re.I)
        )
        listing['location'] = location_elem.get_text(strip=True) if location_elem else None

        # Extract description
        desc_elem = container.find(
            ['p', 'div'],
            class_=re.compile(r'.*(description|summary|details).*', re.I)
        )
        listing['description'] = desc_elem.get_text(strip=True) if desc_elem else None

        # Extract capacity if available
        capacity_match = re.search(r'(\d+)\s*(guests?|people|capacity)',
                                  container.get_text(), re.I)
        listing['capacity'] = int(capacity_match.group(1)) if capacity_match else None

        # Extract reviews/ratings
        rating_elem = container.find(
            ['span', 'div'],
            class_=re.compile(r'.*(rating|stars|review).*', re.I)
        )
        if rating_elem:
            rating_text = rating_elem.get_text()
            rating_match = re.search(r'(\d+\.?\d*)', rating_text)
            listing['rating'] = float(rating_match.group(1)) if rating_match else None

        # Check for reviews
        review_elem = container.find(
            text=re.compile(r'(\d+)\s*reviews?', re.I)
        )
        if review_elem:
            review_match = re.search(r'(\d+)', review_elem)
            listing['review_count'] = int(review_match.group(1)) if review_match else 0
        else:
            listing['review_count'] = 0

        # Only return if we have at least a title
        if listing['title'] and listing['title'] != 'Unknown':
            return listing

        return None

    def _extract_price(self, price_element) -> Optional[str]:
        """
        Extract and normalize price from element

        Args:
            price_element: BeautifulSoup element or string

        Returns:
            Normalized price string
        """
        if hasattr(price_element, 'get_text'):
            price_text = price_element.get_text(strip=True)
        else:
            price_text = str(price_element)

        # Clean up price text
        price_text = re.sub(r'\s+', ' ', price_text)
        return price_text if price_text else None

    def _generate_search_queries(self, criteria: Dict) -> List[str]:
        """
        Generate search queries based on criteria

        Args:
            criteria: Search criteria dictionary

        Returns:
            List of search query strings
        """
        queries = []

        venue_type = criteria.get('venue_type', 'venue')
        location = criteria.get('location', '')

        # Combine venue type and location
        if location and location != 'Any location':
            queries.append(f"{venue_type} in {location}")
            queries.append(f"{location} {venue_type}")
        else:
            queries.append(venue_type)

        return queries

    def _search_generic_venues(self, query: str, criteria: Dict) -> List[Dict]:
        """
        Search for venues using generic patterns

        Args:
            query: Search query string
            criteria: Search criteria

        Returns:
            List of listing dictionaries
        """
        # This is a placeholder for actual search functionality
        # In a real implementation, you would integrate with actual venue APIs
        # or search engines

        self.logger.info(f"Generic search for: {query}")
        return []

    def _generate_sample_listings(self, criteria: Dict, count: int) -> List[Dict]:
        """
        Generate sample listings for demonstration purposes

        Args:
            criteria: Search criteria
            count: Number of samples to generate

        Returns:
            List of sample listing dictionaries
        """
        samples = []
        location = criteria.get('location', 'Various Locations')
        venue_type = criteria.get('venue_type', 'Event Venue')

        venue_names = [
            'Grand', 'Elite', 'Premium', 'Royal', 'Imperial',
            'Elegant', 'Majestic', 'Luxe', 'Classic', 'Modern',
            'Urban', 'Rustic', 'Vintage', 'Contemporary', 'Boutique'
        ]

        venue_suffixes = [
            'Hall', 'Center', 'Space', 'Gallery', 'Ballroom',
            'Pavilion', 'Manor', 'Estate', 'Loft', 'Garden'
        ]

        for i in range(count):
            name = f"{venue_names[i % len(venue_names)]} {venue_suffixes[i % len(venue_suffixes)]}"

            listing = {
                'title': f"{name} - {venue_type}",
                'url': f"https://example-venue-site-{i}.com/venues/{i}",
                'domain': f"example-venue-site-{i % 10}.com",
                'price': self._generate_sample_price(i),
                'location': location,
                'description': f"Beautiful {venue_type.lower()} perfect for your special event. "
                              f"Located in {location} with modern amenities.",
                'capacity': 50 + (i * 10),
                'rating': round(3.5 + (i % 15) / 10, 1),
                'review_count': i * 5 + 10,
                'scraped_at': datetime.now().isoformat(),
                'source_type': 'sample'
            }

            samples.append(listing)

        return samples

    def _generate_sample_price(self, seed: int) -> str:
        """Generate sample price with various formats"""
        base_price = 1000 + (seed * 500)

        # Mix of pricing formats including vague ones
        formats = [
            f"${base_price:,}",
            f"${base_price:,} per event",
            f"From ${base_price:,}",  # Vague
            f"Starting at ${base_price:,}",  # Vague
            f"${base_price:,} - ${base_price + 2000:,}",
            f"Approx. ${base_price:,}",  # Vague
            f"${base_price:,}/day",
            f"Contact for pricing",  # Vague
        ]

        return formats[seed % len(formats)]
