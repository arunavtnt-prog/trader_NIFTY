# Venue Listing Validator - Project Summary

## Overview

A comprehensive Python application for scraping, validating, and ranking venue listings with advanced legitimacy checking and duplicate detection.

## Project Structure

```
venue-listing-validator/
│
├── venue_validator.py              # Main application (run this)
├── example_usage.py                 # Programmatic usage example
├── config.py                        # Configuration settings
├── requirements.txt                 # Python dependencies
│
├── README.md                        # Full documentation
├── QUICKSTART.md                    # Quick start guide
├── PROJECT_SUMMARY.md               # This file
│
├── .gitignore                       # Git ignore rules
├── google_credentials_template.json # Template for Google API setup
│
└── modules/                         # Core application modules
    ├── __init__.py
    ├── cli_interface.py            # Interactive CLI for user input
    ├── scraper.py                  # Web scraping functionality
    ├── validator.py                # Legitimacy validation (HTTPS, domain age, pricing)
    ├── review_analyzer.py          # Review authenticity detection
    ├── ranking_engine.py           # Ranking & duplicate removal
    ├── export_manager.py           # Excel & Google Sheets export
    └── logger.py                   # Logging and error handling
```

## Key Features

### 1. Interactive CLI (cli_interface.py)
- Collects search criteria through user-friendly prompts
- Location, venue type, price range, dates, capacity
- Custom URLs and additional criteria
- Input validation and help text

### 2. Web Scraping (scraper.py)
- Multi-source scraping support
- Custom URL scraping
- Generic venue site patterns
- Retry logic with exponential backoff
- Sample data generation for testing
- Timeout and error handling

### 3. Legitimacy Validation (validator.py)
- **HTTPS Verification**: Ensures secure connections
- **Domain Age Checking**: WHOIS lookups to verify domain age
- **Vague Pricing Detection**: Flags "from", "starting at", "approx.", etc.
- **SEO Spam Detection**: Identifies spam content and black-hat SEO
- **URL Validation**: Checks for suspicious URLs and patterns
- **Data Completeness**: Scores listings by information completeness
- **Price Reasonability**: Detects suspiciously low/high prices

### 4. Review Analysis (review_analyzer.py)
- Fake review pattern detection
- Superlative overuse detection
- Generic phrase identification
- Suspicious reviewer name detection
- Review similarity analysis (copy-paste detection)
- Rating distribution analysis
- Authenticity scoring (0-100%)

### 5. Ranking Engine (ranking_engine.py)
- Weighted scoring system:
  - Legitimacy: 30%
  - Data completeness: 20%
  - Review authenticity: 20%
  - Price transparency: 15%
  - Domain trust: 15%
- Relevance bonus for criteria matching
- Duplicate detection (URL and content-based)
- Advanced similarity matching

### 6. Export Manager (export_manager.py)
- **Excel Export**: Multi-sheet workbooks with formatting
  - Valid Listings sheet
  - Excluded Listings sheet
  - Summary Statistics sheet
- **Google Sheets Export**: Cloud-based sharing
- **CSV Export**: Simple data export
- Auto-formatted headers and columns

### 7. Logging (logger.py)
- File-based detailed logging
- Console warnings and errors
- Timestamped log files
- Structured error handling

## Installation

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run application
python venue_validator.py
```

## Dependencies

- **requests**: HTTP requests for scraping
- **beautifulsoup4**: HTML parsing
- **lxml**: Fast XML/HTML parser
- **pandas**: Data manipulation
- **openpyxl**: Excel file generation
- **python-whois**: Domain age verification
- **tldextract**: Domain extraction
- **gspread**: Google Sheets integration (optional)
- **google-auth**: Google API authentication (optional)

## Configuration

All settings in `config.py`:

```python
# Scraping
SCRAPING_CONFIG = {
    'timeout': 30,
    'max_retries': 3,
    'delay_between_requests': 1
}

# Validation
VALIDATION_CONFIG = {
    'min_domain_age_days': 90,
    'require_https': True,
    'min_data_completeness': 0.3
}

# Ranking
RANKING_WEIGHTS = {
    'legitimacy': 0.30,
    'data_completeness': 0.20,
    'review_authenticity': 0.20,
    'price_transparency': 0.15,
    'domain_trust': 0.15
}
```

## Usage Flow

1. **User Input** → CLI collects search criteria
2. **Scraping** → Fetch listings from multiple sources
3. **Validation** → Check each listing for legitimacy
4. **Review Analysis** → Detect fake reviews
5. **Ranking** → Score and sort by quality
6. **Deduplication** → Remove duplicate listings
7. **Export** → Generate Excel/Google Sheets
8. **Summary** → Generate text report

## Output Files

### Excel File (venue_listings_TIMESTAMP.xlsx)
- **Valid Listings**: Rank, Title, URL, Domain, HTTPS, Domain Age, Price, Vague Pricing, Location, Capacity, Rating, Reviews, SEO Spam, Data Completeness, Ranking Score
- **Excluded Listings**: Same columns plus exclusion reasons
- **Summary**: Statistics and breakdown

### Summary Report (summary_report_TIMESTAMP.txt)
- Total listings scraped
- Valid vs. excluded counts
- Exclusion reasons breakdown
- Detailed listing information

### Log File (logs/venue_validator_TIMESTAMP.log)
- Detailed application logs
- Error traces
- Validation details

## Validation Checks

| Check | Description | Action |
|-------|-------------|--------|
| HTTPS | Verifies secure connection | Exclude if HTTP |
| Domain Age | WHOIS lookup for creation date | Warn if < 90 days |
| Vague Pricing | Detects unclear pricing terms | Exclude if vague |
| SEO Spam | Identifies spam content | Exclude if spam |
| URL Validity | Checks for suspicious URLs | Exclude if invalid |
| Data Completeness | Scores information completeness | Warn if < 30% |
| Review Authenticity | Detects fake reviews | Score 0-100% |
| Price Reasonability | Validates price ranges | Warn if suspicious |

## Ranking Score Calculation

```
Final Score = (
    Legitimacy × 0.30 +
    Data Completeness × 0.20 +
    Review Authenticity × 0.20 +
    Price Transparency × 0.15 +
    Domain Trust × 0.15
) + Relevance Bonus (0-10)
```

## Extension Points

### Add Custom Scrapers
```python
# In modules/scraper.py
def scrape_custom_site(self, url):
    # Your scraping logic
    pass
```

### Add Custom Validators
```python
# In modules/validator.py
def _check_custom_validation(self, listing):
    # Your validation logic
    pass
```

### Add Custom Ranking Factors
```python
# In modules/ranking_engine.py
def _calculate_custom_score(self, listing):
    # Your scoring logic
    pass
```

## Best Practices

1. **Always use virtual environment** to avoid dependency conflicts
2. **Keep credentials secure** - never commit `google_credentials.json`
3. **Respect rate limits** - adjust delays in config
4. **Review logs** for troubleshooting
5. **Test with small datasets** before large scraping operations
6. **Back up configurations** before making changes

## Google Sheets Setup

1. Create Google Cloud project
2. Enable Google Sheets API + Drive API
3. Create service account
4. Download JSON credentials
5. Rename to `google_credentials.json`
6. Place in project root

See README.md for detailed instructions.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found | `pip install -r requirements.txt --upgrade` |
| WHOIS failures | Normal for some domains, continues with warning |
| Scraping timeouts | Increase timeout in config.py |
| Google Sheets auth error | Verify credentials file and API enablement |
| No listings found | Provide custom URLs or check logs |

## Performance

- **Scraping**: ~1-2 seconds per page
- **Validation**: ~0.5-1 second per listing (WHOIS lookup)
- **50 listings**: ~2-3 minutes total
- **100 listings**: ~5-6 minutes total

## Security Considerations

- HTTPS validation ensures secure connections
- Domain age prevents newly registered scam sites
- Review analysis detects fake testimonials
- SEO spam detection filters malicious content
- Vague pricing detection prevents bait-and-switch

## Future Enhancements

- [ ] Availability calendar verification
- [ ] Image analysis for venue photos
- [ ] Social media verification
- [ ] Price history tracking
- [ ] Review sentiment analysis
- [ ] API integrations with venue platforms
- [ ] Machine learning for fake detection
- [ ] Parallel scraping for speed

## License

Provided as-is for educational and personal use.

## Version

**v1.0.0** - Complete implementation with all requested features

---

**Created**: 2025
**Last Updated**: 2025-01-12
