# Venue Listing Validator

A comprehensive Python application that scrapes, validates, and ranks venue listings from multiple sources. The tool performs legitimacy checks, detects vague pricing, validates reviews, and exports results to Excel or Google Sheets.

## Features

- **Interactive CLI**: User-friendly command-line interface for collecting search criteria
- **Multi-Source Scraping**: Scrape venue listings from multiple websites or custom URLs
- **Comprehensive Validation**:
  - HTTPS verification
  - Domain age checking via WHOIS
  - Vague pricing detection (flags "from", "starting at", "approx.", etc.)
  - Availability validation
  - Review authenticity analysis
  - SEO spam detection
- **Intelligent Ranking**: Ranks listings by legitimacy, completeness, and relevance
- **Duplicate Detection**: Removes duplicate listings automatically
- **Excel Export**: Detailed Excel reports with multiple sheets
- **Google Sheets Integration**: Optional cloud-based reporting
- **Comprehensive Logging**: Detailed logs for debugging and auditing
- **Error Handling**: Robust error handling with retry mechanisms

## Requirements

- Python 3.8 or higher
- Internet connection for scraping and domain validation
- (Optional) Google Cloud account for Google Sheets integration

## Installation

### 1. Clone or Download the Repository

```bash
cd venue-listing-validator
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

### Basic Configuration

Edit `config.py` to customize:
- Scraping timeouts and retry settings
- Validation thresholds
- Ranking weights
- Venue sources to scrape

### Google Sheets Setup (Optional)

If you want to export results to Google Sheets, follow these steps:

#### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the following APIs:
   - Google Sheets API
   - Google Drive API

#### Step 2: Create Service Account

1. Navigate to "IAM & Admin" → "Service Accounts"
2. Click "Create Service Account"
3. Give it a name (e.g., "venue-validator")
4. Click "Create and Continue"
5. Grant the role "Editor" (or customize permissions)
6. Click "Done"

#### Step 3: Generate Credentials

1. Click on the created service account
2. Go to "Keys" tab
3. Click "Add Key" → "Create New Key"
4. Choose "JSON" format
5. Click "Create" - this will download a JSON file

#### Step 4: Configure Application

1. Rename the downloaded JSON file to `google_credentials.json`
2. Place it in the project root directory (same directory as `venue_validator.py`)
3. **IMPORTANT**: Add `google_credentials.json` to `.gitignore` to avoid committing sensitive data

```bash
echo "google_credentials.json" >> .gitignore
```

#### Step 5: Share Spreadsheets (Optional)

When the application creates a Google Sheet, you can:
- Update the code in `modules/export_manager.py` to automatically share with your email
- Or manually share the created spreadsheet from Google Drive

## Usage

### Basic Usage

Run the application:

```bash
python venue_validator.py
```

The application will guide you through an interactive prompt asking for:

1. **Location**: Where you want to search for venues
2. **Venue Type**: Type of venue you're looking for
3. **Price Range**: Minimum and maximum price
4. **Date Availability**: Start and end dates
5. **Capacity**: Minimum guest capacity
6. **Amenities**: Specific amenities required
7. **Custom URLs**: Optional specific websites to scrape
8. **Additional Criteria**: Any other requirements

### Example Session

```
======================================================================
VENUE LISTING VALIDATOR - Search Criteria Collection
======================================================================

Please provide your venue search criteria:

What location or venue area do you want? : New York City
What type of venue or listing are you looking for? : Wedding venue
Minimum price (leave empty for no minimum): : 5000
Maximum price (leave empty for no maximum): : 15000
Start date for availability (YYYY-MM-DD): : 2025-06-01
End date for availability (YYYY-MM-DD): : 2025-09-30
Minimum capacity (number of guests): : 150
Any specific amenities required? (comma-separated) : Parking, Catering, Dance floor
Any specific URLs to scrape? (comma-separated, optional) : 
Any other specific criteria or requirements? : Outdoor space preferred

----------------------------------------------------------------------
Search Criteria Summary:
----------------------------------------------------------------------
  Location: New York City
  Venue Type: Wedding venue
  Price Min: 5000.0
  Price Max: 15000.0
  Date From: 2025-06-01
  Date To: 2025-09-30
  Capacity: 150.0
  Amenities: ['Parking', 'Catering', 'Dance floor']
  Custom Urls: []
  Additional Criteria: Outdoor space preferred
----------------------------------------------------------------------

Proceed with these criteria? (yes/no): yes
```

### Output Files

The application generates several files:

1. **Excel File** (`venue_listings_YYYYMMDD_HHMMSS.xlsx`):
   - Sheet 1: Valid Listings - All validated venue listings
   - Sheet 2: Excluded Listings - Listings that failed validation
   - Sheet 3: Summary - Statistics and breakdown

2. **Summary Report** (`summary_report_YYYYMMDD_HHMMSS.txt`):
   - Detailed text report with exclusion reasons
   - Statistics and analysis

3. **Log File** (`logs/venue_validator_YYYYMMDD_HHMMSS.log`):
   - Detailed application logs
   - Useful for debugging

4. **Google Sheet** (if enabled):
   - Cloud-based version of the Excel file
   - Shareable link provided in output

## Understanding the Results

### Excel Columns

The Excel export includes the following columns:

**Basic Information:**
- Rank: Position in sorted results
- Title: Venue name/title
- URL: Link to listing
- Domain: Website domain

**Legitimacy Checks:**
- HTTPS: Whether site uses secure connection
- Domain Age (days): How old the domain is
- Domain Created: Domain registration date
- Registrar: Domain registrar

**Pricing:**
- Price: Listed price
- Vague Pricing: Yes/No if pricing is unclear
- Pricing Issue: Details about pricing problems

**Details:**
- Location: Venue location
- Capacity: Guest capacity
- Description: Venue description

**Reviews:**
- Rating: Average rating
- Review Count: Number of reviews
- Review Authenticity Score: Percentage score
- Suspicious Reviews: Count of suspicious reviews

**Validation:**
- SEO Spam: Spam indicators detected
- Data Completeness: Percentage of fields filled
- Is Legitimate: Overall legitimacy verdict
- Exclusion Reasons: Why listing was excluded
- Ranking Score: Overall quality score (0-100)

### Ranking Score

Listings are ranked based on weighted factors:
- **Legitimacy** (30%): Passes security and validation checks
- **Data Completeness** (20%): Has comprehensive information
- **Review Authenticity** (20%): Reviews appear genuine
- **Price Transparency** (15%): Clear, non-vague pricing
- **Domain Trust** (15%): Established, trustworthy domain

## Customization

### Adding More Venue Sources

Edit `config.py` and add to `VENUE_SOURCES`:

```python
VENUE_SOURCES = [
    {
        'name': 'YourVenueSite',
        'base_url': 'https://www.yourvenusite.com',
        'search_pattern': '/search'
    },
    # ... more sources
]
```

### Adjusting Validation Rules

Modify `config.py`:

```python
VALIDATION_CONFIG = {
    'min_domain_age_days': 90,  # Increase for stricter validation
    'require_https': True,  # Set to False to allow HTTP
    'min_data_completeness': 0.3,  # Increase for more complete data
}
```

### Changing Ranking Weights

Adjust weights in `config.py` (must sum to 1.0):

```python
RANKING_WEIGHTS = {
    'legitimacy': 0.40,  # Prioritize legitimacy more
    'data_completeness': 0.20,
    'review_authenticity': 0.15,
    'price_transparency': 0.15,
    'domain_trust': 0.10
}
```

## Troubleshooting

### Common Issues

**1. Module Not Found Error**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt --upgrade
```

**2. WHOIS Lookup Failures**
- WHOIS lookups may fail for some domains
- The application continues with a warning
- Domain age will be marked as "Unknown"

**3. Scraping Timeouts**
- Some websites may be slow or block requests
- Adjust timeout in `config.py`
- Check your internet connection

**4. Google Sheets Authentication Error**
- Verify `google_credentials.json` exists in project root
- Ensure Google Sheets API and Drive API are enabled
- Check service account has necessary permissions

**5. No Listings Found**
- Try providing custom URLs in the criteria
- Check if venue sources in `config.py` are accessible
- Review logs for detailed error messages

### Viewing Logs

Detailed logs are saved in the `logs/` directory:

```bash
# View latest log
ls -lt logs/ | head -1
cat logs/venue_validator_YYYYMMDD_HHMMSS.log
```

## Extending the Application

### Adding Custom Scrapers

Create a new scraper in `modules/scraper.py`:

```python
def scrape_custom_site(self, url: str) -> List[Dict]:
    """Custom scraper for specific website"""
    # Implement your scraping logic
    pass
```

### Adding Custom Validators

Add validation logic in `modules/validator.py`:

```python
def _check_custom_validation(self, listing: Dict) -> Dict:
    """Custom validation check"""
    # Implement your validation logic
    pass
```

### Custom Export Formats

Add export methods in `modules/export_manager.py`:

```python
def export_to_json(self, listings: List[Dict], filename: str):
    """Export to JSON format"""
    # Implement JSON export
    pass
```

## Project Structure

```
venue-listing-validator/
│
├── venue_validator.py          # Main application entry point
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── google_credentials.json      # (Optional) Google API credentials
│
├── modules/                     # Application modules
│   ├── __init__.py
│   ├── cli_interface.py        # Interactive CLI
│   ├── scraper.py              # Web scraping logic
│   ├── validator.py            # Legitimacy validation
│   ├── review_analyzer.py      # Review authenticity
│   ├── ranking_engine.py       # Ranking and deduplication
│   ├── export_manager.py       # Excel/Sheets export
│   └── logger.py               # Logging setup
│
└── logs/                        # Log files (auto-created)
    └── venue_validator_*.log
```

## Security & Privacy

- **Never commit `google_credentials.json`** to version control
- Be respectful of website terms of service when scraping
- Implement rate limiting to avoid overwhelming servers
- Review logs for sensitive information before sharing

## License

This project is provided as-is for educational and personal use.

## Support

For issues, questions, or contributions:
1. Check the logs in `logs/` directory
2. Review this README's troubleshooting section
3. Ensure all dependencies are up to date

## Version History

- **v1.0.0** - Initial release with full feature set

## Credits

Built with Python and the following open-source libraries:
- requests
- beautifulsoup4
- pandas
- openpyxl
- python-whois
- gspread
