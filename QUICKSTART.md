# Quick Start Guide - Venue Listing Validator

Get started with the Venue Listing Validator in 5 minutes!

## Prerequisites

- Python 3.8 or higher installed
- Internet connection

## Installation Steps

### 1. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python venue_validator.py
```

## First Run Example

When you run the application, you'll be prompted for search criteria:

```
What location or venue area do you want? : Los Angeles
What type of venue or listing are you looking for? : Wedding venue  
Minimum price (leave empty for no minimum): : 3000
Maximum price (leave empty for no maximum): : 10000
Start date for availability (YYYY-MM-DD): : 2025-07-01
End date for availability (YYYY-MM-DD): : 2025-12-31
Minimum capacity (number of guests): : 100
Any specific amenities required? (comma-separated) : Parking, WiFi
Any specific URLs to scrape? (comma-separated, optional) : 
Any other specific criteria or requirements? : Outdoor space
```

Press Enter after each input. The application will then:
1. Scrape venue listings (50+ venues)
2. Validate each listing for legitimacy
3. Rank and filter results
4. Export to Excel
5. Optionally export to Google Sheets

## Output Files

After completion, you'll find:

1. **Excel file**: `venue_listings_YYYYMMDD_HHMMSS.xlsx`
   - Open with Excel or Google Sheets
   - Contains Valid Listings, Excluded Listings, and Summary

2. **Summary report**: `summary_report_YYYYMMDD_HHMMSS.txt`
   - Text summary of results

3. **Log file**: `logs/venue_validator_YYYYMMDD_HHMMSS.log`
   - Detailed application logs

## Understanding the Excel Output

The Excel file has three sheets:

### Sheet 1: Valid Listings
All venue listings that passed validation checks, with columns:
- Rank, Title, URL, Domain
- HTTPS status, Domain Age
- Price, Vague Pricing flag
- Location, Capacity
- Rating, Review Count, Review Authenticity
- SEO Spam detection
- Overall Ranking Score

### Sheet 2: Excluded Listings
Listings that failed validation, with reasons for exclusion

### Sheet 3: Summary
Statistics including:
- Total listings scraped
- Valid vs. excluded counts
- Breakdown of exclusion reasons

## Google Sheets Setup (Optional)

To export to Google Sheets:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project
3. Enable Google Sheets API and Google Drive API
4. Create a Service Account
5. Download JSON credentials
6. Rename to `google_credentials.json` and place in project root

See README.md for detailed instructions.

## Tips for Best Results

1. **Provide Custom URLs**: If you know specific venue listing sites, provide them in "custom URLs"
2. **Be Specific**: More specific criteria = better matching results
3. **Review Logs**: Check `logs/` directory if something goes wrong
4. **Adjust Settings**: Edit `config.py` to customize validation rules

## Common Commands

```bash
# Run the application
python venue_validator.py

# View latest log
ls -lt logs/ | head -1

# Upgrade dependencies
pip install -r requirements.txt --upgrade
```

## Need Help?

- Read the full README.md for detailed documentation
- Check logs in `logs/` directory for errors
- Ensure all dependencies are installed

## What Gets Validated?

The application checks:
- ✓ HTTPS usage (secure connection)
- ✓ Domain age (filters very new domains)
- ✓ Vague pricing (flags "from $X", "starting at", etc.)
- ✓ Review authenticity (detects fake reviews)
- ✓ SEO spam (filters spam content)
- ✓ Data completeness (ensures sufficient info)

## Next Steps

1. Run the application with your criteria
2. Review the Excel output
3. Check excluded listings to understand filtering
4. Adjust `config.py` if needed
5. Re-run with refined criteria

Happy venue hunting! 🎉
