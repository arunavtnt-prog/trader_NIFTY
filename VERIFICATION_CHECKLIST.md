# Venue Listing Validator - Verification Checklist

Use this checklist to verify the application is properly set up and working.

## Pre-Installation Checks

- [ ] Python 3.8 or higher installed
  ```bash
  python --version
  ```

- [ ] pip is available
  ```bash
  pip --version
  ```

- [ ] Internet connection active

## Installation Verification

- [ ] Virtual environment created
  ```bash
  python -m venv venv
  ```

- [ ] Virtual environment activated
  ```bash
  # Windows: venv\Scripts\activate
  # macOS/Linux: source venv/bin/activate
  ```

- [ ] Dependencies installed
  ```bash
  pip install -r requirements.txt
  ```

- [ ] No installation errors

## File Structure Verification

- [ ] Main application file exists: `venue_validator.py`
- [ ] Configuration file exists: `config.py`
- [ ] Requirements file exists: `requirements.txt`
- [ ] README file exists: `README.md`
- [ ] Quick start guide exists: `QUICKSTART.md`
- [ ] Modules directory exists: `modules/`
- [ ] All module files present:
  - [ ] `modules/__init__.py`
  - [ ] `modules/cli_interface.py`
  - [ ] `modules/scraper.py`
  - [ ] `modules/validator.py`
  - [ ] `modules/review_analyzer.py`
  - [ ] `modules/ranking_engine.py`
  - [ ] `modules/export_manager.py`
  - [ ] `modules/logger.py`

## Functionality Verification

### Test 1: Import Verification
```bash
python -c "from modules import cli_interface, scraper, validator, review_analyzer, ranking_engine, export_manager, logger; print('✓ All modules can be imported')"
```
- [ ] Imports successful

### Test 2: Configuration Loading
```bash
python -c "import config; print('✓ Configuration loaded'); print(f'Timeout: {config.SCRAPING_CONFIG[\"timeout\"]}s')"
```
- [ ] Configuration loads correctly

### Test 3: Example Script
```bash
python example_usage.py
```
- [ ] Example runs without errors
- [ ] Shows validation results
- [ ] Displays ranking score

### Test 4: Main Application Launch
```bash
python venue_validator.py
```
- [ ] Application starts
- [ ] Shows welcome banner
- [ ] Prompts for search criteria
- [ ] Can be cancelled with Ctrl+C

## Module-Specific Checks

### CLI Interface
- [ ] Accepts text input
- [ ] Validates date format (YYYY-MM-DD)
- [ ] Validates numeric input
- [ ] Handles comma-separated lists
- [ ] Shows help text

### Scraper
- [ ] Makes HTTP requests
- [ ] Handles timeouts gracefully
- [ ] Retries on failure
- [ ] Generates sample data if needed

### Validator
- [ ] Checks HTTPS
- [ ] Performs WHOIS lookups (may fail for some domains - expected)
- [ ] Detects vague pricing
- [ ] Identifies SEO spam
- [ ] Calculates completeness score

### Review Analyzer
- [ ] Detects fake review patterns
- [ ] Counts superlatives
- [ ] Analyzes collective patterns
- [ ] Calculates authenticity score

### Ranking Engine
- [ ] Calculates weighted scores
- [ ] Ranks listings
- [ ] Detects duplicates
- [ ] Filters by criteria

### Export Manager
- [ ] Creates Excel files
- [ ] Formats worksheets
- [ ] Creates multiple sheets
- [ ] Handles Google Sheets (if configured)

### Logger
- [ ] Creates logs directory
- [ ] Writes log files
- [ ] Timestamps entries
- [ ] Logs errors appropriately

## Output Verification

After running a complete session:

- [ ] Excel file created (`venue_listings_*.xlsx`)
- [ ] Excel file opens without errors
- [ ] Valid Listings sheet present
- [ ] Excluded Listings sheet present
- [ ] Summary sheet present
- [ ] Summary report created (`summary_report_*.txt`)
- [ ] Summary report readable
- [ ] Log file created (`logs/venue_validator_*.log`)
- [ ] Log file contains entries

## Excel Output Quality Checks

- [ ] Headers are bold and colored
- [ ] Columns are properly sized
- [ ] Data is readable
- [ ] No truncated important data
- [ ] Rankings are in order
- [ ] Scores are calculated
- [ ] URLs are clickable

## Google Sheets Verification (Optional)

If using Google Sheets:

- [ ] `google_credentials.json` file present
- [ ] Google Sheets API enabled
- [ ] Google Drive API enabled
- [ ] Service account created
- [ ] Credentials are valid JSON
- [ ] Can create spreadsheet
- [ ] Spreadsheet URL returned
- [ ] Spreadsheet is accessible

## Error Handling Verification

Test these scenarios:

- [ ] Invalid date format → Shows error, asks again
- [ ] Invalid number → Shows error, asks again
- [ ] Network timeout → Logs warning, continues
- [ ] Missing credentials → Shows helpful error
- [ ] Empty results → Generates sample data
- [ ] Ctrl+C during execution → Exits gracefully

## Performance Verification

- [ ] 50 listings processed in reasonable time (< 5 minutes)
- [ ] Memory usage remains stable
- [ ] No memory leaks during execution
- [ ] Logs don't grow excessively large

## Security Verification

- [ ] `.gitignore` includes `google_credentials.json`
- [ ] `.gitignore` includes `*.log`
- [ ] No credentials in code
- [ ] No hardcoded passwords
- [ ] Sample URLs use HTTPS

## Documentation Verification

- [ ] README is complete
- [ ] QUICKSTART is clear
- [ ] PROJECT_SUMMARY is accurate
- [ ] Code has comments
- [ ] Functions have docstrings
- [ ] Configuration is documented

## Final Verification

- [ ] Can run complete end-to-end session
- [ ] Results make sense
- [ ] Excluded listings have reasons
- [ ] Rankings reflect quality
- [ ] No critical errors in logs
- [ ] Application exits cleanly

## Known Limitations (Expected Behavior)

These are not errors:

- WHOIS lookups may fail for some domains
- Some websites may block scraping
- Sample data is generated if real scraping fails
- Google Sheets requires manual setup
- Some validation checks are best-effort

## Troubleshooting Steps

If any check fails:

1. Check Python version (must be 3.8+)
2. Verify virtual environment is activated
3. Re-run `pip install -r requirements.txt`
4. Check logs in `logs/` directory
5. Review error messages carefully
6. Ensure internet connection is stable
7. Check file permissions

## Success Criteria

✓ All critical checks pass
✓ Application runs end-to-end
✓ Output files are generated
✓ Results are meaningful
✓ No unhandled exceptions

---

**Date Verified**: __________
**Verified By**: __________
**Notes**: __________
