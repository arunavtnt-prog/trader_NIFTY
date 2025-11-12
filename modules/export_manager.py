"""
Export Manager Module
Handles exporting results to Excel and Google Sheets
"""

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows
from typing import Dict, List, Optional
import logging
import os


class ExportManager:
    """Manages export of results to various formats"""

    def __init__(self):
        """Initialize the export manager"""
        self.logger = logging.getLogger(__name__)

    def export_to_excel(self, valid_listings: List[Dict],
                       excluded_listings: List[Dict],
                       filename: str) -> str:
        """
        Export results to Excel file with multiple sheets

        Args:
            valid_listings: List of valid listing dictionaries
            excluded_listings: List of excluded listing dictionaries
            filename: Output filename

        Returns:
            Path to created file
        """
        self.logger.info(f"Exporting to Excel: {filename}")

        # Create Excel writer
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: Valid Listings
            if valid_listings:
                valid_df = self._prepare_listings_dataframe(valid_listings)
                valid_df.to_excel(writer, sheet_name='Valid Listings', index=False)
                self._format_worksheet(writer.sheets['Valid Listings'])

            # Sheet 2: Excluded Listings
            if excluded_listings:
                excluded_df = self._prepare_listings_dataframe(excluded_listings)
                excluded_df.to_excel(writer, sheet_name='Excluded Listings', index=False)
                self._format_worksheet(writer.sheets['Excluded Listings'])

            # Sheet 3: Summary Statistics
            summary_df = self._create_summary_dataframe(
                valid_listings, excluded_listings
            )
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            self._format_worksheet(writer.sheets['Summary'])

        self.logger.info(f"Excel export completed: {filename}")
        return filename

    def _prepare_listings_dataframe(self, listings: List[Dict]) -> pd.DataFrame:
        """
        Prepare listings data for DataFrame

        Args:
            listings: List of listing dictionaries

        Returns:
            pandas DataFrame
        """
        rows = []

        for listing in listings:
            validation = listing.get('validation', {})
            checks = validation.get('checks', {})

            # Extract validation details
            https_check = checks.get('https', {})
            domain_check = checks.get('domain_age', {})
            pricing_check = checks.get('vague_pricing', {})
            spam_check = checks.get('seo_spam', {})
            completeness_check = checks.get('data_completeness', {})
            review_auth = validation.get('review_authenticity', {})

            row = {
                'Rank': listing.get('rank', 'N/A'),
                'Title': listing.get('title', ''),
                'URL': listing.get('url', ''),
                'Domain': listing.get('domain', ''),

                # Legitimacy checks
                'HTTPS': 'Yes' if https_check.get('uses_https') else 'No',
                'Domain Age (days)': domain_check.get('age_days', 'Unknown'),
                'Domain Created': domain_check.get('creation_date', 'Unknown'),
                'Registrar': domain_check.get('registrar', 'Unknown'),

                # Pricing
                'Price': listing.get('price', 'N/A'),
                'Vague Pricing': 'Yes' if pricing_check.get('is_vague') else 'No',
                'Pricing Issue': pricing_check.get('reason', 'None'),

                # Location & Details
                'Location': listing.get('location', 'N/A'),
                'Capacity': listing.get('capacity', 'N/A'),
                'Description': listing.get('description', 'N/A'),

                # Reviews
                'Rating': listing.get('rating', 'N/A'),
                'Review Count': listing.get('review_count', 0),
                'Review Authenticity Score': f"{review_auth.get('authenticity_score', 0) * 100:.0f}%" if review_auth else 'N/A',
                'Suspicious Reviews': review_auth.get('suspicious_reviews', 0) if review_auth else 'N/A',

                # Validation
                'SEO Spam': 'Yes' if spam_check.get('is_spam') else 'No',
                'Spam Indicators': ', '.join(spam_check.get('indicators_found', [])) or 'None',
                'Data Completeness': f"{completeness_check.get('score', 0) * 100:.0f}%" if completeness_check else 'N/A',
                'Missing Fields': ', '.join(completeness_check.get('missing_fields', [])) or 'None',

                # Overall scores
                'Is Legitimate': 'Yes' if validation.get('is_legitimate') else 'No',
                'Exclusion Reasons': ', '.join(validation.get('exclusion_reasons', [])) or 'None',
                'Warnings': ', '.join(validation.get('warnings', [])) or 'None',
                'Ranking Score': listing.get('ranking_score', 0),

                # Metadata
                'Source Type': listing.get('source_type', 'unknown'),
                'Scraped At': listing.get('scraped_at', 'Unknown'),
            }

            rows.append(row)

        return pd.DataFrame(rows)

    def _create_summary_dataframe(self, valid_listings: List[Dict],
                                  excluded_listings: List[Dict]) -> pd.DataFrame:
        """
        Create summary statistics DataFrame

        Args:
            valid_listings: List of valid listings
            excluded_listings: List of excluded listings

        Returns:
            pandas DataFrame with summary
        """
        total_listings = len(valid_listings) + len(excluded_listings)

        # Calculate statistics
        stats = []

        stats.append({
            'Metric': 'Total Listings Scraped',
            'Value': total_listings
        })

        stats.append({
            'Metric': 'Valid Listings',
            'Value': len(valid_listings)
        })

        stats.append({
            'Metric': 'Excluded Listings',
            'Value': len(excluded_listings)
        })

        stats.append({
            'Metric': 'Validation Pass Rate',
            'Value': f"{len(valid_listings) / total_listings * 100:.1f}%" if total_listings > 0 else '0%'
        })

        # Exclusion reasons breakdown
        if excluded_listings:
            stats.append({
                'Metric': '',
                'Value': ''
            })
            stats.append({
                'Metric': 'EXCLUSION REASONS:',
                'Value': ''
            })

            exclusion_counts = {}
            for listing in excluded_listings:
                reasons = listing.get('validation', {}).get('exclusion_reasons', [])
                for reason in reasons:
                    exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1

            for reason, count in sorted(exclusion_counts.items(),
                                       key=lambda x: x[1], reverse=True):
                stats.append({
                    'Metric': f"  • {reason}",
                    'Value': count
                })

        # Average ranking score for valid listings
        if valid_listings:
            avg_score = sum(
                l.get('ranking_score', 0) for l in valid_listings
            ) / len(valid_listings)

            stats.append({
                'Metric': '',
                'Value': ''
            })
            stats.append({
                'Metric': 'Average Ranking Score',
                'Value': f"{avg_score:.2f}"
            })

            # HTTPS statistics
            https_count = sum(
                1 for l in valid_listings
                if l.get('validation', {}).get('checks', {}).get('https', {}).get('uses_https')
            )
            stats.append({
                'Metric': 'Listings with HTTPS',
                'Value': f"{https_count} ({https_count / len(valid_listings) * 100:.0f}%)"
            })

        return pd.DataFrame(stats)

    def _format_worksheet(self, worksheet):
        """
        Apply formatting to worksheet

        Args:
            worksheet: openpyxl worksheet object
        """
        # Style for header row
        header_fill = PatternFill(start_color='366092',
                                  end_color='366092',
                                  fill_type='solid')
        header_font = Font(bold=True, color='FFFFFF')

        # Format header row
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center',
                                      vertical='center',
                                      wrap_text=True)

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter

            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass

            adjusted_width = min(max_length + 2, 50)  # Cap at 50
            worksheet.column_dimensions[column_letter].width = adjusted_width

        # Freeze header row
        worksheet.freeze_panes = 'A2'

    def export_to_google_sheets(self, valid_listings: List[Dict],
                                excluded_listings: List[Dict],
                                sheet_name: str) -> str:
        """
        Export results to Google Sheets

        Args:
            valid_listings: List of valid listings
            excluded_listings: List of excluded listings
            sheet_name: Name for the Google Sheet

        Returns:
            URL of created Google Sheet
        """
        self.logger.info(f"Exporting to Google Sheets: {sheet_name}")

        try:
            import gspread
            from google.oauth2.service_account import Credentials

            # Define the scope
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]

            # Load credentials
            creds_file = 'google_credentials.json'
            if not os.path.exists(creds_file):
                raise FileNotFoundError(
                    f"Google credentials file not found: {creds_file}\n"
                    "Please follow the setup instructions in README.md"
                )

            creds = Credentials.from_service_account_file(creds_file, scopes=scope)
            client = gspread.authorize(creds)

            # Create a new spreadsheet
            spreadsheet = client.create(sheet_name)

            # Share with your email (optional - update with actual email)
            # spreadsheet.share('your-email@example.com', perm_type='user', role='writer')

            # Prepare data for each sheet
            valid_df = self._prepare_listings_dataframe(valid_listings) if valid_listings else pd.DataFrame()
            excluded_df = self._prepare_listings_dataframe(excluded_listings) if excluded_listings else pd.DataFrame()
            summary_df = self._create_summary_dataframe(valid_listings, excluded_listings)

            # Write to sheets
            if not valid_df.empty:
                worksheet = spreadsheet.get_worksheet(0)
                worksheet.update_title('Valid Listings')
                worksheet.update([valid_df.columns.values.tolist()] + valid_df.values.tolist())

            if not excluded_df.empty:
                worksheet = spreadsheet.add_worksheet(
                    title='Excluded Listings',
                    rows=len(excluded_df) + 1,
                    cols=len(excluded_df.columns)
                )
                worksheet.update([excluded_df.columns.values.tolist()] + excluded_df.values.tolist())

            # Add summary sheet
            if not summary_df.empty:
                worksheet = spreadsheet.add_worksheet(
                    title='Summary',
                    rows=len(summary_df) + 1,
                    cols=len(summary_df.columns)
                )
                worksheet.update([summary_df.columns.values.tolist()] + summary_df.values.tolist())

            url = spreadsheet.url
            self.logger.info(f"Google Sheets export completed: {url}")
            return url

        except ImportError:
            error_msg = (
                "Google Sheets libraries not installed.\n"
                "Install with: pip install gspread google-auth google-auth-oauthlib google-auth-httplib2"
            )
            self.logger.error(error_msg)
            raise ImportError(error_msg)

        except Exception as e:
            self.logger.error(f"Failed to export to Google Sheets: {e}")
            raise

    def export_to_csv(self, listings: List[Dict], filename: str) -> str:
        """
        Export listings to CSV file

        Args:
            listings: List of listing dictionaries
            filename: Output filename

        Returns:
            Path to created file
        """
        self.logger.info(f"Exporting to CSV: {filename}")

        df = self._prepare_listings_dataframe(listings)
        df.to_csv(filename, index=False)

        self.logger.info(f"CSV export completed: {filename}")
        return filename
