from django.core.management.base import BaseCommand
from predictor.models import ExpressEntryDraw, DrawCategory
from django.utils import timezone
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
import time
import pandas as pd


class Command(BaseCommand):
    help = 'Sync new Express Entry draws from IRCC website and update predictions'

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true', help='Show what would be updated without making changes')
        parser.add_argument('--force', action='store_true', help='Force update even if no new draws detected')
        parser.add_argument('--days-back', type=int, default=14, help='Check for draws in the last N days (default: 14)')

    def handle(self, *args, **options):
        self.stdout.write('🔄 Starting IRCC Express Entry draws synchronization...')
        self.stdout.write('=' * 60)
        
        # Configuration
        self.dry_run = options['dry_run']
        self.force_update = options['force']
        self.days_back = options['days_back']
        
        if self.dry_run:
            self.stdout.write('📋 DRY RUN MODE - No changes will be made')
        
        try:
            # Step 1: Fetch latest draws from IRCC website
            new_draws = self.fetch_ircc_draws()
            
            # Step 2: Compare with existing database
            updated_categories = self.update_database(new_draws)
            
            # Step 3: Regenerate predictions for affected categories
            if updated_categories and not self.dry_run:
                self.regenerate_predictions(updated_categories)
            
            # Step 4: Summary
            self.show_summary(new_draws, updated_categories)
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Synchronization failed: {str(e)}'))
            raise

    def fetch_ircc_draws(self):
        """Fetch latest draws from IRCC website"""
        self.stdout.write('\n🌐 Fetching data from IRCC website...')
        
        url = "https://www.canada.ca/en/immigration-refugees-citizenship/corporate/mandate/policies-operational-instructions-agreements/ministerial-instructions/express-entry-rounds.html"
        
        try:
            # Add headers to mimic a real browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            self.stdout.write(f'✅ Successfully fetched IRCC page ({len(response.content)} bytes)')
            
            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the draws table - it should be the main data table
            draws_table = soup.find('table')
            if not draws_table:
                # Try alternative selectors
                draws_table = soup.find('table', {'class': re.compile(r'table|data', re.I)})
            
            if not draws_table:
                raise ValueError("Could not find draws table on IRCC page")
            
            # Parse table data
            draws_data = self.parse_draws_table(draws_table)
            
            # Filter for recent draws only
            cutoff_date = timezone.now().date() - timedelta(days=self.days_back)
            recent_draws = [
                draw for draw in draws_data 
                if draw['date'] >= cutoff_date
            ]
            
            self.stdout.write(f'📊 Found {len(draws_data)} total draws, {len(recent_draws)} recent draws')
            
            return recent_draws
            
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to fetch IRCC website: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse IRCC data: {e}")

    def parse_draws_table(self, table):
        """Parse the draws table from IRCC website"""
        draws = []
        
        # Find table headers to understand structure
        headers = []
        header_row = table.find('thead')
        if header_row:
            for th in header_row.find_all(['th', 'td']):
                headers.append(th.get_text(strip=True).lower())
        else:
            # Try first row as headers
            first_row = table.find('tr')
            if first_row:
                for th in first_row.find_all(['th', 'td']):
                    headers.append(th.get_text(strip=True).lower())
        
        self.stdout.write(f'📋 Table headers: {headers}')
        
        # Find expected column indices
        round_col = self.find_column_index(headers, ['#', 'round', 'number'])
        date_col = self.find_column_index(headers, ['date'])
        type_col = self.find_column_index(headers, ['round type', 'type', 'category'])
        invitations_col = self.find_column_index(headers, ['invitations issued', 'invitations'])
        crs_col = self.find_column_index(headers, ['crs score', 'lowest', 'score'])
        
        # Parse data rows
        rows = table.find_all('tr')[1:]  # Skip header row
        actual_data_rows = []
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 4:  # Need at least basic data
                continue
            
            # Check if this is actually a data row (not another header)
            cell_texts = [cell.get_text(strip=True) for cell in cells]
            if all(text.lower() in ['#', 'date', 'round type', 'invitations issued', 'crs score', ''] for text in cell_texts):
                continue  # This is a header row, skip it
            
            actual_data_rows.append(row)
        
        # Check if we have actual data
        if not actual_data_rows:
            self.stdout.write('⚠️  No data rows found in table')
            self.stdout.write('🔍 This likely means the IRCC website loads data dynamically via JavaScript')
            self.stdout.write('💡 Consider these alternatives:')
            self.stdout.write('   1. Manual CSV import: Export IRCC data and use load_draw_data command')
            self.stdout.write('   2. API integration: Check if IRCC provides an API endpoint')
            self.stdout.write('   3. Browser automation: Use Selenium to load JavaScript content')
            return draws
        
        for row in actual_data_rows:
            cells = row.find_all(['td', 'th'])
                
            try:
                # Extract data with fallback for missing columns
                round_num = self.extract_cell_data(cells, round_col, int) if round_col is not None else None
                date_str = self.extract_cell_data(cells, date_col, str) if date_col is not None else ""
                round_type = self.extract_cell_data(cells, type_col, str) if type_col is not None else ""
                invitations = self.extract_cell_data(cells, invitations_col, int) if invitations_col is not None else 0
                crs_score = self.extract_cell_data(cells, crs_col, int) if crs_col is not None else 0
                
                # Parse date
                draw_date = self.parse_date(date_str)
                if not draw_date:
                    continue
                
                # Determine category from round type
                category_name = self.determine_category(round_type)
                
                draws.append({
                    'round_number': round_num,
                    'date': draw_date,
                    'category_name': category_name,
                    'round_type': round_type,
                    'invitations_issued': invitations,
                    'lowest_crs_score': crs_score
                })
                
            except Exception as e:
                self.stdout.write(f'⚠️  Skipping row due to parsing error: {e}')
                continue
        
        return draws

    def find_column_index(self, headers, possible_names):
        """Find column index by matching possible header names"""
        for i, header in enumerate(headers):
            for name in possible_names:
                if name in header:
                    return i
        return None

    def extract_cell_data(self, cells, col_index, data_type):
        """Extract and convert data from table cell"""
        if col_index is None or col_index >= len(cells):
            return None
            
        cell_text = cells[col_index].get_text(strip=True)
        
        if data_type == int:
            # Extract numbers, handle commas
            numbers = re.findall(r'\d+', cell_text.replace(',', ''))
            return int(numbers[0]) if numbers else 0
        elif data_type == str:
            return cell_text
        else:
            return cell_text

    def parse_date(self, date_str):
        """Parse date string in various formats"""
        if not date_str:
            return None
            
        # Common date formats from IRCC
        date_formats = [
            '%Y-%m-%d',      # 2025-01-15
            '%B %d, %Y',     # January 15, 2025
            '%b %d, %Y',     # Jan 15, 2025
            '%d-%m-%Y',      # 15-01-2025
            '%m/%d/%Y',      # 01/15/2025
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        
        self.stdout.write(f'⚠️  Could not parse date: {date_str}')
        return None

    def determine_category(self, round_type):
        """Map IRCC round type to our category names"""
        round_type = round_type.lower()
        
        # Category mapping based on IRCC descriptions
        category_mapping = {
            # General draws
            'general': 'No Program Specified',
            'no program specified': 'No Program Specified',
            
            # Program-specific
            'canadian experience class': 'Canadian Experience Class',
            'provincial nominee program': 'Provincial Nominee Program',
            'federal skilled worker': 'Federal Skilled Worker',
            'federal skilled trades': 'Federal Skilled Trades',
            
            # Category-based draws
            'french': 'French language proficiency (Version 1)',
            'healthcare': 'Healthcare occupations (Version 1)',
            'stem': 'STEM occupations (Version 1)',
            'trade': 'Trade occupations (Version 1)',
            'agriculture': 'Agriculture and agri-food occupations (Version 1)',
            'education': 'Education occupations (Version 1)',
        }
        
        # Check for matches
        for key, category in category_mapping.items():
            if key in round_type:
                return category
        
        # Default fallback
        return 'No Program Specified'

    def update_database(self, new_draws):
        """Update database with new draws and return affected categories"""
        self.stdout.write('\n📊 Updating database...')
        
        updated_categories = set()
        created_count = 0
        updated_count = 0
        
        for draw_data in new_draws:
            try:
                # Get or create category
                category, created = DrawCategory.objects.get_or_create(
                    name=draw_data['category_name'],
                    defaults={'is_active': True}
                )
                
                # Check if draw already exists
                existing_draw = ExpressEntryDraw.objects.filter(
                    date=draw_data['date'],
                    category=category
                ).first()
                
                if existing_draw:
                    # Update existing draw if data has changed
                    updated = False
                    if existing_draw.lowest_crs_score != draw_data['lowest_crs_score']:
                        existing_draw.lowest_crs_score = draw_data['lowest_crs_score']
                        updated = True
                    if existing_draw.invitations_issued != draw_data['invitations_issued']:
                        existing_draw.invitations_issued = draw_data['invitations_issued']
                        updated = True
                    
                    if updated and not self.dry_run:
                        existing_draw.save()
                        updated_count += 1
                        updated_categories.add(category)
                        self.stdout.write(f'✏️  Updated: {category.name} - {draw_data["date"]}')
                    elif updated:
                        self.stdout.write(f'📝 Would update: {category.name} - {draw_data["date"]}')
                else:
                    # Create new draw
                    if not self.dry_run:
                        ExpressEntryDraw.objects.create(
                            round_number=draw_data['round_number'],
                            date=draw_data['date'],
                            category=category,
                            invitations_issued=draw_data['invitations_issued'],
                            lowest_crs_score=draw_data['lowest_crs_score']
                        )
                        created_count += 1
                        updated_categories.add(category)
                        self.stdout.write(f'✅ Created: {category.name} - {draw_data["date"]} (CRS: {draw_data["lowest_crs_score"]}, Invitations: {draw_data["invitations_issued"]})')
                    else:
                        self.stdout.write(f'📝 Would create: {category.name} - {draw_data["date"]}')
                        
            except Exception as e:
                self.stdout.write(f'❌ Failed to process draw {draw_data}: {e}')
                continue
        
        if not self.dry_run:
            self.stdout.write(f'\n📈 Database updated: {created_count} created, {updated_count} updated')
        else:
            self.stdout.write(f'\n📋 Would update database: {created_count} creates, {updated_count} updates')
        
        return list(updated_categories)

    def regenerate_predictions(self, updated_categories):
        """Regenerate predictions for categories with new draws"""
        if not updated_categories:
            return
            
        self.stdout.write(f'\n🔮 Regenerating predictions for {len(updated_categories)} categories...')
        
        for category in updated_categories:
            try:
                # Call compute_predictions for this category
                from django.core.management import call_command
                self.stdout.write(f'🎯 Updating predictions for: {category.name}')
                
                call_command('compute_predictions', 
                           category=category.name, 
                           force=True, 
                           verbosity=1)
                
                self.stdout.write(f'✅ Predictions updated for: {category.name}')
                
            except Exception as e:
                self.stdout.write(f'❌ Failed to update predictions for {category.name}: {e}')
                continue

    def show_summary(self, new_draws, updated_categories):
        """Show synchronization summary"""
        self.stdout.write('\n' + '=' * 60)
        self.stdout.write('📊 SYNCHRONIZATION SUMMARY')
        self.stdout.write('=' * 60)
        
        self.stdout.write(f'🌐 IRCC draws fetched: {len(new_draws)}')
        
        if updated_categories:
            self.stdout.write(f'📈 Categories updated: {len(updated_categories)}')
            for category in updated_categories:
                pooled_draws, ircc_category, num_pooled = category.get_pooled_data()
                self.stdout.write(f'   ├─ {category.name}: {pooled_draws.count()} total draws (pooled)')
        else:
            self.stdout.write('✅ No new draws found - database is up to date')
        
        # Next check recommendation
        next_check = timezone.now() + timedelta(days=7)
        self.stdout.write(f'⏰ Next recommended check: {next_check.strftime("%Y-%m-%d")}')
        
        if not self.dry_run and updated_categories:
            self.stdout.write('\n🎉 Synchronization completed successfully!')
            self.stdout.write('💡 Tip: Set up a weekly cron job to automate this process')
        elif self.dry_run:
            self.stdout.write('\n📋 Dry run completed - use --force to apply changes')
        else:
            self.stdout.write('\n✅ No updates needed - system is current') 