import csv
import pandas as pd
from datetime import datetime, date
from django.core.management.base import BaseCommand
from django.db import transaction
from predictor.models import DrawCategory, ExpressEntryDraw
from django.db import models


class Command(BaseCommand):
    help = 'Load Express Entry draw data from CSV file'

    def add_arguments(self, parser):
        parser.add_argument(
            '--file', 
            type=str, 
            default='data/draw_data.csv',
            help='Path to the CSV file containing draw data'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing data before loading new data'
        )

    def handle(self, *args, **options):
        file_path = options['file']
        clear_data = options['clear']

        if clear_data:
            self.stdout.write(
                self.style.WARNING('Clearing existing draw data...')
            )
            ExpressEntryDraw.objects.all().delete()
            DrawCategory.objects.all().delete()

        self.stdout.write(f'Loading data from {file_path}...')

        try:
            # Read CSV data
            df = pd.read_csv(file_path)
            
            # Clean and process the data
            df['invitations_issued'] = df['invitations_issued'].str.replace(',', '').astype(int)
            df['date'] = pd.to_datetime(df['date'], format='%d-%b-%y').dt.date
            
            # Create categories and load data
            categories_created = set()
            draws_created = 0
            
            with transaction.atomic():
                for _, row in df.iterrows():
                    # Create or get category
                    category_name = row['type']
                    if category_name not in categories_created:
                        category_code = self._generate_category_code(category_name)
                        
                        # Try to get existing category first
                        try:
                            category = DrawCategory.objects.get(name=category_name)
                        except DrawCategory.DoesNotExist:
                            # Generate unique code if needed
                            base_code = category_code
                            counter = 1
                            while DrawCategory.objects.filter(code=category_code).exists():
                                category_code = f"{base_code}_{counter}"
                                counter += 1
                            
                            category, created = DrawCategory.objects.get_or_create(
                                name=category_name,
                                defaults={'code': category_code}
                            )
                            if created:
                                self.stdout.write(f'Created category: {category_name} ({category_code})')
                        
                        categories_created.add(category_name)
                    else:
                        category = DrawCategory.objects.get(name=category_name)

                    # Calculate days since last draw for this category
                    last_draw = ExpressEntryDraw.objects.filter(
                        category=category
                    ).order_by('-date').first()
                    
                    days_since_last = None
                    if last_draw:
                        days_since_last = (row['date'] - last_draw.date).days

                    # Handle special round numbers like '91a', '91b'
                    round_str = str(row['round'])
                    try:
                        round_number = int(round_str)
                    except ValueError:
                        # Extract numeric part from strings like '91a', '91b'
                        import re
                        match = re.search(r'\d+', round_str)
                        if match:
                            base_number = int(match.group())
                            # Add a small decimal to make it unique (91.1, 91.2, etc.)
                            suffix = ord(round_str[-1]) - ord('a') + 1 if round_str[-1].isalpha() else 0
                            round_number = base_number * 100 + suffix  # 9101, 9102, etc.
                        else:
                            continue  # Skip invalid entries

                    # Create draw record
                    draw, created = ExpressEntryDraw.objects.get_or_create(
                        round_number=round_number,
                        defaults={
                            'date': row['date'],
                            'category': category,
                            'invitations_issued': row['invitations_issued'],
                            'lowest_crs_score': row['lowest_crs_score'],
                            'url': row.get('url', ''),
                            'days_since_last_draw': days_since_last,
                        }
                    )
                    
                    if created:
                        draws_created += 1

            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully loaded {draws_created} draws and '
                    f'{len(categories_created)} categories'
                )
            )

            # Display summary statistics
            self._display_summary()

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error loading data: {str(e)}')
            )

    def _generate_category_code(self, category_name):
        """Generate a short code for category"""
        # Map common categories to codes
        category_codes = {
            'Provincial Nominee Program': 'PNP',
            'Canadian Experience Class': 'CEC',
            'Federal Skilled Worker': 'FSW',
            'Federal Skilled Trades': 'FST',
            'No Program Specified': 'GENERAL',
            'General': 'GENERAL',
        }
        
        if category_name in category_codes:
            return category_codes[category_name]
        
        # For occupation-specific categories, create codes
        if 'Healthcare' in category_name:
            return 'HEALTH'
        elif 'STEM' in category_name:
            return 'STEM'
        elif 'French' in category_name:
            return 'FRENCH'
        elif 'Transport' in category_name:
            return 'TRANSPORT'
        elif 'Trade' in category_name:
            return 'TRADE'
        elif 'Agriculture' in category_name:
            return 'AGRI'
        elif 'Education' in category_name:
            return 'EDU'
        else:
            # Generate code from first letters
            words = category_name.split()
            return ''.join([w[0].upper() for w in words[:3]])

    def _display_summary(self):
        """Display summary statistics of loaded data"""
        total_draws = ExpressEntryDraw.objects.count()
        total_categories = DrawCategory.objects.count()
        
        self.stdout.write('\n--- DATA SUMMARY ---')
        self.stdout.write(f'Total draws: {total_draws}')
        self.stdout.write(f'Total categories: {total_categories}')
        
        # Latest and earliest draws
        latest_draw = ExpressEntryDraw.objects.order_by('-date').first()
        earliest_draw = ExpressEntryDraw.objects.order_by('date').first()
        
        if latest_draw and earliest_draw:
            self.stdout.write(f'Date range: {earliest_draw.date} to {latest_draw.date}')
        
        # Category breakdown
        self.stdout.write('\nDraws by category:')
        for category in DrawCategory.objects.all():
            count = ExpressEntryDraw.objects.filter(category=category).count()
            avg_score = ExpressEntryDraw.objects.filter(category=category).aggregate(
                avg_score=models.Avg('lowest_crs_score')
            )['avg_score']
            avg_score = round(avg_score, 1) if avg_score else 0
            self.stdout.write(f'  {category.name}: {count} draws (avg CRS: {avg_score})') 