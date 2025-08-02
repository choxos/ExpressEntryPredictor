import requests
import pandas as pd
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.utils import timezone
from predictor.models import EconomicIndicator

class Command(BaseCommand):
    help = 'Collect economic indicators from Statistics Canada and other sources'

    def add_arguments(self, parser):
        parser.add_argument(
            '--start-date',
            type=str,
            help='Start date for data collection (YYYY-MM-DD)'
        )
        parser.add_argument(
            '--end-date', 
            type=str,
            help='End date for data collection (YYYY-MM-DD)'
        )

    def handle(self, *args, **options):
        self.stdout.write('Collecting economic indicators...')
        
        # Set date range
        end_date = datetime.strptime(options['end_date'], '%Y-%m-%d').date() if options['end_date'] else timezone.now().date()
        start_date = datetime.strptime(options['start_date'], '%Y-%m-%d').date() if options['start_date'] else end_date - timedelta(days=365)
        
        # Statistics Canada API endpoints
        statcan_tables = {
            'unemployment': '14-10-0287-01',
            'job_vacancy': '14-10-0325-01', 
            'gdp': '36-10-0104-01'
        }
        
        indicators_created = 0
        
        try:
            # Bank of Canada overnight rate
            self.stdout.write('Fetching Bank of Canada rates...')
            bank_data = self._fetch_bank_of_canada_rate(start_date, end_date)
            
            # Statistics Canada data
            for indicator_name, table_id in statcan_tables.items():
                self.stdout.write(f'Fetching {indicator_name} data...')
                statcan_data = self._fetch_statcan_data(table_id, start_date, end_date)
                
                # Process and save data
                for _, row in statcan_data.iterrows():
                    indicator, created = EconomicIndicator.objects.get_or_create(
                        indicator_name=f"{indicator_name}_{row.get('geography', 'canada')}",
                        date=row['date'],
                        defaults={
                            'value': row['value'],
                            'source': f'Statistics Canada - Table {table_id}',
                            'description': f'{indicator_name.replace("_", " ").title()} data'
                        }
                    )
                    if created:
                        indicators_created += 1
            
            # Process Bank of Canada data
            for _, row in bank_data.iterrows():
                indicator, created = EconomicIndicator.objects.get_or_create(
                    indicator_name='bank_overnight_rate',
                    date=row['date'],
                    defaults={
                        'value': row['value'],
                        'source': 'Bank of Canada',
                        'description': 'Overnight interest rate'
                    }
                )
                if created:
                    indicators_created += 1
                    
            self.stdout.write(
                self.style.SUCCESS(f'Successfully collected {indicators_created} economic indicators')
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error collecting economic data: {str(e)}')
            )

    def _fetch_statcan_data(self, table_id, start_date, end_date):
        """Fetch data from Statistics Canada API"""
        # Note: This is a simplified example. The actual Statistics Canada API
        # requires specific formatting and authentication
        
        # For now, return sample data structure
        # In production, implement actual API calls to:
        # https://www.statcan.gc.ca/eng/developers/wds
        
        sample_data = pd.DataFrame({
            'date': pd.date_range(start=start_date, end=end_date, freq='M'),
            'geography': ['Canada'] * pd.date_range(start=start_date, end=end_date, freq='M').shape[0],
            'value': [5.5, 5.3, 5.1, 5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2][:pd.date_range(start=start_date, end=end_date, freq='M').shape[0]]
        })
        
        return sample_data

    def _fetch_bank_of_canada_rate(self, start_date, end_date):
        """Fetch overnight rate from Bank of Canada"""
        # Note: Bank of Canada provides API access
        # https://www.bankofcanada.ca/valet/docs
        
        try:
            # Example API call (requires actual implementation)
            # url = f"https://www.bankofcanada.ca/valet/observations/V39065/json?start_date={start_date}&end_date={end_date}"
            # response = requests.get(url)
            
            # For now, return sample data
            sample_data = pd.DataFrame({
                'date': pd.date_range(start=start_date, end=end_date, freq='D'),
                'value': [5.0] * pd.date_range(start=start_date, end=end_date, freq='D').shape[0]
            })
            
            return sample_data
            
        except Exception as e:
            self.stdout.write(self.style.WARNING(f'Could not fetch Bank of Canada data: {e}'))
            return pd.DataFrame() 