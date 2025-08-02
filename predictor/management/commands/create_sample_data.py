import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db import transaction
from predictor.models import EconomicIndicator
import random

class Command(BaseCommand):
    help = 'Create sample data for testing the prediction models'

    def add_arguments(self, parser):
        parser.add_argument(
            '--months',
            type=int,
            default=12,
            help='Number of months of sample data to create (default: 12)'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing economic indicator data before creating samples'
        )

    def handle(self, *args, **options):
        months = options['months']
        clear_data = options['clear']

        if clear_data:
            self.stdout.write(
                self.style.WARNING('Clearing existing economic indicator data...')
            )
            EconomicIndicator.objects.all().delete()

        self.stdout.write(f'Creating {months} months of sample economic data...')
        
        # Generate sample economic indicators
        self._create_economic_indicators(months)
        
        # Create sample CSV files for manual testing
        self._create_sample_csv_files(months)
        
        self.stdout.write(
            self.style.SUCCESS('Sample data creation completed!')
        )

    def _create_economic_indicators(self, months):
        """Create sample economic indicators in the database"""
        
        indicators_data = {
            'unemployment_rate_canada': {'base': 5.2, 'volatility': 0.3, 'trend': -0.02},
            'unemployment_rate_ontario': {'base': 5.0, 'volatility': 0.4, 'trend': -0.01},
            'unemployment_rate_bc': {'base': 4.8, 'volatility': 0.35, 'trend': -0.015},
            'unemployment_rate_alberta': {'base': 5.5, 'volatility': 0.5, 'trend': -0.03},
            'job_vacancy_rate': {'base': 5.3, 'volatility': 0.2, 'trend': 0.01},
            'gdp_growth_rate': {'base': 2.1, 'volatility': 0.4, 'trend': 0.005},
            'bank_overnight_rate': {'base': 5.0, 'volatility': 0.25, 'trend': -0.01},
            'cpi_inflation_rate': {'base': 3.4, 'volatility': 0.3, 'trend': -0.02}
        }
        
        start_date = timezone.now().date().replace(day=1) - timedelta(days=30*months)
        
        indicators_created = 0
        
        with transaction.atomic():
            for month in range(months):
                current_date = start_date + timedelta(days=30*month)
                
                for indicator_name, params in indicators_data.items():
                    # Generate realistic time series data with trend and noise
                    trend_value = params['base'] + (params['trend'] * month)
                    noise = np.random.normal(0, params['volatility'])
                    value = max(0, trend_value + noise)  # Ensure non-negative values
                    
                    # Add seasonal patterns for some indicators
                    if 'unemployment' in indicator_name:
                        # Higher unemployment in winter months
                        seasonal_factor = 0.2 * np.sin(2 * np.pi * (current_date.month - 1) / 12 + np.pi)
                        value += seasonal_factor
                    
                    indicator, created = EconomicIndicator.objects.get_or_create(
                        indicator_name=indicator_name,
                        date=current_date,
                        defaults={
                            'value': round(value, 2),
                            'source': 'Sample Data Generator',
                            'description': f'Sample {indicator_name.replace("_", " ").title()} data for testing'
                        }
                    )
                    
                    if created:
                        indicators_created += 1
        
        self.stdout.write(f'Created {indicators_created} economic indicator records')

    def _create_sample_csv_files(self, months):
        """Create sample CSV files for manual testing"""
        
        start_date = date.today() - timedelta(days=30*months)
        
        # Economic Indicators CSV
        economic_data = []
        for month in range(months):
            current_date = start_date + timedelta(days=30*month)
            economic_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'unemployment_rate_ca': round(5.2 + random.uniform(-0.5, 0.5), 1),
                'unemployment_rate_on': round(5.0 + random.uniform(-0.4, 0.4), 1),
                'unemployment_rate_bc': round(4.8 + random.uniform(-0.3, 0.3), 1),
                'unemployment_rate_ab': round(5.5 + random.uniform(-0.6, 0.6), 1),
                'unemployment_rate_sk': round(4.2 + random.uniform(-0.3, 0.3), 1),
                'unemployment_rate_mb': round(4.9 + random.uniform(-0.4, 0.4), 1),
                'job_vacancy_rate': round(5.3 + random.uniform(-0.2, 0.2), 1),
                'gdp_growth_rate': round(2.1 + random.uniform(-0.4, 0.4), 1),
                'bank_overnight_rate': round(5.0 + random.uniform(-0.25, 0.25), 2),
                'cpi_inflation_rate': round(3.4 + random.uniform(-0.3, 0.3), 1)
            })
        
        # Save economic indicators CSV
        economic_df = pd.DataFrame(economic_data)
        economic_df.to_csv('data/sample_economic_indicators.csv', index=False)
        self.stdout.write('Created data/sample_economic_indicators.csv')
        
        # Pool Data CSV (bi-weekly)
        pool_data = []
        for week in range(0, months*4, 2):  # Bi-weekly data
            current_date = start_date + timedelta(weeks=week)
            base_candidates = 270000
            pool_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'total_candidates': base_candidates + random.randint(-10000, 10000),
                'candidates_600_plus': random.randint(4000, 6000),
                'candidates_500_599': random.randint(10000, 15000),
                'candidates_450_499': random.randint(30000, 40000),
                'candidates_400_449': random.randint(70000, 90000),
                'candidates_below_400': random.randint(120000, 150000),
                'new_registrations_weekly': random.randint(2000, 3500),
                'avg_pool_crs': random.randint(415, 435)
            })
        
        pool_df = pd.DataFrame(pool_data)
        pool_df.to_csv('data/sample_pool_data.csv', index=False)
        self.stdout.write('Created data/sample_pool_data.csv')
        
        # PNP Data CSV (weekly)
        pnp_data = []
        for week in range(months*4):  # Weekly data
            current_date = start_date + timedelta(weeks=week)
            pnp_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'ontario_invites': random.randint(200, 400),
                'bc_invites': random.randint(150, 250),
                'alberta_invites': random.randint(100, 200),
                'saskatchewan_invites': random.randint(80, 150),
                'manitoba_invites': random.randint(40, 80),
                'nova_scotia_invites': random.randint(20, 50),
                'new_brunswick_invites': random.randint(15, 35),
                'pei_invites': random.randint(10, 25),
                'newfoundland_invites': random.randint(5, 20),
                'yukon_invites': random.randint(1, 8),
                'nwt_invites': random.randint(1, 5),
                'total_pnp_weekly': 0  # Will be calculated
            })
            # Calculate total
            pnp_data[-1]['total_pnp_weekly'] = sum([
                pnp_data[-1][col] for col in pnp_data[-1].keys() 
                if col.endswith('_invites') and col != 'total_pnp_weekly'
            ])
        
        pnp_df = pd.DataFrame(pnp_data)
        pnp_df.to_csv('data/sample_pnp_data.csv', index=False)
        self.stdout.write('Created data/sample_pnp_data.csv')
        
        # Calendar Events CSV (daily for current year)
        calendar_data = []
        current_year_start = date(date.today().year, 1, 1)
        for day in range(365):
            current_date = current_year_start + timedelta(days=day)
            
            # Federal holidays (simplified)
            federal_holidays = [
                date(current_date.year, 1, 1),  # New Year
                date(current_date.year, 7, 1),  # Canada Day
                date(current_date.year, 12, 25), # Christmas
            ]
            
            is_federal_holiday = current_date in federal_holidays
            is_weekend = current_date.weekday() >= 5
            
            calendar_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'is_federal_holiday': 1 if is_federal_holiday else 0,
                'is_provincial_holiday': 1 if is_federal_holiday or random.random() < 0.02 else 0,
                'is_long_weekend': 1 if is_weekend and (is_federal_holiday or random.random() < 0.1) else 0,
                'days_to_next_holiday': random.randint(1, 30),
                'is_system_maintenance': 1 if random.random() < 0.01 else 0,
                'parliament_sitting': 1 if current_date.weekday() < 5 and not is_federal_holiday else 0,
                'minister_announcement': 1 if random.random() < 0.05 else 0,
                'policy_change': 1 if random.random() < 0.02 else 0
            })
        
        calendar_df = pd.DataFrame(calendar_data)
        calendar_df.to_csv('data/sample_calendar_events.csv', index=False)
        self.stdout.write('Created data/sample_calendar_events.csv')
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Created sample CSV files with {months} months of data in data/ directory'
            )
        ) 