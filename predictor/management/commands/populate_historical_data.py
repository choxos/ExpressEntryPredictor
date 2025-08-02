from django.core.management.base import BaseCommand
from datetime import datetime, date
import pandas as pd
import numpy as np
from predictor.models import (
    EconomicIndicator, PolicyAnnouncement, GovernmentContext, 
    PoolComposition, PNPActivity, DrawCategory
)


class Command(BaseCommand):
    help = 'Populate historical data for enhanced prediction factors (2015-2024)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing data before populating'
        )
        parser.add_argument(
            '--start-year',
            type=int,
            default=2015,
            help='Start year for data population'
        )
        parser.add_argument(
            '--end-year',
            type=int,
            default=2024,
            help='End year for data population'
        )

    def handle(self, *args, **options):
        if options['clear']:
            self.stdout.write(self.style.WARNING('Clearing existing historical data...'))
            EconomicIndicator.objects.all().delete()
            PolicyAnnouncement.objects.all().delete()
            GovernmentContext.objects.all().delete()
            PoolComposition.objects.all().delete()
            PNPActivity.objects.all().delete()
            self.stdout.write('✅ Existing data cleared')

        start_year = options['start_year']
        end_year = options['end_year']
        
        self.stdout.write(f'🚀 Populating historical data from {start_year} to {end_year}...')
        
        # Population functions
        self.populate_government_context(start_year, end_year)
        self.populate_economic_indicators(start_year, end_year)
        self.populate_policy_announcements(start_year, end_year)
        self.populate_pool_composition(start_year, end_year)
        self.populate_pnp_activity(start_year, end_year)
        
        self.stdout.write(self.style.SUCCESS('🎉 Historical data population completed!'))
        
        # Summary statistics
        self.print_summary()

    def populate_government_context(self, start_year, end_year):
        """Populate government context data based on Canadian political history"""
        self.stdout.write('📊 Populating government context...')
        
        government_periods = [
            {
                'start_date': date(2015, 11, 4),
                'end_date': date(2021, 9, 19),
                'government_type': 'LIBERAL_MAJORITY',
                'prime_minister': 'Justin Trudeau',
                'immigration_minister': 'John McCallum',  # 2015-2017
                'economic_immigration_priority': 8,
                'humanitarian_priority': 7,
                'francophone_priority': 6,
            },
            {
                'start_date': date(2021, 9, 20),
                'end_date': None,  # Current government
                'government_type': 'LIBERAL_MINORITY',
                'prime_minister': 'Justin Trudeau',
                'immigration_minister': 'Marc Miller',  # Current
                'economic_immigration_priority': 8,
                'humanitarian_priority': 6,
                'francophone_priority': 7,
            }
        ]
        
        created_count = 0
        for gov_data in government_periods:
            gov, created = GovernmentContext.objects.get_or_create(
                start_date=gov_data['start_date'],
                defaults=gov_data
            )
            if created:
                created_count += 1
                
        self.stdout.write(f'✅ Created {created_count} government context records')

    def populate_economic_indicators(self, start_year, end_year):
        """Populate economic indicators with realistic Canadian data trends"""
        self.stdout.write('📈 Populating economic indicators...')
        
        created_count = 0
        
        # Generate monthly data
        dates = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='MS')
        
        for i, current_date in enumerate(dates):
            # Realistic unemployment rate trends
            if current_date.year <= 2019:
                # Pre-COVID: Generally declining unemployment
                base_unemployment = 7.0 - (current_date.year - 2015) * 0.3
                seasonal_adj = 0.5 * np.sin(2 * np.pi * current_date.month / 12)
                unemployment_rate = max(5.5, base_unemployment + seasonal_adj + np.random.normal(0, 0.3))
            elif current_date.year == 2020:
                # COVID impact: Sharp spike in unemployment
                if current_date.month <= 3:
                    unemployment_rate = 6.0 + np.random.normal(0, 0.2)
                elif current_date.month <= 6:
                    unemployment_rate = 13.0 + np.random.normal(0, 1.0)  # Peak COVID
                else:
                    unemployment_rate = 10.0 - (current_date.month - 6) * 0.5 + np.random.normal(0, 0.5)
            elif current_date.year == 2021:
                # COVID recovery
                unemployment_rate = 8.0 - current_date.month * 0.3 + np.random.normal(0, 0.4)
            else:
                # Post-COVID: Gradual normalization
                unemployment_rate = max(5.0, 6.5 - (current_date.year - 2021) * 0.3 + np.random.normal(0, 0.3))
            
            # Job vacancy rate (inverse relationship with unemployment)
            if current_date.year >= 2019:  # Data more available from 2019
                job_vacancy_rate = max(2.0, 5.5 - (unemployment_rate - 6.0) * 0.4 + np.random.normal(0, 0.2))
            else:
                job_vacancy_rate = max(2.0, 4.0 + np.random.normal(0, 0.3))
            
            # GDP growth rate
            if current_date.year <= 2019:
                # Pre-COVID: Steady growth
                gdp_growth = 2.0 + np.random.normal(0, 0.5)
            elif current_date.year == 2020:
                # COVID recession
                if current_date.month <= 6:
                    gdp_growth = -8.0 + np.random.normal(0, 2.0)
                else:
                    gdp_growth = -2.0 + np.random.normal(0, 1.0)
            elif current_date.year == 2021:
                # Recovery
                gdp_growth = 4.0 + np.random.normal(0, 1.0)
            else:
                # Post-COVID normalization
                gdp_growth = 2.5 + np.random.normal(0, 0.7)
            
            # Immigration targets (historical progression)
            target_progression = {
                2015: 260000, 2016: 300000, 2017: 300000, 2018: 310000, 2019: 330800,
                2020: 341000, 2021: 401000, 2022: 431645, 2023: 465000, 2024: 485000
            }
            immigration_target = target_progression.get(current_date.year, 485000)
            
            indicator, created = EconomicIndicator.objects.get_or_create(
                date=current_date.date(),
                defaults={
                    'unemployment_rate': round(unemployment_rate, 1),
                    'job_vacancy_rate': round(job_vacancy_rate, 1),
                    'gdp_growth': round(gdp_growth, 1),
                    'immigration_target': immigration_target,
                }
            )
            
            if created:
                created_count += 1
                
        self.stdout.write(f'✅ Created {created_count} economic indicator records')

    def populate_policy_announcements(self, start_year, end_year):
        """Populate major immigration policy announcements"""
        self.stdout.write('📋 Populating policy announcements...')
        
        major_announcements = [
            {
                'date': date(2015, 11, 12),
                'announcement_type': 'MINISTER_MANDATE',
                'title': 'Minister McCallum Mandate Letter',
                'description': 'Increase immigration levels and improve Express Entry system',
                'expected_impact': 'HIGH',
                'target_change': 25000,
            },
            {
                'date': date(2017, 11, 1),
                'announcement_type': 'IMMIGRATION_TARGET',
                'title': '2018-2020 Immigration Levels Plan',
                'description': 'Multi-year plan to increase immigration to 350,000 by 2021',
                'expected_impact': 'HIGH',
                'target_change': 40000,
            },
            {
                'date': date(2017, 11, 15),
                'announcement_type': 'PROGRAM_CHANGE',
                'title': 'Express Entry CRS Updates',
                'description': 'Additional points for French language proficiency',
                'expected_impact': 'MEDIUM',
            },
            {
                'date': date(2020, 3, 18),
                'announcement_type': 'SPECIAL_MEASURE',
                'title': 'COVID-19 Immigration Measures',
                'description': 'Temporary suspension of draws and special measures',
                'expected_impact': 'HIGH',
                'target_change': -50000,
            },
            {
                'date': date(2020, 10, 30),
                'announcement_type': 'IMMIGRATION_TARGET',
                'title': '2021-2023 Immigration Levels Plan',
                'description': 'Ambitious targets: 401,000 in 2021, increasing to 421,000',
                'expected_impact': 'HIGH',
                'target_change': 80000,
            },
            {
                'date': date(2021, 2, 13),
                'announcement_type': 'SPECIAL_MEASURE',
                'title': 'CEC-only draws introduction',
                'description': 'Targeted draws for Canadian Experience Class',
                'expected_impact': 'HIGH',
            },
            {
                'date': date(2021, 5, 6),
                'announcement_type': 'SPECIAL_MEASURE',
                'title': 'French Language Priority Draws',
                'description': 'Special draws for French-speaking candidates',
                'expected_impact': 'MEDIUM',
            },
            {
                'date': date(2022, 2, 14),
                'announcement_type': 'IMMIGRATION_TARGET',
                'title': '2022-2024 Immigration Levels Plan',
                'description': 'Targets: 431,645 (2022) to 451,000 (2024)',
                'expected_impact': 'HIGH',
                'target_change': 30000,
            },
            {
                'date': date(2023, 11, 1),
                'announcement_type': 'IMMIGRATION_TARGET',
                'title': '2024-2026 Immigration Levels Plan',
                'description': 'Increased targets: 485,000 by 2024, 500,000 by 2025',
                'expected_impact': 'HIGH',
                'target_change': 35000,
            },
            {
                'date': date(2024, 1, 31),
                'announcement_type': 'PROGRAM_CHANGE',
                'title': 'Category-Based Express Entry Selection',
                'description': 'Introduction of category-based selection system',
                'expected_impact': 'HIGH',
            }
        ]
        
        created_count = 0
        for announcement_data in major_announcements:
            if start_year <= announcement_data['date'].year <= end_year:
                announcement, created = PolicyAnnouncement.objects.get_or_create(
                    date=announcement_data['date'],
                    title=announcement_data['title'],
                    defaults=announcement_data
                )
                if created:
                    created_count += 1
                    
        self.stdout.write(f'✅ Created {created_count} policy announcement records')

    def populate_pool_composition(self, start_year, end_year):
        """Populate Express Entry pool composition data"""
        self.stdout.write('👥 Populating pool composition...')
        
        created_count = 0
        
        # Generate bi-weekly data (every 2 weeks)
        start_date = date(start_year, 1, 15)
        end_date = date(end_year, 12, 31)
        
        current_date = start_date
        while current_date <= end_date:
            # Pool size evolution over time
            if current_date.year <= 2017:
                base_pool = 100000 + (current_date.year - 2015) * 15000
            elif current_date.year <= 2019:
                base_pool = 130000 + (current_date.year - 2017) * 10000
            elif current_date.year == 2020:
                # COVID impact: Pool growth due to reduced draws
                base_pool = 150000 + current_date.month * 2000
            elif current_date.year >= 2021:
                # Post-COVID: Large pool sizes
                base_pool = 180000 + (current_date.year - 2021) * 5000
            
            # Seasonal variation
            seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * current_date.month / 12)
            total_candidates = int(base_pool * seasonal_factor + np.random.normal(0, 5000))
            total_candidates = max(50000, total_candidates)
            
            # CRS score distribution (realistic proportions)
            candidates_600_plus = int(total_candidates * (0.02 + np.random.normal(0, 0.005)))
            candidates_500_599 = int(total_candidates * (0.15 + np.random.normal(0, 0.02)))
            candidates_450_499 = int(total_candidates * (0.35 + np.random.normal(0, 0.03)))
            candidates_400_449 = int(total_candidates * (0.30 + np.random.normal(0, 0.02)))
            candidates_below_400 = total_candidates - (candidates_600_plus + candidates_500_599 + 
                                                     candidates_450_499+ candidates_400_449)
            
            # Average CRS calculation
            avg_crs = (
                candidates_600_plus * 650 + 
                candidates_500_599 * 550 + 
                candidates_450_499 * 475 + 
                candidates_400_449 * 425 + 
                candidates_below_400 * 350
            ) / total_candidates
            
            # New registrations (weekly average)
            new_registrations = int(1000 + np.random.normal(0, 200))
            
            pool, created = PoolComposition.objects.get_or_create(
                date=current_date,
                defaults={
                    'total_candidates': total_candidates,
                    'candidates_600_plus': max(0, candidates_600_plus),
                    'candidates_500_599': max(0, candidates_500_599),
                    'candidates_450_499': max(0, candidates_450_499),
                    'candidates_400_449': max(0, candidates_400_449),
                    'candidates_below_400': max(0, candidates_below_400),
                    'new_registrations': new_registrations,
                    'average_crs': round(avg_crs, 1),
                    'median_crs': round(avg_crs - 20 + np.random.normal(0, 10), 1),
                }
            )
            
            if created:
                created_count += 1
            
            # Move to next bi-weekly period
            current_date = pd.to_datetime(current_date) + pd.Timedelta(days=14)
            current_date = current_date.date()
                
        self.stdout.write(f'✅ Created {created_count} pool composition records')

    def populate_pnp_activity(self, start_year, end_year):
        """Populate Provincial Nominee Program activity data"""
        self.stdout.write('🏛️ Populating PNP activity...')
        
        # Major PNP programs with realistic volumes
        provinces_data = {
            'ON': {'base_monthly': 800, 'growth_rate': 50, 'volatility': 100},
            'BC': {'base_monthly': 400, 'growth_rate': 30, 'volatility': 80},
            'AB': {'base_monthly': 300, 'growth_rate': 20, 'volatility': 100},
            'SK': {'base_monthly': 200, 'growth_rate': 15, 'volatility': 50},
            'MB': {'base_monthly': 150, 'growth_rate': 10, 'volatility': 40},
            'NS': {'base_monthly': 100, 'growth_rate': 8, 'volatility': 30},
            'NB': {'base_monthly': 80, 'growth_rate': 5, 'volatility': 25},
        }
        
        created_count = 0
        
        # Generate monthly data for each province
        dates = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='MS')
        
        for current_date in dates:
            for province_code, data in provinces_data.items():
                # Calculate realistic invitation numbers
                years_since_start = current_date.year - start_year
                base_invitations = data['base_monthly'] + (years_since_start * data['growth_rate'])
                
                # COVID impact (reduced PNP activity in 2020)
                if current_date.year == 2020:
                    covid_factor = 0.6 if current_date.month <= 8 else 0.8
                    base_invitations *= covid_factor
                
                # Seasonal variation (less activity in summer/holidays)
                seasonal_factor = 1.0
                if current_date.month in [7, 8, 12, 1]:
                    seasonal_factor = 0.8
                elif current_date.month in [3, 4, 9, 10]:
                    seasonal_factor = 1.2
                
                invitations = int(base_invitations * seasonal_factor + 
                                np.random.normal(0, data['volatility']))
                invitations = max(0, invitations)
                
                # Estimate minimum scores (inverse relationship with volume)
                if invitations > 0:
                    base_score = 65 if province_code in ['ON', 'BC'] else 60
                    score_variation = max(0, 20 - (invitations / data['base_monthly']) * 10)
                    minimum_score = int(base_score + score_variation + np.random.normal(0, 5))
                else:
                    minimum_score = None
                
                # Provincial unemployment (rough estimates)
                provincial_unemployment = {
                    'ON': 6.5, 'BC': 6.0, 'AB': 7.5, 'SK': 6.0, 
                    'MB': 6.5, 'NS': 8.0, 'NB': 9.0
                }.get(province_code, 7.0)
                
                # Add some variation
                provincial_unemployment += np.random.normal(0, 0.5)
                
                # Key sectors by province
                key_sectors = {
                    'ON': ['Technology', 'Healthcare', 'Finance'],
                    'BC': ['Technology', 'Healthcare', 'Natural Resources'],
                    'AB': ['Energy', 'Healthcare', 'Agriculture'],
                    'SK': ['Agriculture', 'Mining', 'Healthcare'],
                    'MB': ['Agriculture', 'Manufacturing', 'Healthcare'],
                    'NS': ['Healthcare', 'Information Technology', 'Ocean Technologies'],
                    'NB': ['Healthcare', 'Information Technology', 'Agriculture'],
                }
                
                pnp, created = PNPActivity.objects.get_or_create(
                    date=current_date.date(),
                    province=province_code,
                    defaults={
                        'invitations_issued': invitations,
                        'minimum_score': minimum_score,
                        'program_stream': f'{province_code} General',
                        'provincial_unemployment': round(provincial_unemployment, 1),
                        'key_sectors': key_sectors.get(province_code, ['General']),
                    }
                )
                
                if created:
                    created_count += 1
                    
        self.stdout.write(f'✅ Created {created_count} PNP activity records')

    def print_summary(self):
        """Print summary statistics of populated data"""
        self.stdout.write('\n📊 DATA POPULATION SUMMARY:')
        self.stdout.write(f'Government Contexts: {GovernmentContext.objects.count()}')
        self.stdout.write(f'Economic Indicators: {EconomicIndicator.objects.count()}')
        self.stdout.write(f'Policy Announcements: {PolicyAnnouncement.objects.count()}')
        self.stdout.write(f'Pool Compositions: {PoolComposition.objects.count()}')
        self.stdout.write(f'PNP Activities: {PNPActivity.objects.count()}')
        
        # Date ranges
        if EconomicIndicator.objects.exists():
            first_econ = EconomicIndicator.objects.order_by('date').first()
            last_econ = EconomicIndicator.objects.order_by('-date').first()
            self.stdout.write(f'Economic Data Range: {first_econ.date} to {last_econ.date}')
        
        if PoolComposition.objects.exists():
            first_pool = PoolComposition.objects.order_by('date').first()
            last_pool = PoolComposition.objects.order_by('-date').first()
            self.stdout.write(f'Pool Data Range: {first_pool.date} to {last_pool.date}')
            
        self.stdout.write('\n🎯 Ready for enhanced prediction testing!') 