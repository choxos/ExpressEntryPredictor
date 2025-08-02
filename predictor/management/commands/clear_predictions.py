from django.core.management.base import BaseCommand
from predictor.models import PreComputedPrediction, PredictionCache, DrawCategory
from django.utils import timezone


class Command(BaseCommand):
    help = 'Clear prediction data and caches with various options'

    def add_arguments(self, parser):
        parser.add_argument(
            '--all',
            action='store_true',
            help='Clear all predictions and caches'
        )
        parser.add_argument(
            '--predictions',
            action='store_true',
            help='Clear only pre-computed predictions'
        )
        parser.add_argument(
            '--cache',
            action='store_true',
            help='Clear only prediction caches'
        )
        parser.add_argument(
            '--inactive',
            action='store_true',
            help='Clear only inactive predictions (keep active ones)'
        )
        parser.add_argument(
            '--category',
            type=str,
            help='Clear predictions for specific category (by name)'
        )
        parser.add_argument(
            '--older-than',
            type=int,
            help='Clear predictions older than N days'
        )
        parser.add_argument(
            '--confirm',
            action='store_true',
            help='Skip confirmation prompt'
        )

    def handle(self, *args, **options):
        if not any([options['all'], options['predictions'], options['cache'], 
                   options['inactive'], options['category'], options['older_than']]):
            self.stdout.write(
                self.style.ERROR('Please specify what to clear. Use --help for options.')
            )
            return

        # Show current counts
        self.show_current_counts()

        # Confirmation unless --confirm is used
        if not options['confirm']:
            confirm = input('\nAre you sure you want to proceed? (yes/no): ')
            if confirm.lower() != 'yes':
                self.stdout.write('❌ Operation cancelled')
                return

        # Execute cleanup operations
        if options['all']:
            self.clear_all()
        elif options['predictions']:
            self.clear_predictions()
        elif options['cache']:
            self.clear_cache()
        elif options['inactive']:
            self.clear_inactive_predictions()
        elif options['category']:
            self.clear_category_predictions(options['category'])
        elif options['older_than']:
            self.clear_old_predictions(options['older_than'])

        # Show final counts
        self.stdout.write('\n' + '='*50)
        self.stdout.write('📊 FINAL COUNTS:')
        self.show_current_counts()

    def show_current_counts(self):
        """Display current prediction and cache counts"""
        total_predictions = PreComputedPrediction.objects.count()
        active_predictions = PreComputedPrediction.objects.filter(is_active=True).count()
        inactive_predictions = total_predictions - active_predictions
        cache_entries = PredictionCache.objects.count()

        self.stdout.write(f'📊 Pre-computed Predictions: {total_predictions}')
        self.stdout.write(f'   ├─ Active: {active_predictions}')
        self.stdout.write(f'   └─ Inactive: {inactive_predictions}')
        self.stdout.write(f'🗂️  Cache Entries: {cache_entries}')

        # Show predictions by category
        categories_with_predictions = PreComputedPrediction.objects.values('category__name').distinct()
        if categories_with_predictions:
            self.stdout.write('📂 Predictions by Category:')
            for cat in categories_with_predictions:
                cat_name = cat['category__name']
                cat_count = PreComputedPrediction.objects.filter(category__name=cat_name).count()
                self.stdout.write(f'   ├─ {cat_name}: {cat_count}')

    def clear_all(self):
        """Clear all predictions and caches"""
        self.stdout.write('🗑️  Clearing ALL prediction data...')
        
        pred_count = PreComputedPrediction.objects.count()
        cache_count = PredictionCache.objects.count()
        
        PreComputedPrediction.objects.all().delete()
        PredictionCache.objects.all().delete()
        
        self.stdout.write(self.style.SUCCESS(f'✅ Deleted {pred_count} predictions'))
        self.stdout.write(self.style.SUCCESS(f'✅ Deleted {cache_count} cache entries'))

    def clear_predictions(self):
        """Clear only pre-computed predictions"""
        self.stdout.write('🗑️  Clearing pre-computed predictions...')
        
        count = PreComputedPrediction.objects.count()
        PreComputedPrediction.objects.all().delete()
        
        self.stdout.write(self.style.SUCCESS(f'✅ Deleted {count} predictions'))

    def clear_cache(self):
        """Clear only prediction caches"""
        self.stdout.write('🗑️  Clearing prediction caches...')
        
        count = PredictionCache.objects.count()
        PredictionCache.objects.all().delete()
        
        self.stdout.write(self.style.SUCCESS(f'✅ Deleted {count} cache entries'))

    def clear_inactive_predictions(self):
        """Clear only inactive predictions"""
        self.stdout.write('🗑️  Clearing inactive predictions...')
        
        count = PreComputedPrediction.objects.filter(is_active=False).count()
        PreComputedPrediction.objects.filter(is_active=False).delete()
        
        self.stdout.write(self.style.SUCCESS(f'✅ Deleted {count} inactive predictions'))

    def clear_category_predictions(self, category_name):
        """Clear predictions for specific category"""
        self.stdout.write(f'🗑️  Clearing predictions for category: {category_name}')
        
        try:
            category = DrawCategory.objects.get(name__icontains=category_name)
            count = PreComputedPrediction.objects.filter(category=category).count()
            PreComputedPrediction.objects.filter(category=category).delete()
            
            self.stdout.write(self.style.SUCCESS(f'✅ Deleted {count} predictions for {category.name}'))
        except DrawCategory.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'❌ Category not found: {category_name}'))
        except DrawCategory.MultipleObjectsReturned:
            categories = DrawCategory.objects.filter(name__icontains=category_name)
            self.stdout.write(self.style.ERROR(f'❌ Multiple categories found: {[c.name for c in categories]}'))

    def clear_old_predictions(self, days):
        """Clear predictions older than specified days"""
        self.stdout.write(f'🗑️  Clearing predictions older than {days} days...')
        
        cutoff_date = timezone.now() - timezone.timedelta(days=days)
        count = PreComputedPrediction.objects.filter(created_at__lt=cutoff_date).count()
        PreComputedPrediction.objects.filter(created_at__lt=cutoff_date).delete()
        
        # Also clear old cache entries
        cache_count = PredictionCache.objects.filter(created_at__lt=cutoff_date).count()
        PredictionCache.objects.filter(created_at__lt=cutoff_date).delete()
        
        self.stdout.write(self.style.SUCCESS(f'✅ Deleted {count} old predictions'))
        self.stdout.write(self.style.SUCCESS(f'✅ Deleted {cache_count} old cache entries')) 