from django.core.management.base import BaseCommand
from django.conf import settings
from django.db import connection
from predictor.models import PreComputedPrediction, ExpressEntryDraw, DrawCategory
import os
import subprocess


class Command(BaseCommand):
    help = 'Verify VPS deployment configuration and detect common issues'

    def handle(self, *args, **options):
        self.stdout.write('🔍 Express Entry Predictor - Deployment Verification')
        self.stdout.write('=' * 60)
        
        # Check database connection
        self.check_database()
        
        # Check static files configuration  
        self.check_static_files()
        
        # Check data status
        self.check_data_status()
        
        # Check predictions
        self.check_predictions()
        
        # Check system status
        self.check_system_status()
        
        self.stdout.write('\n' + '=' * 60)
        self.stdout.write('✅ Deployment verification completed!')

    def check_database(self):
        """Check database connection and configuration"""
        self.stdout.write('\n📊 DATABASE STATUS:')
        try:
            connection.ensure_connection()
            db_name = connection.settings_dict['NAME']
            db_user = connection.settings_dict['USER']
            db_host = connection.settings_dict['HOST'] or 'localhost'
            
            self.stdout.write(f'   ✅ Database: Connected to {db_name}')
            self.stdout.write(f'   ✅ User: {db_user}')
            self.stdout.write(f'   ✅ Host: {db_host}')
            
        except Exception as e:
            self.stdout.write(f'   ❌ Database: Connection failed - {e}')

    def check_static_files(self):
        """Check static files configuration"""
        self.stdout.write('\n📁 STATIC FILES STATUS:')
        
        try:
            static_root = getattr(settings, 'STATIC_ROOT', None)
            staticfiles_dirs = getattr(settings, 'STATICFILES_DIRS', [])
            
            self.stdout.write(f'   📂 STATIC_ROOT: {static_root}')
            
            if static_root and os.path.exists(static_root):
                self.stdout.write(f'   ✅ STATIC_ROOT directory exists')
            elif static_root:
                self.stdout.write(f'   ⚠️  STATIC_ROOT directory missing: {static_root}')
                self.stdout.write(f'   💡 Fix: mkdir -p {static_root}')
            
            self.stdout.write(f'   📂 STATICFILES_DIRS: {staticfiles_dirs}')
            
            for static_dir in staticfiles_dirs:
                if os.path.exists(static_dir):
                    self.stdout.write(f'   ✅ Static directory exists: {static_dir}')
                else:
                    self.stdout.write(f'   ⚠️  Static directory missing: {static_dir}')
                    self.stdout.write(f'   💡 Fix: mkdir -p {static_dir}')
                    
        except Exception as e:
            self.stdout.write(f'   ❌ Static files check failed: {e}')

    def check_data_status(self):
        """Check data loading status"""
        self.stdout.write('\n📈 DATA STATUS:')
        
        try:
            total_draws = ExpressEntryDraw.objects.count()
            total_categories = DrawCategory.objects.count()
            active_categories = DrawCategory.objects.filter(is_active=True).count()
            
            self.stdout.write(f'   📊 Total draws: {total_draws}')
            self.stdout.write(f'   📂 Total categories: {total_categories}')
            self.stdout.write(f'   ✅ Active categories: {active_categories}')
            
            if total_draws < 300:
                self.stdout.write(f'   ⚠️  Low draw count - expected 350+')
                self.stdout.write(f'   💡 Fix: python manage.py load_draw_data')
                
        except Exception as e:
            self.stdout.write(f'   ❌ Data check failed: {e}')

    def check_predictions(self):
        """Check prediction status"""
        self.stdout.write('\n🔮 PREDICTIONS STATUS:')
        
        try:
            total_predictions = PreComputedPrediction.objects.count()
            active_predictions = PreComputedPrediction.objects.filter(is_active=True).count()
            categories_with_predictions = PreComputedPrediction.objects.filter(
                is_active=True
            ).values('category').distinct().count()
            
            self.stdout.write(f'   🎯 Total predictions: {total_predictions}')
            self.stdout.write(f'   ✅ Active predictions: {active_predictions}')
            self.stdout.write(f'   📂 Categories with predictions: {categories_with_predictions}')
            
            if active_predictions < 100:
                self.stdout.write(f'   ⚠️  Low prediction count - expected 100+')
                self.stdout.write(f'   💡 Fix: python manage.py compute_predictions --force')
                
        except Exception as e:
            self.stdout.write(f'   ❌ Predictions check failed: {e}')

    def check_system_status(self):
        """Check system configuration"""
        self.stdout.write('\n⚙️  SYSTEM STATUS:')
        
        try:
            # Check DEBUG setting
            debug_mode = getattr(settings, 'DEBUG', True)
            if debug_mode:
                self.stdout.write(f'   ⚠️  DEBUG=True (should be False for production)')
                self.stdout.write(f'   💡 Fix: Set DEBUG=False in .env file')
            else:
                self.stdout.write(f'   ✅ DEBUG=False (production ready)')
            
            # Check allowed hosts
            allowed_hosts = getattr(settings, 'ALLOWED_HOSTS', [])
            if 'expressentry.xeradb.com' in allowed_hosts:
                self.stdout.write(f'   ✅ Domain configured: expressentry.xeradb.com')
            else:
                self.stdout.write(f'   ⚠️  Domain not in ALLOWED_HOSTS: {allowed_hosts}')
            
            # Check if we can detect the current directory
            current_dir = os.getcwd()
            if '/var/www/expressentry' in current_dir:
                self.stdout.write(f'   ✅ Running from correct directory: {current_dir}')
            else:
                self.stdout.write(f'   ⚠️  Unexpected directory: {current_dir}')
                
        except Exception as e:
            self.stdout.write(f'   ❌ System check failed: {e}') 