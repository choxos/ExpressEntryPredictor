#!/usr/bin/env python
"""
HEALTHCARE PREDICTION FIX SCRIPT
================================

This script fixes the unrealistic Healthcare predictions showing 2026 dates by:
1. Clearing old Healthcare predictions from database
2. Clearing prediction cache 
3. Providing instructions for re-running predictions

Usage: python fix_healthcare_predictions.py
"""

import os
import sys
import django
from datetime import date

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'expressentry_predictor.settings')
django.setup()

from predictor.models import PreComputedPrediction, PredictionCache, ExpressEntryCategory

def main():
    print("🚀 HEALTHCARE PREDICTION FIX SCRIPT")
    print("=" * 50)
    
    # 1. Find Healthcare categories
    healthcare_categories = ExpressEntryCategory.objects.filter(
        name__icontains='Healthcare'
    )
    
    print(f"\n📊 Found {healthcare_categories.count()} Healthcare categories:")
    for cat in healthcare_categories:
        print(f"   - {cat.name} (ID: {cat.id})")
    
    # 2. Check existing Healthcare predictions
    healthcare_predictions = PreComputedPrediction.objects.filter(
        category__in=healthcare_categories
    ).order_by('category__name', 'prediction_rank')
    
    print(f"\n📋 Found {healthcare_predictions.count()} Healthcare predictions:")
    
    # Group by category and show date ranges
    for cat in healthcare_categories:
        cat_preds = healthcare_predictions.filter(category=cat)
        if cat_preds.exists():
            dates = [p.predicted_date for p in cat_preds]
            min_date = min(dates)
            max_date = max(dates)
            ranks = cat_preds.count()
            
            # Check for 2026+ dates
            future_dates = [d for d in dates if d.year >= 2026]
            status = "🚨 HAS 2026+ DATES" if future_dates else "✅ Reasonable dates"
            
            print(f"   {cat.name[:50]:50} | {ranks:2d} ranks | {min_date} to {max_date} | {status}")
            
            if future_dates:
                print(f"      └─ 2026+ dates: {len(future_dates)}/{len(dates)} predictions")
    
    # 3. Confirm deletion
    if healthcare_predictions.exists():
        print(f"\n⚠️  ABOUT TO DELETE {healthcare_predictions.count()} Healthcare predictions")
        print("   This will remove old predictions with unrealistic 2026+ dates")
        
        response = input("\n🔥 Proceed with deletion? (yes/no): ").lower().strip()
        
        if response == 'yes':
            # Delete Healthcare predictions
            deleted_count = healthcare_predictions.delete()[0]
            print(f"   ✅ Deleted {deleted_count} Healthcare predictions")
            
            # Clear prediction cache
            cache_deleted = PredictionCache.objects.filter(
                cache_key__icontains='Healthcare'
            ).delete()[0]
            print(f"   ✅ Cleared {cache_deleted} Healthcare cache entries")
            
            # Clear general prediction cache too
            general_cache_deleted = PredictionCache.objects.filter(
                cache_key__icontains='predictions_api'
            ).delete()[0]
            print(f"   ✅ Cleared {general_cache_deleted} general prediction cache entries")
            
            print("\n🎯 NEXT STEPS:")
            print("   1. The Healthcare interval has been reduced from 94.4 to 35.0 days")
            print("   2. Added comprehensive date logging to prediction system")
            print("   3. Run predictions with: python manage.py compute_predictions --force")
            print("   4. Check the log file for detailed date assignments")
            print("   5. Restart your expressentry service on VPS")
            
        else:
            print("   ❌ Deletion cancelled")
    else:
        print("\n✅ No Healthcare predictions found to delete")
    
    print(f"\n📅 Current date: {date.today()}")
    print("=" * 50)
    print("🏁 Healthcare fix script completed!")

if __name__ == "__main__":
    main()