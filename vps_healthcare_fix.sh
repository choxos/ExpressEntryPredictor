#!/bin/bash

# VPS HEALTHCARE PREDICTION FIX SCRIPT
# ====================================
# This script fixes the 2026 Healthcare prediction issue on your VPS

echo "🚀 VPS HEALTHCARE PREDICTION FIX"
echo "================================"

# Set database URL for VPS
export DATABASE_URL="postgresql://eep_user:Choxos10203040@localhost:5432/eep_production"

cd /var/www/eep

echo ""
echo "📊 STEP 1: Analyzing current Healthcare predictions..."
python manage.py shell -c "
from predictor.models import PreComputedPrediction, ExpressEntryCategory
from datetime import date

healthcare_cats = ExpressEntryCategory.objects.filter(name__icontains='Healthcare')
print(f'Healthcare categories found: {healthcare_cats.count()}')

for cat in healthcare_cats:
    preds = PreComputedPrediction.objects.filter(category=cat).order_by('prediction_rank')
    if preds.exists():
        dates = [p.predicted_date for p in preds]
        future_dates = [d for d in dates if d.year >= 2026]
        print(f'{cat.name}: {preds.count()} predictions, {len(future_dates)} in 2026+')
        if future_dates:
            print(f'  First: {min(dates)}, Last: {max(dates)}')
"

echo ""
echo "🗑️  STEP 2: Clearing old Healthcare predictions..."
python manage.py shell -c "
from predictor.models import PreComputedPrediction, PredictionCache, ExpressEntryCategory

# Delete Healthcare predictions
healthcare_cats = ExpressEntryCategory.objects.filter(name__icontains='Healthcare')
deleted = PreComputedPrediction.objects.filter(category__in=healthcare_cats).delete()
print(f'Deleted {deleted[0]} Healthcare predictions')

# Clear caches
cache_deleted = PredictionCache.objects.all().delete()
print(f'Cleared {cache_deleted[0]} cache entries')

print('✅ Cleanup completed!')
"

echo ""
echo "🔧 STEP 3: Running updated predictions with enhanced logging..."
python manage.py compute_predictions --force 2>&1 | tee healthcare_fix_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "📋 STEP 4: Verifying new Healthcare predictions..."
python manage.py shell -c "
from predictor.models import PreComputedPrediction, ExpressEntryCategory
from datetime import date

healthcare_cats = ExpressEntryCategory.objects.filter(name__icontains='Healthcare')
print(f'=== NEW HEALTHCARE PREDICTIONS ===')

for cat in healthcare_cats:
    preds = PreComputedPrediction.objects.filter(category=cat, is_active=True).order_by('prediction_rank')[:5]
    if preds.exists():
        print(f'{cat.name}:')
        for pred in preds:
            days_from_now = (pred.predicted_date - date.today()).days
            print(f'  Rank {pred.prediction_rank}: {pred.predicted_date} ({days_from_now} days) CRS {pred.predicted_crs_score}')
    else:
        print(f'{cat.name}: No predictions found!')

# Check for any remaining 2026+ dates
future_preds = PreComputedPrediction.objects.filter(predicted_date__gte='2026-01-01', is_active=True)
if future_preds.exists():
    print(f'⚠️  Still have {future_preds.count()} predictions in 2026+!')
    for pred in future_preds[:3]:
        print(f'  {pred.category.name}: {pred.predicted_date}')
else:
    print('✅ No 2026+ dates found!')
"

echo ""
echo "🔄 STEP 5: Restarting expressentry service..."
sudo systemctl restart expressentry
sleep 5
sudo systemctl status expressentry

echo ""
echo "🌐 STEP 6: Testing API response..."
curl -s "https://expressentry.xeradb.com/api/predict/?t=$(date +%s)" | head -200

echo ""
echo "✅ Healthcare fix completed!"
echo "Check the log file for detailed date assignments"
echo "Visit your website to verify Healthcare predictions are now reasonable"