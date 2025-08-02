#!/bin/bash

# Express Entry Predictor Data Update Script
# Usage: ./update_data.sh [--with-predictions] [--clear-data]

set -e  # Exit on any error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  Virtual environment not found, continuing..."
fi

echo "🔄 Starting Express Entry Predictor data update process..."
echo "📍 Working directory: $(pwd)"

# Parse command line arguments
CLEAR_DATA=false
WITH_PREDICTIONS=false

for arg in "$@"; do
    case $arg in
        --clear-data)
            CLEAR_DATA=true
            shift
            ;;
        --with-predictions)
            WITH_PREDICTIONS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--clear-data] [--with-predictions]"
            echo "  --clear-data        Clear existing data before loading"
            echo "  --with-predictions  Compute new predictions after data update"
            exit 0
            ;;
        *)
            echo "⚠️  Unknown argument: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Step 1: Update Express Entry draw data
echo ""
echo "📊 Loading Express Entry draw data..."
if [ -f "data/draw_data.csv" ]; then
    if [ "$CLEAR_DATA" = true ]; then
        echo "🗑️  Clearing existing draw data..."
        python manage.py load_draw_data --file data/draw_data.csv --clear
    else
        echo "📥 Loading draw data (preserving existing data)..."
        python manage.py load_draw_data --file data/draw_data.csv
    fi
    echo "✅ Draw data loaded successfully"
else
    echo "⚠️  data/draw_data.csv not found, skipping draw data update"
fi

# Step 2: Update economic indicators
echo ""
echo "💰 Updating economic indicators..."

# Collect recent economic data (last 3 months)
START_DATE=$(date -d '3 months ago' '+%Y-%m-%d' 2>/dev/null || date -v-3m '+%Y-%m-%d' 2>/dev/null || echo "2024-05-01")
END_DATE=$(date '+%Y-%m-%d')

echo "📈 Collecting economic data from $START_DATE to $END_DATE..."
if python manage.py collect_economic_data --start-date "$START_DATE" --end-date "$END_DATE"; then
    echo "✅ Economic indicators updated successfully"
else
    echo "⚠️  Economic data collection failed, continuing with existing data..."
fi

# Step 3: Check for new IRCC draws (if sync command exists)
echo ""
echo "🇨🇦 Checking for new IRCC draws..."
if python manage.py help | grep -q "sync_ircc_draws"; then
    python manage.py sync_ircc_draws --days-back 30
    echo "✅ IRCC draws synchronized"
else
    echo "⚠️  IRCC sync not available, skipping..."
fi

# Step 4: Update predictions if requested
if [ "$WITH_PREDICTIONS" = true ]; then
    echo ""
    echo "🔮 Computing new predictions..."
    echo "⏱️  This may take several minutes..."
    
    python manage.py compute_predictions --force
    echo "✅ Predictions computed successfully"
else
    echo ""
    echo "ℹ️  Skipping prediction computation (use --with-predictions to enable)"
fi

# Step 5: Restart web services (if running on VPS)
if command -v systemctl >/dev/null 2>&1; then
    echo ""
    echo "🔄 Restarting web services..."
    
    # Check if services exist before attempting restart
    if systemctl is-active --quiet gunicorn; then
        sudo systemctl restart gunicorn
        echo "✅ Gunicorn restarted"
    fi
    
    if systemctl is-active --quiet nginx; then
        sudo systemctl reload nginx
        echo "✅ Nginx reloaded"
    fi
fi

# Step 6: Clean up old cache
echo ""
echo "🧹 Cleaning up cache..."
python manage.py shell -c "
from django.core.cache import cache
cache.clear()
print('✅ Cache cleared')
"

echo ""
echo "🎉 Data update process completed successfully!"
echo "📊 Summary:"
echo "   ├─ Draw data: Updated"
echo "   ├─ Economic indicators: Updated"
echo "   ├─ IRCC sync: Attempted"
if [ "$WITH_PREDICTIONS" = true ]; then
    echo "   ├─ Predictions: Computed"
else
    echo "   ├─ Predictions: Skipped"
fi
echo "   └─ Cache: Cleared"
echo ""
echo "💡 Next steps:"
echo "   • Visit your website to verify updates"
echo "   • Check logs for any warnings"
echo "   • Run with --with-predictions to update forecasts" 