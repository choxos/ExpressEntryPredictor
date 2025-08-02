#!/bin/bash

# Setup script for automatic weekly IRCC Express Entry draws synchronization
# This script creates a cron job that runs every Monday at 9:00 AM

echo "🔧 Setting up weekly IRCC Express Entry draws synchronization..."

# Get the current directory (should be the project root)
PROJECT_DIR=$(pwd)
PYTHON_PATH="$PROJECT_DIR/venv/bin/python"
MANAGE_PY="$PROJECT_DIR/manage.py"
LOG_DIR="$PROJECT_DIR/logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Check if virtual environment exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "❌ Virtual environment not found at $PYTHON_PATH"
    echo "Please run this script from the project root directory with an active virtual environment"
    exit 1
fi

# Check if manage.py exists
if [ ! -f "$MANAGE_PY" ]; then
    echo "❌ manage.py not found at $MANAGE_PY"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Create the sync script
SYNC_SCRIPT="$PROJECT_DIR/weekly_ircc_sync.sh"
cat > "$SYNC_SCRIPT" << EOF
#!/bin/bash
# Weekly IRCC Express Entry Draws Synchronization Script
# Generated automatically by setup_weekly_sync.sh

echo "\$(date): Starting weekly IRCC synchronization..." >> "$LOG_DIR/ircc_sync.log"

# Navigate to project directory
cd "$PROJECT_DIR"

# Activate virtual environment and run sync
source venv/bin/activate

# Run the synchronization
$PYTHON_PATH $MANAGE_PY sync_ircc_draws --days-back 14 >> "$LOG_DIR/ircc_sync.log" 2>&1

# Check if sync was successful
if [ \$? -eq 0 ]; then
    echo "\$(date): IRCC synchronization completed successfully" >> "$LOG_DIR/ircc_sync.log"
else
    echo "\$(date): IRCC synchronization failed" >> "$LOG_DIR/ircc_sync.log"
fi

echo "----------------------------------------" >> "$LOG_DIR/ircc_sync.log"
EOF

# Make the sync script executable
chmod +x "$SYNC_SCRIPT"

echo "✅ Created sync script: $SYNC_SCRIPT"

# Create the cron job entry
CRON_ENTRY="0 9 * * 1 $SYNC_SCRIPT"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "$SYNC_SCRIPT"; then
    echo "⚠️  Cron job already exists for IRCC sync"
    echo "Current cron jobs:"
    crontab -l | grep ircc
else
    # Add the cron job
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
    echo "✅ Added weekly cron job: Every Monday at 9:00 AM"
fi

# Test the sync command
echo ""
echo "🧪 Testing the sync command (dry run)..."
cd "$PROJECT_DIR"
source venv/bin/activate
$PYTHON_PATH $MANAGE_PY sync_ircc_draws --dry-run --days-back 7

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📋 SETUP SUMMARY:"
    echo "   ├─ Sync script: $SYNC_SCRIPT"
    echo "   ├─ Log file: $LOG_DIR/ircc_sync.log"
    echo "   ├─ Schedule: Every Monday at 9:00 AM"
    echo "   └─ Command: python manage.py sync_ircc_draws"
    echo ""
    echo "💡 MANUAL COMMANDS:"
    echo "   # Test sync (dry run):"
    echo "   python manage.py sync_ircc_draws --dry-run"
    echo ""
    echo "   # Run sync manually:"
    echo "   python manage.py sync_ircc_draws"
    echo ""
    echo "   # Check logs:"
    echo "   tail -f $LOG_DIR/ircc_sync.log"
    echo ""
    echo "   # View current cron jobs:"
    echo "   crontab -l"
    echo ""
    echo "   # Remove cron job:"
    echo "   crontab -l | grep -v '$SYNC_SCRIPT' | crontab -"
else
    echo "❌ Test failed. Please check your setup and try again."
    exit 1
fi 