# 🇨🇦 IRCC Express Entry Synchronization Guide

This guide explains how to keep your Express Entry Predictor automatically updated with the latest draws from the official [IRCC Express Entry rounds page](https://www.canada.ca/en/immigration-refugees-citizenship/corporate/mandate/policies-operational-instructions-agreements/ministerial-instructions/express-entry-rounds.html).

## 🚀 **Automated Weekly Synchronization**

### **Quick Setup**
```bash
# Install dependencies
pip install requests beautifulsoup4 lxml

# Set up weekly automation (runs every Wednesday at 3 PM)
chmod +x setup_weekly_sync.sh
./setup_weekly_sync.sh
```

### **Manual Commands**
```bash
# Test sync (dry run)
python manage.py sync_ircc_draws --dry-run

# Run sync manually
python manage.py sync_ircc_draws

# Check recent draws (last 7 days)
python manage.py sync_ircc_draws --days-back 7

# Force update even if no new draws
python manage.py sync_ircc_draws --force
```

## 🔧 **How It Works**

### **1. Weekly Monitoring**
- **Schedule**: Every Wednesday at 3:00 PM
- **Data Source**: Official IRCC website
- **Process**: 
  1. Fetches the latest Express Entry rounds page
  2. Parses the table for new draws
  3. Updates database with new entries
  4. Regenerates predictions for affected categories
  5. Logs all activities for monitoring

### **2. Intelligent Category Mapping**
The system automatically maps IRCC round types to our categories:

| IRCC Round Type | Database Category |
|----------------|-------------------|
| General | No Program Specified |
| Canadian Experience Class | Canadian Experience Class |
| Provincial Nominee Program | Provincial Nominee Program |
| French-language proficiency | French language proficiency (Version 1) |
| Healthcare occupations | Healthcare occupations (Version 1) |
| STEM occupations | STEM occupations (Version 1) |
| Trade occupations | Trade occupations (Version 1) |
| Agriculture and agri-food | Agriculture and agri-food occupations (Version 1) |
| Education occupations | Education occupations (Version 1) |

### **3. Automatic Prediction Updates**
When new draws are detected:
- ✅ Database updated with new draw data
- 🔄 Predictions regenerated for affected categories using **pooled data**
- 📊 Enhanced models leverage more training data
- 🎯 Category-specific predictions remain accurate

## ⚠️ **Current Limitation: JavaScript-Loaded Data**

**Issue**: The IRCC website loads draw data dynamically via JavaScript, making direct HTML scraping challenging.

**Detection**: When you see this message:
```
⚠️ No data rows found in table
🔍 This likely means the IRCC website loads data dynamically via JavaScript
```

## 🛠️ **Alternative Solutions**

### **Option 1: Manual CSV Import (Recommended)**

**Steps:**
1. **Export IRCC Data**: Visit the [IRCC page](https://www.canada.ca/en/immigration-refugees-citizenship/corporate/mandate/policies-operational-instructions-agreements/ministerial-instructions/express-entry-rounds.html) and export/copy the table data
2. **Update CSV**: Add new draws to `data/draw_data.csv`
3. **Load Data**: 
   ```bash
   python manage.py load_draw_data
   python manage.py compute_predictions --force
   ```

**CSV Format:**
```csv
Date,Round,Category,Invitations_Issued,Lowest_CRS_Score,URL
2025-01-15,123,Canadian Experience Class,3000,456,https://...
2025-01-22,124,Healthcare occupations (Version 1),500,478,https://...
```

### **Option 2: Browser Automation (Advanced)**

**Install Selenium:**
```bash
pip install selenium webdriver-manager
```

**Create Enhanced Scraper:**
```python
# predictor/management/commands/sync_ircc_selenium.py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def fetch_with_selenium():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    
    try:
        driver.get("https://www.canada.ca/...")
        # Wait for JavaScript to load data
        time.sleep(5)
        
        # Extract table data after JS execution
        table = driver.find_element(By.TAG_NAME, "table")
        # ... parse data ...
        
    finally:
        driver.quit()
```

### **Option 3: IRCC API Integration (If Available)**

**Check for Official APIs:**
```bash
# Research potential IRCC data APIs
curl -H "Accept: application/json" "https://www.canada.ca/api/..."
```

### **Option 4: Data Monitoring Service**

**Set Up Change Detection:**
```bash
# Use a service like visualping.io or uptimerobot.com
# Monitor IRCC page for changes
# Trigger manual updates when changes detected
```

## 📊 **Monitoring & Logs**

### **Check Sync Logs**
```bash
# View recent sync activity
tail -f logs/ircc_sync.log

# Check last 50 log entries
tail -n 50 logs/ircc_sync.log

# Search for errors
grep -i "error\|failed" logs/ircc_sync.log
```

### **Verify Database Updates**
```bash
# Check latest draws
python manage.py shell -c "
from predictor.models import ExpressEntryDraw
latest = ExpressEntryDraw.objects.order_by('-date')[:5]
for draw in latest:
    print(f'{draw.date}: {draw.category.name} - CRS {draw.lowest_crs_score}')
"

# Check prediction counts
python manage.py shell -c "
from predictor.models import PreComputedPrediction
print(f'Total predictions: {PreComputedPrediction.objects.count()}')
print(f'Active predictions: {PreComputedPrediction.objects.filter(is_active=True).count()}')
"
```

## 🔄 **Cron Job Management**

### **View Current Jobs**
```bash
crontab -l
```

### **Edit Cron Jobs**
```bash
crontab -e
```

### **Remove IRCC Sync Job**
```bash
crontab -l | grep -v 'weekly_ircc_sync.sh' | crontab -
```

### **Manual Cron Setup**
```bash
# Add this line to crontab (Wednesday 3 PM)
0 15 * * 3 /path/to/project/weekly_ircc_sync.sh
```

## 🎯 **Best Practices**

### **1. Weekly Monitoring Routine**
- ✅ Check logs every Wednesday after 4 PM
- ✅ Verify new draws were processed correctly
- ✅ Confirm predictions were updated
- ✅ Monitor for any error messages

### **2. Manual Verification**
```bash
# After each sync, verify data integrity
python manage.py shell -c "
from predictor.models import ExpressEntryDraw
from datetime import datetime, timedelta

# Check for recent draws
recent = datetime.now() - timedelta(days=14)
new_draws = ExpressEntryDraw.objects.filter(created_at__gte=recent)
print(f'New draws in last 14 days: {new_draws.count()}')
"
```

### **3. Backup Strategy**
```bash
# Before major updates, backup database
pg_dump your_database > backup_before_sync_$(date +%Y%m%d).sql

# Backup predictions
python manage.py dumpdata predictor.PreComputedPrediction > predictions_backup.json
```

## 🚨 **Troubleshooting**

### **Common Issues**

**1. No Data Found**
```
Solution: Switch to manual CSV import method
Check: IRCC website structure may have changed
```

**2. Parsing Errors**
```
Solution: Update category mapping in determine_category()
Check: New round types introduced by IRCC
```

**3. Connection Timeouts**
```
Solution: Increase timeout in requests.get(timeout=60)
Check: Network connectivity and IRCC website status
```

**4. Duplicate Draws**
```
Solution: System handles duplicates automatically
Check: Date parsing accuracy in parse_date()
```

### **Debug Commands**
```bash
# Test IRCC website connectivity
python manage.py test_ircc_parser

# Verbose sync with debug info
python manage.py sync_ircc_draws --dry-run --verbosity=2

# Clear problematic predictions
python manage.py clear_predictions --older-than 30 --confirm
```

## 🎉 **Success Indicators**

### **Healthy Sync Operation**
- ✅ Weekly logs show successful completion
- ✅ New draws appear in database within 24 hours
- ✅ Predictions update automatically for new categories
- ✅ No error messages in logs
- ✅ Website shows current predictions

### **Expected Log Output**
```
2025-08-13 15:00:01: Starting weekly IRCC synchronization...
✅ Successfully fetched IRCC page (145623 bytes)
📊 Found 2 total draws, 1 recent draws
✅ Created: Canadian Experience Class - 2025-08-12 (CRS: 462, Invitations: 3000)
🎯 Updating predictions for: Canadian Experience Class
✅ Predictions updated for: Canadian Experience Class
📊 SYNCHRONIZATION SUMMARY: 1 categories updated
2025-08-13 15:02:15: IRCC synchronization completed successfully
```

## 📞 **Support**

### **Getting Help**
1. **Check Logs**: `tail -f logs/ircc_sync.log`
2. **Test Commands**: Run with `--dry-run` first
3. **Manual Fallback**: Use CSV import method
4. **Verify Setup**: Run `./setup_weekly_sync.sh` again

### **Reporting Issues**
When reporting sync issues, include:
- Error messages from logs
- Output of `python manage.py test_ircc_parser`
- Recent changes to IRCC website
- Your current sync configuration

---

**🇨🇦 Keep your Express Entry predictions current with automated IRCC synchronization!** 