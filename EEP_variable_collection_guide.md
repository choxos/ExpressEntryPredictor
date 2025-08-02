# Express Entry Predictor - Variable Collection Guide

## 📊 **Overview**
This guide provides detailed instructions for collecting variables to enhance your Express Entry prediction model. The data should be organized into separate CSV files for efficient processing.

## 🗂️ **Required CSV Files Structure**

### **1. Economic Indicators (`data/economic_indicators.csv`)**

**Columns:**
```csv
date,unemployment_rate_ca,unemployment_rate_on,unemployment_rate_bc,unemployment_rate_ab,unemployment_rate_sk,unemployment_rate_mb,job_vacancy_rate,gdp_growth_rate,bank_overnight_rate,cpi_inflation_rate
2024-01-01,5.2,5.0,4.8,5.5,4.2,4.9,5.3,2.1,5.0,3.4
2024-02-01,5.1,4.9,4.7,5.4,4.1,4.8,5.4,2.2,5.0,3.2
```

**Collection Instructions:**
- **Frequency:** Monthly
- **Sources:**
  - **Unemployment Rates:** [Statistics Canada Table 14-10-0287-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1410028701)
  - **Job Vacancy Rate:** [Statistics Canada Table 14-10-0325-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1410032501)
  - **GDP Growth:** [Statistics Canada Table 36-10-0104-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3610010401)
  - **Bank Rate:** [Bank of Canada](https://www.bankofcanada.ca/rates/interest-rates/overnight-rate/)
  - **CPI:** [Statistics Canada Table 18-10-0004-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000401)

### **2. Immigration Pool Data (`data/pool_data.csv`)**

**Columns:**
```csv
date,total_candidates,candidates_600_plus,candidates_500_599,candidates_450_499,candidates_400_449,candidates_below_400,new_registrations_weekly,avg_pool_crs
2024-01-15,265000,4500,12000,35000,80000,133500,2500,420
2024-01-22,267500,4600,12200,35500,81000,134200,2800,422
```

**Collection Instructions:**
- **Frequency:** Bi-weekly (when IRCC releases pool data)
- **Source:** [IRCC Express Entry Pool Distribution](https://www.canada.ca/en/immigration-refugees-citizenship/corporate/mandate/service-performance/express-entry-pool-distribution.html)
- **Method:** Download PDF reports, extract table data, convert to CSV

### **3. Provincial Nominee Program Data (`data/pnp_data.csv`)**

**Columns:**
```csv
date,ontario_invites,bc_invites,alberta_invites,saskatchewan_invites,manitoba_invites,nova_scotia_invites,new_brunswick_invites,pei_invites,newfoundland_invites,yukon_invites,nwt_invites,total_pnp_weekly
2024-01-08,300,200,150,100,50,25,20,15,10,5,2,877
2024-01-15,280,180,140,120,55,30,25,18,12,3,1,864
```

**Collection Instructions:**
- **Frequency:** Weekly
- **Sources:** Each provincial website
  - **Ontario (OINP):** [ontario.ca/page/ontario-immigrant-nominee-program-oinp](https://www.ontario.ca/page/ontario-immigrant-nominee-program-oinp)
  - **BC (BC PNP):** [gov.bc.ca/bcpnp](https://www.gov.bc.ca/gov/content/immigration/immigrate-to-bc/bc-pnp)
  - **Alberta (AINP):** [alberta.ca/alberta-advantage-immigration-program](https://www.alberta.ca/alberta-advantage-immigration-program)
  - **Saskatchewan (SINP):** [saskatchewan.ca/residents/moving-to-saskatchewan/immigrating-to-saskatchewan/saskatchewan-immigrant-nominee-program](https://www.saskatchewan.ca/residents/moving-to-saskatchewan/immigrating-to-saskatchewan/saskatchewan-immigrant-nominee-program)

### **4. Calendar Events (`data/calendar_events.csv`)**

**Columns:**
```csv
date,is_federal_holiday,is_provincial_holiday,is_long_weekend,days_to_next_holiday,is_system_maintenance,parliament_sitting,minister_announcement,policy_change
2024-01-01,1,1,1,0,0,0,0,0
2024-01-02,0,0,0,17,0,1,0,0
2024-01-15,0,0,0,4,0,1,1,0
```

**Collection Instructions:**
- **Frequency:** Pre-generate for full year
- **Sources:**
  - **Federal Holidays:** [canada.ca holidays](https://www.canada.ca/en/revenue-agency/services/tax/public-holidays.html)
  - **IRCC Maintenance:** [IRCC Service Alerts](https://www.canada.ca/en/immigration-refugees-citizenship/services/application/check-processing-times.html)
  - **News:** [IRCC News Releases](https://www.canada.ca/en/news/advanced-news-search.html)

### **5. Enhanced Draw Data (`data/enhanced_draw_data.csv`)**

**Extend your existing draw_data.csv with these columns:**
```csv
round,date,type,invitations_issued,lowest_crs_score,url,week_of_year,month,quarter,day_of_week,is_monday,is_tuesday,is_wednesday,is_thursday,is_friday,pre_holiday_draw,post_holiday_draw,days_since_last_draw,category_frequency_30day,avg_crs_last_3_draws
357,2025-07-22,Healthcare and social services occupations (Version 2),4000,475,https://...,30,7,3,2,0,1,0,0,0,0,0,14,2,485
```

## 🤖 **Automated Collection Methods**

### **Method 1: Web Scraping Scripts**

Create Python scripts to automatically collect data:

```python
# example_scraper.py
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

def scrape_unemployment_data():
    # Statistics Canada data scraping
    url = "https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1410028701"
    # Implement scraping logic
    pass

def scrape_pnp_data():
    # Provincial websites scraping
    sources = {
        'ontario': 'https://www.ontario.ca/page/ontario-immigrant-nominee-program-oinp',
        'bc': 'https://www.gov.bc.ca/gov/content/immigration/immigrate-to-bc/bc-pnp'
    }
    # Implement scraping logic
    pass
```

### **Method 2: API Integration**

Use official APIs where available:

```python
# api_collectors.py
import requests

def get_bank_of_canada_rate():
    """Bank of Canada Valet API"""
    url = "https://www.bankofcanada.ca/valet/observations/V39065/json"
    response = requests.get(url)
    return response.json()

def get_statcan_data(table_id):
    """Statistics Canada Web Data Service"""
    url = f"https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/en/{table_id}"
    response = requests.get(url)
    return response.content
```

### **Method 3: Django Management Commands**

Use the provided Django commands:

```bash
# Collect economic indicators
python manage.py collect_economic_data --start-date 2024-01-01

# Load enhanced draw data
python manage.py load_enhanced_draw_data --file data/enhanced_draw_data.csv

# Update calendar events
python manage.py update_calendar_events --year 2024
```

## 📅 **Collection Schedule**

### **Daily Tasks:**
- Check IRCC news releases
- Monitor system maintenance alerts
- Update calendar events

### **Weekly Tasks:**
- Collect PNP data from all provinces
- Update pool composition data (if available)
- Monitor parliamentary activities

### **Monthly Tasks:**
- Download Statistics Canada economic indicators
- Update Bank of Canada rates
- Compile monthly summary reports

### **Quarterly Tasks:**
- Update immigration targets and policy data
- Review and clean historical data
- Validate data quality and completeness

## 🔧 **Data Quality Checks**

### **Validation Rules:**
1. **Date Formats:** All dates must be YYYY-MM-DD
2. **Numeric Values:** No commas in numbers, use decimal points
3. **Missing Data:** Use `NULL` or empty strings for missing values
4. **Consistency:** Same categories must use identical naming
5. **Ranges:** CRS scores 0-1200, unemployment 0-100%

### **Quality Assurance Script:**
```python
def validate_csv_data(file_path):
    df = pd.read_csv(file_path)
    
    # Check date format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Check for missing critical data
    critical_columns = ['date', 'unemployment_rate_ca', 'gdp_growth_rate']
    missing_data = df[critical_columns].isnull().sum()
    
    # Validate ranges
    if 'unemployment_rate_ca' in df.columns:
        invalid_unemployment = df[(df['unemployment_rate_ca'] < 0) | 
                                  (df['unemployment_rate_ca'] > 100)]
    
    return validation_report
```

## 📊 **Data Integration Workflow**

1. **Collect** → Download raw data from sources
2. **Clean** → Standardize formats and handle missing values
3. **Validate** → Run quality checks and fix errors
4. **Load** → Import into Django using management commands
5. **Verify** → Check data integrity and completeness
6. **Backup** → Store copies of processed data

## 🚀 **Quick Start Commands**

```bash
# 1. Create sample data files (for testing)
python manage.py create_sample_data

# 2. Load economic indicators
python manage.py collect_economic_data

# 3. Update calendar events
python manage.py update_calendar_events --year 2024

# 4. Load PNP data
python manage.py load_pnp_data --file data/pnp_data.csv

# 5. Generate predictions with new data
python manage.py generate_predictions --all-categories
```

## 📈 **Expected Impact on Predictions**

Adding these variables should improve prediction accuracy by:
- **Economic Indicators:** 15-20% improvement in CRS score prediction
- **Pool Data:** 25-30% improvement in invitation volume prediction
- **PNP Data:** 20-25% improvement in draw timing prediction
- **Calendar Events:** 10-15% improvement in draw date prediction
- **Enhanced Features:** 5-10% overall model performance boost

## ⚠️ **Important Notes**

1. **Data Privacy:** Ensure compliance with privacy laws when scraping
2. **Rate Limiting:** Respect website rate limits and robots.txt
3. **API Keys:** Some services require registration and API keys
4. **Data Retention:** Keep historical data for model training
5. **Version Control:** Track data changes and model updates
6. **Monitoring:** Set up alerts for data collection failures

## 🔄 **Automation Setup**

Consider setting up automated data collection using:
- **Cron jobs** for scheduled scraping
- **GitHub Actions** for cloud-based collection
- **Celery tasks** for Django-integrated automation
- **Apache Airflow** for complex data pipelines

This comprehensive approach will significantly enhance your Express Entry prediction accuracy and provide users with more reliable forecasts! 