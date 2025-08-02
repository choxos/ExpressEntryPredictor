# Express Entry Predictor - Deployment Guide

*Updated: August 2025 - Latest Version with Pre-computed Prediction System*

This comprehensive guide provides step-by-step instructions for deploying the Express Entry Predictor application with all the latest features and improvements.

## 🎯 System Overview

The Express Entry Predictor is a Django-based web application that uses **8 machine learning models** to predict Canadian Express Entry draws. Key features:

- ⚡ **Pre-computed predictions** for fast loading (no real-time ML calculations)
- 🤖 **8 ML models**: ARIMA, LSTM, XGBoost, Random Forest, Linear Regression, Neural Networks, Prophet, Ensemble
- 📊 **Clean navigation**: Home | Predictions | Analytics
- 🔄 **Admin-controlled updates** when new data is added
- 📈 **358+ historical draws** across 14 categories

---

## 🚀 Quick Start (Recommended)

### 1. Prerequisites

- **Python 3.12+**
- **Git**
- **Virtual environment** (recommended)

### 2. Installation & Setup

```bash
# 1. Clone and setup environment
git clone https://github.com/choxos/ExpressEntryPredictor.git
cd ExpressEntryPredictor
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies (Python 3.12 compatible versions)
pip install -r requirements.txt

# 3. Setup database with fresh schema
python3 manage.py migrate

# 4. Load initial data and models
python3 manage.py setup_initial_data
python3 manage.py load_draw_data

# 5. Generate pre-computed predictions
python3 manage.py compute_predictions

# 6. Create admin user (optional)
python3 manage.py createsuperuser

# 7. Start server
python3 manage.py runserver 8002
```

### 3. Access Your Application

- **🏠 Home Page**: http://127.0.0.1:8002/
- **🔮 Predictions**: http://127.0.0.1:8002/predictions/
- **📊 Analytics**: http://127.0.0.1:8002/analytics/
- **⚙️ Admin Panel**: http://127.0.0.1:8002/admin/
- **🔗 API Root**: http://127.0.0.1:8002/api/

---

## 📦 Dependencies & Versions

### Core Dependencies (Latest Versions)
```python
# Framework
Django==4.2.16
djangorestframework==3.14.0
django-cors-headers==4.3.1

# Database & Storage
psycopg2-binary==2.9.9  # PostgreSQL (production)
dj-database-url==2.1.0
whitenoise==6.6.0

# Configuration
python-decouple==3.8

# Data Science & ML (Python 3.12 compatible)
pandas==2.2.2
numpy>=1.26.0
scikit-learn==1.4.2
xgboost==2.0.3
tensorflow>=2.16.0
statsmodels==0.14.5
prophet==1.1.7

# Visualization
plotly==5.19.0
seaborn==0.13.2
matplotlib==3.8.4

# Utilities
requests==2.32.3
python-dateutil==2.9.0
holidays==0.45
beautifulsoup4==4.12.3

# Background Tasks
celery==5.3.6
redis==5.0.4

# Development
django-debug-toolbar==4.3.0
black==24.4.2
flake8==7.0.0
```

### Verify Installation
```bash
# Test all ML libraries
python3 -c "
import django, sklearn, xgboost, tensorflow, statsmodels, pandas, numpy, plotly
from prophet import Prophet
print('✅ All dependencies working!')
print(f'Python: {django.get_version()}')
print(f'TensorFlow: {tensorflow.__version__}')
print(f'Prophet: Available')
"
```

---

## 🗄️ Data Management

### Initial Data Loading

```bash
# 1. Setup prediction models in database
python3 manage.py setup_initial_data
# ✅ Creates 8 prediction models: ARIMA, LSTM, XGBoost, Random Forest, 
#    Linear Regression, Neural Networks, Prophet, Ensemble

# 2. Load historical Express Entry draws
python3 manage.py load_draw_data
# ✅ Loads 358+ draws across 14 categories from data/draw_data.csv

# 3. Generate predictions for all categories
python3 manage.py compute_predictions
# ✅ Creates pre-computed predictions for categories with sufficient data
```

### Adding New Data (Admin Workflow)

When new Express Entry draws are published:

```bash
# 1. Update your CSV file (data/draw_data.csv) with new draws

# 2. Reload the data
python3 manage.py load_draw_data

# 3. Regenerate predictions with latest data
python3 manage.py compute_predictions --force

# 4. Auto-commit changes (optional)
./auto_commit.sh
```

### Advanced Data Management

```bash
# Generate predictions for specific category
python3 manage.py compute_predictions --category "Canadian Experience Class"

# Generate more predictions (default is 10)
python3 manage.py compute_predictions --predictions 15

# Force recomputation even if recent predictions exist
python3 manage.py compute_predictions --force

# Evaluate model performance
python3 manage.py evaluate_models
```

---

## 🤖 Machine Learning Models

### Available Models (8 Total)

| Model | Type | Best For | Accuracy |
|-------|------|----------|----------|
| **ARIMA** | Time Series | Seasonal patterns | ⭐⭐⭐⭐⭐ |
| **LSTM** | Deep Learning | Complex sequences | ⭐⭐⭐⭐⭐ |
| **XGBoost** | Gradient Boosting | Feature-based | ⭐⭐⭐⭐⭐ |
| **Prophet** | Time Series | Trend + seasonality | ⭐⭐⭐⭐ |
| **Random Forest** | Ensemble | Robust predictions | ⭐⭐⭐⭐ |
| **Linear Regression** | Statistical | Interpretable baseline | ⭐⭐⭐ |
| **Neural Network** | Deep Learning | Non-linear patterns | ⭐⭐⭐⭐ |
| **Ensemble** | Combined | Maximum accuracy | ⭐⭐⭐⭐⭐ |

### Model Selection Strategy

The system automatically selects the best model for each category based on:
- **Data quantity** (more data = more sophisticated models)
- **Data quality** (stability and variance)
- **Model performance** (validation metrics)

### Testing Models

```bash
# Test individual models
python3 manage.py run_predictions --model sarima --steps 4
python3 manage.py run_predictions --model xgboost --steps 4
python3 manage.py run_predictions --model lstm --steps 4
python3 manage.py run_predictions --model neural --steps 4
python3 manage.py run_predictions --model ensemble --steps 4

# Compare all models
python3 manage.py evaluate_models
```

---

## 🌐 Production Deployment

### Environment Variables

Create `.env` file:
```bash
# Django Settings
SECRET_KEY=your-super-secret-key-here
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

# Database (PostgreSQL recommended for production)
DATABASE_URL=postgresql://user:password@localhost:5432/express_entry_db

# Optional: API Keys
STATCAN_API_KEY=your-statistics-canada-api-key
BOC_API_KEY=your-bank-of-canada-api-key
```

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Setup application
RUN python manage.py collectstatic --noinput
RUN python manage.py migrate
RUN python manage.py setup_initial_data
RUN python manage.py load_draw_data

# Generate initial predictions
RUN python manage.py compute_predictions

EXPOSE 8000

CMD ["gunicorn", "expressentry_predictor.wsgi:application", "--bind", "0.0.0.0:8000"]
```

**Build and run:**
```bash
docker build -t express-entry-predictor .
docker run -p 8000:8000 express-entry-predictor
```

### Heroku Deployment

```bash
# 1. Install Heroku CLI and login
heroku login

# 2. Create app
heroku create your-app-name

# 3. Add PostgreSQL
heroku addons:create heroku-postgresql:essential-0

# 4. Set environment variables
heroku config:set SECRET_KEY=your-secret-key
heroku config:set DEBUG=False

# 5. Deploy
git push heroku main

# 6. Setup database
heroku run python manage.py migrate
heroku run python manage.py setup_initial_data
heroku run python manage.py load_draw_data
heroku run python manage.py compute_predictions
heroku run python manage.py createsuperuser
```

### AWS/DigitalOcean Deployment

1. **Server Setup:**
```bash
# Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv nginx postgresql

# Setup application
git clone https://github.com/choxos/ExpressEntryPredictor.git
cd ExpressEntryPredictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt gunicorn
```

2. **Database Setup:**
```bash
sudo -u postgres createdb express_entry_db
sudo -u postgres createuser express_entry_user
```

3. **Nginx Configuration:**
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location = /favicon.ico { access_log off; log_not_found off; }
    location /static/ {
        root /path/to/ExpressEntryPredictor;
    }

    location / {
        include proxy_params;
        proxy_pass http://unix:/path/to/ExpressEntryPredictor/gunicorn.sock;
    }
}
```

4. **Systemd Service:**
```ini
[Unit]
Description=Express Entry Predictor gunicorn daemon
Requires=gunicorn.socket
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/path/to/ExpressEntryPredictor
ExecStart=/path/to/ExpressEntryPredictor/venv/bin/gunicorn \
          --access-logfile - \
          --workers 3 \
          --bind unix:/path/to/ExpressEntryPredictor/gunicorn.sock \
          expressentry_predictor.wsgi:application

[Install]
WantedBy=multi-user.target
```

---

## 🔄 Maintenance & Updates

### Daily Operations

**For ongoing maintenance, you only need:**

1. **Check website is running**: Visit your URL
2. **Update predictions when new draws published**:
   ```bash
   # Update CSV with new data, then:
   python3 manage.py load_draw_data
   python3 manage.py compute_predictions --force
   ```

### Weekly Tasks

```bash
# 1. Backup database
python3 manage.py dumpdata > backup_$(date +%Y%m%d).json

# 2. Check model performance
python3 manage.py evaluate_models

# 3. Update system
git pull origin main
pip install -r requirements.txt --upgrade
python3 manage.py migrate
python3 manage.py compute_predictions --force
```

### Auto-Commit Workflow

Use the included auto-commit script:
```bash
# Make changes, then auto-commit
./auto_commit.sh

# Or manually
git add .
git commit -m "Updated predictions with latest data"
git push origin main
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Prophet Import Error:**
```bash
# Fix NumPy compatibility
pip install prophet==1.1.7 --upgrade --force-reinstall
```

**2. No Predictions Generated:**
```bash
# Check if data is loaded
python3 manage.py shell -c "from predictor.models import ExpressEntryDraw; print(f'Draws: {ExpressEntryDraw.objects.count()}')"

# Check if models exist
python3 manage.py shell -c "from predictor.models import PredictionModel; print(f'Models: {PredictionModel.objects.count()}')"

# Regenerate predictions
python3 manage.py compute_predictions --force
```

**3. Database Issues:**
```bash
# Reset migrations (if needed)
rm predictor/migrations/000*.py
python3 manage.py makemigrations predictor
python3 manage.py migrate
```

**4. Memory Issues:**
```bash
# Use specific models for large datasets
python3 manage.py compute_predictions --model linear  # Fastest
python3 manage.py compute_predictions --model xgboost  # Good balance
```

### Performance Optimization

**1. Enable Caching:**
```python
# In settings.py
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
    }
}
```

**2. Database Optimization:**
```bash
# PostgreSQL recommended for production
pip install psycopg2-binary
# Set DATABASE_URL in environment
```

**3. Static Files:**
```bash
# Collect static files for production
python3 manage.py collectstatic
```

---

## 📊 API Documentation

### Key Endpoints

```bash
# Get all predictions
GET /api/predict/

# Get predictions for specific category
GET /api/predict/{category_id}/

# Get dashboard statistics
GET /api/stats/

# Get recent draws
GET /api/draws/recent/

# Get category statistics
GET /api/categories/{id}/statistics/
```

### API Response Examples

**Predictions API:**
```json
{
  "success": true,
  "total_categories": 2,
  "generated_at": "2025-08-02T12:00:00Z",
  "data": [
    {
      "category_name": "Canadian Experience Class",
      "predictions": [
        {
          "rank": 1,
          "predicted_date": "2025-08-16",
          "predicted_crs_score": 467,
          "confidence_score": 78.5,
          "model_used": "XGBoost"
        }
      ]
    }
  ]
}
```

---

## 🚨 Security

### Production Security Checklist

- [ ] Set `DEBUG = False`
- [ ] Use strong `SECRET_KEY`
- [ ] Enable HTTPS
- [ ] Set proper `ALLOWED_HOSTS`
- [ ] Use PostgreSQL (not SQLite)
- [ ] Regular security updates
- [ ] Enable CSRF protection
- [ ] Use environment variables for secrets

### Basic Security Settings

```python
# settings.py for production
DEBUG = False
ALLOWED_HOSTS = ['yourdomain.com', 'www.yourdomain.com']

SECURE_SSL_REDIRECT = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
CSRF_COOKIE_SECURE = True
SESSION_COOKIE_SECURE = True
```

---

## 📈 Monitoring

### Key Metrics to Monitor

1. **Prediction Accuracy**: Check against actual draws
2. **Response Times**: API and page load times
3. **Data Freshness**: Last prediction update time
4. **Error Rates**: Failed predictions or API calls
5. **User Traffic**: Page views and API usage

### Health Check Endpoint

```python
# Custom health check
def health_check(request):
    try:
        # Check database
        from predictor.models import ExpressEntryDraw
        draw_count = ExpressEntryDraw.objects.count()
        
        # Check predictions
        from predictor.models import PreComputedPrediction
        prediction_count = PreComputedPrediction.objects.filter(is_active=True).count()
        
        return JsonResponse({
            'status': 'healthy',
            'draws': draw_count,
            'predictions': prediction_count,
            'timestamp': timezone.now().isoformat()
        })
    except Exception as e:
        return JsonResponse({'status': 'unhealthy', 'error': str(e)}, status=500)
```

---

## 📞 Support

### Getting Help

1. **Check logs**: `python3 manage.py runserver` output
2. **Review this guide**: Most common issues covered
3. **Check GitHub issues**: https://github.com/choxos/ExpressEntryPredictor/issues
4. **Test with minimal setup**: Use quick start section

### Useful Commands Reference

```bash
# Data Management
python3 manage.py load_draw_data           # Load historical data
python3 manage.py compute_predictions      # Generate all predictions
python3 manage.py setup_initial_data       # Setup ML models

# Model Operations
python3 manage.py evaluate_models          # Check model performance
python3 manage.py run_predictions --model ensemble --steps 5

# System Management
python3 manage.py check                    # System health check
python3 manage.py migrate                  # Database updates
python3 manage.py collectstatic           # Collect static files

# Development
python3 manage.py runserver 8002          # Development server
python3 manage.py shell                   # Django shell
python3 manage.py createsuperuser         # Create admin user
```

---

## 🎯 Summary

Your Express Entry Predictor is now equipped with:

✅ **8 sophisticated ML models** with automatic selection  
✅ **Pre-computed predictions** for lightning-fast performance  
✅ **Clean, modern interface** with Home | Predictions | Analytics  
✅ **Admin-friendly workflow** for data updates  
✅ **Production-ready deployment** options  
✅ **Comprehensive API** for external integrations  
✅ **Automated workflows** for maintenance  

**The system is designed to be set-and-forget** - just update your CSV when new draws are published, run the prediction update command, and you're done! 🇨🇦🚀

---

*Last Updated: August 2025*  
*Express Entry Predictor v2.0 - Pre-computed Prediction System* 