# Express Entry Predictor - Deployment Guide

This guide provides step-by-step instructions for deploying the Express Entry Predictor application.

## 🚀 Quick Start (Minimal Setup)

For a quick demo with basic functionality:

### 1. Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ExpressEntryPredictor.git
cd ExpressEntryPredictor

# Install minimal dependencies
pip install django djangorestframework django-cors-headers pandas numpy

# Setup database
python manage.py migrate
python manage.py setup_initial_data
python manage.py load_draw_data --file data/draw_data.csv

# Create admin user
python manage.py createsuperuser

# Run server
python manage.py runserver
```

### 2. Access the Application

- **Web Interface**: http://127.0.0.1:8000/
- **Admin Panel**: http://127.0.0.1:8000/admin/
- **API**: http://127.0.0.1:8000/api/

## 📦 Full Installation (All Features)

For complete functionality including all ML models:

### 1. Install All Dependencies

```bash
# Install full requirements
pip install -r requirements.txt

# Or install specific packages
pip install scikit-learn xgboost tensorflow statsmodels plotly seaborn matplotlib
```

### 2. Verify Installation

```bash
python -c "import sklearn, xgboost, tensorflow, statsmodels; print('All ML libraries installed successfully')"
```

## 🐋 Docker Deployment

### 1. Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN python manage.py collectstatic --noinput
RUN python manage.py migrate
RUN python manage.py setup_initial_data
RUN python manage.py load_draw_data --file data/draw_data.csv

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
```

### 2. Build and Run

```bash
docker build -t express-entry-predictor .
docker run -p 8000:8000 express-entry-predictor
```

## ☁️ Cloud Deployment

### Heroku Deployment

1. **Install Heroku CLI** and login:
```bash
heroku login
```

2. **Create Heroku app**:
```bash
heroku create your-app-name
```

3. **Add environment variables**:
```bash
heroku config:set DEBUG=False
heroku config:set SECRET_KEY=your-secret-key
heroku config:set DATABASE_URL=postgres://...
```

4. **Deploy**:
```bash
git push heroku main
heroku run python manage.py migrate
heroku run python manage.py setup_initial_data
heroku run python manage.py load_draw_data --file data/draw_data.csv
```

### AWS Deployment

1. **EC2 Instance Setup**:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv nginx -y

# Create virtual environment
python3 -m venv express_entry_env
source express_entry_env/bin/activate

# Clone and setup project
git clone https://github.com/your-username/ExpressEntryPredictor.git
cd ExpressEntryPredictor
pip install -r requirements.txt
```

2. **Configure Nginx**:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /static/ {
        alias /path/to/ExpressEntryPredictor/staticfiles/;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

3. **Run with Gunicorn**:
```bash
pip install gunicorn
gunicorn expressentry_predictor.wsgi:application --bind 127.0.0.1:8000
```

## 🗄️ Database Configuration

### PostgreSQL Setup

1. **Install PostgreSQL**:
```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib

# macOS
brew install postgresql
```

2. **Create database**:
```sql
CREATE DATABASE expressentry_db;
CREATE USER expressentry_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE expressentry_db TO expressentry_user;
```

3. **Update settings**:
```python
# settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'expressentry_db',
        'USER': 'expressentry_user',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

## 🔧 Configuration

### Environment Variables

Create `.env` file:
```env
DEBUG=False
SECRET_KEY=your-super-secret-key-here
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
ALLOWED_HOSTS=your-domain.com,127.0.0.1

# Optional
REDIS_URL=redis://localhost:6379/0
IRCC_API_KEY=your-api-key
STATSCAN_API_KEY=your-api-key
```

### Production Settings

```python
# settings_production.py
import os
from .settings import *

DEBUG = False
ALLOWED_HOSTS = ['your-domain.com', 'www.your-domain.com']

# Security settings
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

# Static files
STATIC_ROOT = '/var/www/static/'
MEDIA_ROOT = '/var/www/media/'
```

## 📊 Performance Optimization

### Caching Setup

```python
# settings.py
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# Cache timeout settings
CACHE_TIMEOUT = 60 * 15  # 15 minutes
```

### Database Optimization

```python
# settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'expressentry_db',
        'USER': 'expressentry_user',
        'PASSWORD': 'password',
        'HOST': 'localhost',
        'PORT': '5432',
        'OPTIONS': {
            'MAX_CONNS': 20,
            'CONN_MAX_AGE': 600,
        }
    }
}
```

## 🔄 Automation & Monitoring

### Celery for Background Tasks

```python
# celery.py
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'expressentry_predictor.settings')

app = Celery('expressentry_predictor')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# Periodic task for updating predictions
from celery.schedules import crontab

app.conf.beat_schedule = {
    'update-predictions': {
        'task': 'predictor.tasks.update_predictions',
        'schedule': crontab(hour=0, minute=0),  # Daily at midnight
    },
}
```

### Monitoring with Django Extensions

```bash
pip install django-extensions
pip install werkzeug

# settings.py
INSTALLED_APPS += ['django_extensions']

# Run with profiling
python manage.py runserver_plus --print-sql
```

## 🛡️ Security

### SSL Certificate (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Firewall Configuration

```bash
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw enable
```

## 📋 Maintenance

### Regular Tasks

```bash
# Update predictions
python manage.py generate_predictions

# Backup database
pg_dump expressentry_db > backup_$(date +%Y%m%d).sql

# Clean old logs
find /var/log -name "*.log" -type f -mtime +30 -delete

# Update dependencies
pip list --outdated
pip install -U package_name
```

### Health Checks

```python
# views.py
from django.http import JsonResponse
from django.db import connection

def health_check(request):
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        return JsonResponse({'status': 'healthy'})
    except Exception as e:
        return JsonResponse({'status': 'unhealthy', 'error': str(e)}, status=500)
```

## 🚨 Troubleshooting

### Common Issues

1. **Missing ML Libraries**:
```bash
# Error: ModuleNotFoundError: No module named 'sklearn'
pip install scikit-learn
```

2. **Database Connection Error**:
```bash
# Check PostgreSQL status
sudo systemctl status postgresql
# Restart if needed
sudo systemctl restart postgresql
```

3. **Static Files Not Loading**:
```bash
python manage.py collectstatic --clear
```

4. **Memory Issues with ML Models**:
```python
# Reduce model complexity in settings
MODEL_SETTINGS = {
    'RF_N_ESTIMATORS': 50,  # Reduce from 100
    'LSTM_EPOCHS': 25,      # Reduce from 50
}
```

### Logs and Debugging

```bash
# Django logs
tail -f debug.log

# Nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log

# System logs
journalctl -u your-service-name -f
```

## 📞 Support

For deployment issues:

1. Check the logs first
2. Verify all dependencies are installed
3. Ensure database is properly configured
4. Check firewall and network settings
5. Review the [main README](README.md) for additional help

---

**Note**: This guide assumes Ubuntu/Debian for Linux instructions. Adjust commands for other distributions as needed. 