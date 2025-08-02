# 🚀 VPS Deployment Guide - Express Entry Predictor

Complete guide to deploy the Express Entry Predictor on your VPS with PostgreSQL and Nginx.

## 📋 Server Configuration

- **Domain**: expressentry.xeradb.com
- **Port**: 8010 (internal application port)
- **Database**: PostgreSQL
- **Database Name**: eep_production
- **Database User**: eep_user
- **Database Password**: Choxos10203040

## 🔧 Prerequisites

### 1. Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install System Dependencies
```bash
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    postgresql \
    postgresql-contrib \
    nginx \
    git \
    curl \
    supervisor \
    ufw \
    certbot \
    python3-certbot-nginx
```

## 🗄️ Database Setup

### 1. Configure PostgreSQL
```bash
# Switch to postgres user
sudo -u postgres psql

# Create database and user
CREATE DATABASE eep_production;
CREATE USER eep_user WITH PASSWORD 'Choxos10203040';

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE eep_production TO eep_user;
ALTER USER eep_user CREATEDB;

# Exit PostgreSQL
\q
```

### 2. Configure PostgreSQL Access
```bash
# Edit PostgreSQL configuration
sudo nano /etc/postgresql/*/main/pg_hba.conf

# Add this line after the existing lines:
local   eep_production    eep_user                                md5

# Restart PostgreSQL
sudo systemctl restart postgresql
sudo systemctl enable postgresql
```

## 📁 Application Deployment

### 1. Create Application Directory
```bash
sudo mkdir -p /var/www/expressentry
sudo chown $USER:$USER /var/www/expressentry
cd /var/www/expressentry
```

### 2. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/ExpressEntryPredictor.git .
```

### 3. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn psycopg2-binary
```

## ⚙️ Environment Configuration

### 1. Create Environment File
```bash
nano .env
```

Add the following content:
```env
# Django Settings
DEBUG=False
SECRET_KEY=your_very_long_secret_key_change_this_in_production
ALLOWED_HOSTS=expressentry.xeradb.com,www.expressentry.xeradb.com,127.0.0.1,localhost

# Database Configuration
DATABASE_URL=postgresql://eep_user:Choxos10203040@localhost:5432/eep_production

# Time Zone
TIME_ZONE=America/Toronto

# Email Configuration (optional)
EMAIL_BACKEND=django.core.mail.backends.console.EmailBackend

# Static Files
STATIC_ROOT=/var/www/expressentry/staticfiles
MEDIA_ROOT=/var/www/expressentry/media

# Security Settings
SECURE_SSL_REDIRECT=True
SECURE_HSTS_SECONDS=31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS=True
SECURE_HSTS_PRELOAD=True
SECURE_CONTENT_TYPE_NOSNIFF=True
SECURE_BROWSER_XSS_FILTER=True
SECURE_REFERRER_POLICY=same-origin
SESSION_COOKIE_SECURE=True
CSRF_COOKIE_SECURE=True
```

### 2. Generate Secret Key
```bash
python3 -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```
Replace `your_very_long_secret_key_change_this_in_production` with the generated key.

## 🔄 Database Migration & Setup

### 1. Run Migrations
```bash
source venv/bin/activate
python manage.py makemigrations
python manage.py migrate
```

### 2. Create Superuser
```bash
python manage.py createsuperuser
```

### 3. Load Initial Data
```bash
# Load Express Entry draw data
python manage.py load_draw_data

# Set up initial data (categories, models)
python manage.py setup_initial_data

# Generate initial predictions
python manage.py compute_predictions --predictions=10

# Create sample data if needed
python manage.py create_sample_data
```

### 4. Collect Static Files
```bash
python manage.py collectstatic --noinput
```

## 🌐 Nginx Configuration

### 1. Create Nginx Configuration
```bash
sudo nano /etc/nginx/sites-available/expressentry
```

Add the following content:
```nginx
server {
    listen 80;
    server_name expressentry.xeradb.com www.expressentry.xeradb.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name expressentry.xeradb.com www.expressentry.xeradb.com;

    # SSL Configuration (will be configured by Certbot)
    ssl_certificate /etc/letsencrypt/live/expressentry.xeradb.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/expressentry.xeradb.com/privkey.pem;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # Static files
    location /static/ {
        alias /var/www/expressentry/staticfiles/;
        expires 30d;
        add_header Cache-Control "public, no-transform";
    }

    # Media files
    location /media/ {
        alias /var/www/expressentry/media/;
        expires 30d;
        add_header Cache-Control "public, no-transform";
    }

    # Main application
    location / {
        proxy_pass http://127.0.0.1:8010;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

    # File size limits
    client_max_body_size 10M;
}
```

### 2. Enable Site
```bash
sudo ln -s /etc/nginx/sites-available/expressentry /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 🔒 SSL Certificate Setup

### 1. Obtain SSL Certificate
```bash
sudo certbot --nginx -d expressentry.xeradb.com -d www.expressentry.xeradb.com
```

### 2. Set Up Auto-Renewal
```bash
sudo crontab -e
```
Add this line:
```bash
0 12 * * * /usr/bin/certbot renew --quiet
```

## 🔄 Process Management with Supervisor

### 1. Create Supervisor Configuration
```bash
sudo nano /etc/supervisor/conf.d/expressentry.conf
```

Add the following content:
```ini
[program:expressentry]
command=/var/www/expressentry/venv/bin/gunicorn expressentry_predictor.wsgi:application --bind 127.0.0.1:8010 --workers 3
directory=/var/www/expressentry
user=www-data
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/supervisor/expressentry.log
stderr_logfile=/var/log/supervisor/expressentry_error.log
environment=PATH="/var/www/expressentry/venv/bin"
```

### 2. Update Supervisor
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start expressentry
sudo supervisorctl status
```

## 🔥 Firewall Configuration

```bash
# Enable UFW
sudo ufw enable

# Allow SSH
sudo ufw allow OpenSSH

# Allow HTTP and HTTPS
sudo ufw allow 'Nginx Full'

# Check status
sudo ufw status
```

## 📊 Set Up Automated Tasks

### 1. Create Cron Jobs for Data Updates
```bash
crontab -e
```

Add these lines:
```bash
# Update predictions daily at 6 AM
0 6 * * * cd /var/www/expressentry && /var/www/expressentry/venv/bin/python manage.py compute_predictions --predictions=10 >> /var/log/eep_cron.log 2>&1

# Collect economic data weekly (Sunday at 2 AM)
0 2 * * 0 cd /var/www/expressentry && /var/www/expressentry/venv/bin/python manage.py collect_economic_data >> /var/log/eep_cron.log 2>&1

# Clean up old cached data daily at midnight
0 0 * * * cd /var/www/expressentry && /var/www/expressentry/venv/bin/python manage.py shell -c "from predictor.models import PredictionCache; PredictionCache.objects.filter(created_at__lt=timezone.now()-timedelta(days=7)).delete()" >> /var/log/eep_cron.log 2>&1
```

## 📝 Logging Configuration

### 1. Create Log Directory
```bash
sudo mkdir -p /var/log/expressentry
sudo chown www-data:www-data /var/log/expressentry
```

### 2. Configure Django Logging
Add to your Django settings:
```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': '/var/log/expressentry/django.log',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
```

## 🔍 Monitoring & Maintenance

### 1. Check Application Status
```bash
# Check Supervisor status
sudo supervisorctl status

# Check Nginx status
sudo systemctl status nginx

# Check database connection
sudo -u postgres psql -d eep_production -c "SELECT COUNT(*) FROM predictor_expressentrydraw;"

# View application logs
sudo tail -f /var/log/supervisor/expressentry.log
```

### 2. Restart Services
```bash
# Restart application
sudo supervisorctl restart expressentry

# Restart Nginx
sudo systemctl restart nginx

# Restart PostgreSQL
sudo systemctl restart postgresql
```

### 3. Update Application
```bash
cd /var/www/expressentry
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py collectstatic --noinput
sudo supervisorctl restart expressentry
```

## 🚀 Performance Optimization

### 1. Database Optimization
```sql
-- Connect to database
sudo -u postgres psql -d eep_production

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_draws_date ON predictor_expressentrydraw(date);
CREATE INDEX IF NOT EXISTS idx_draws_category ON predictor_expressentrydraw(category_id);
CREATE INDEX IF NOT EXISTS idx_predictions_active ON predictor_precomputedprediction(is_active);
CREATE INDEX IF NOT EXISTS idx_predictions_rank ON predictor_precomputedprediction(prediction_rank);
```

### 2. Nginx Optimization
Add to Nginx configuration:
```nginx
# Enable gzip compression
gzip on;
gzip_comp_level 6;
gzip_types text/css application/javascript application/json;

# Enable browser caching
location ~* \.(css|js|png|jpg|jpeg|gif|ico|svg)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

## 🔧 Troubleshooting

### Common Issues:

1. **Application won't start**:
   ```bash
   sudo supervisorctl tail expressentry stderr
   ```

2. **Database connection errors**:
   ```bash
   sudo -u postgres psql -d eep_production -c "\dt"
   ```

3. **Static files not loading**:
   ```bash
   python manage.py collectstatic --noinput
   sudo chown -R www-data:www-data /var/www/expressentry/staticfiles
   ```

4. **SSL certificate issues**:
   ```bash
   sudo certbot certificates
   sudo certbot renew --dry-run
   ```

## 📞 Support

### Application URLs:
- **Main Site**: https://expressentry.xeradb.com
- **Admin Panel**: https://expressentry.xeradb.com/admin/
- **API Documentation**: https://expressentry.xeradb.com/api-docs/
- **Health Check**: https://expressentry.xeradb.com/api/health/

### Log Locations:
- Application logs: `/var/log/supervisor/expressentry.log`
- Nginx logs: `/var/log/nginx/access.log` and `/var/log/nginx/error.log`
- Django logs: `/var/log/expressentry/django.log`
- Cron logs: `/var/log/eep_cron.log`

### Commands Quick Reference:
```bash
# Check status
sudo supervisorctl status
sudo systemctl status nginx
sudo systemctl status postgresql

# Restart services
sudo supervisorctl restart expressentry
sudo systemctl restart nginx

# View logs
sudo tail -f /var/log/supervisor/expressentry.log
sudo tail -f /var/log/nginx/error.log

# Update predictions
cd /var/www/expressentry && source venv/bin/activate
python manage.py compute_predictions --force --predictions=10

# Access database
sudo -u postgres psql -d eep_production
```

---

🎉 **Deployment Complete!** Your Express Entry Predictor should now be running at https://expressentry.xeradb.com 