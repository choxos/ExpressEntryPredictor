## 🌐 VPS Production Deployment

*Deploy to expressentry.xeradb.com with PostgreSQL and enhanced data management*

### 🔍 **Pre-Deployment Checklist**

Before starting, ensure you have:
- ✅ **VPS Access**: SSH access to your server with sudo privileges
- ✅ **Domain Setup**: DNS pointing `expressentry.xeradb.com` to your VPS IP
- ✅ **Database Credentials**: Ready to use `eep_production`, `eep_user`, `Choxos10203040`
- ✅ **Latest Code**: All recent fixes including prediction bugs (commit: 5639741+)
- ✅ **Port 8010**: Confirmed available for your application
- ✅ **SSL Ready**: Plan to use Let's Encrypt for HTTPS certificates

### ⚠️ **Security Requirements Met**:
- ✅ **Strong Secret Key**: Auto-generated 50+ character key
- ✅ **HTTPS Enforcement**: SSL redirect and HSTS headers
- ✅ **Secure Cookies**: HTTPOnly and Secure flags enabled
- ✅ **Environment Variables**: All secrets in `.env` file (chmod 600)

### 1. VPS Prerequisites

**Server Requirements:**
- Ubuntu 20.04+ or CentOS 8+
- Python 3.12+
- PostgreSQL 12+
- Nginx
- At least 2GB RAM, 20GB storage

**User Configuration:**
- VPS User: `xeradb`
- Group: `xeradb`
- Domain: `expressentry.xeradb.com`
- Application Port: `8010`

### 2. Initial VPS Setup

```bash
# SSH into your VPS
ssh xeradb@your-vps-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3.12 python3.12-venv python3-pip git nginx postgresql postgresql-contrib
sudo apt install -y python3.12-dev libpq-dev build-essential

# Create application directory
sudo mkdir -p /var/www/eep
sudo chown xeradb:xeradb /var/www/eep
cd /var/www/eep
```

### 3. PostgreSQL Database Setup

```bash
# Switch to postgres user and create database
sudo -u postgres psql

-- In PostgreSQL prompt:
CREATE DATABASE eep_production;
CREATE USER eep_user WITH PASSWORD 'Choxos10203040';
GRANT ALL PRIVILEGES ON DATABASE eep_production TO eep_user;
GRANT ALL ON SCHEMA public TO eep_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO eep_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO eep_user;
ALTER USER eep_user CREATEDB;
\q

# Test database connection
psql -h localhost -U eep_user -d eep_production -W
# Enter password: Choxos10203040
# If successful, type \q to quit
```

### 4. Application Deployment

```bash
# Clone repository
cd /var/www/eep
git clone https://github.com/choxos/ExpressEntryPredictor.git .

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install additional production dependencies
pip install gunicorn psycopg2-binary

# Create production environment file
cat > .env << EOF
# Database Configuration
DATABASE_URL=postgresql://eep_user:Choxos10203040@localhost:5432/eep_production

# Django Settings - Generate secure secret key
SECRET_KEY=$(python3 -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())")
DEBUG=False
ALLOWED_HOSTS=expressentry.xeradb.com,www.expressentry.xeradb.com

# Security Settings (Production)
SECURE_SSL_REDIRECT=True
SECURE_PROXY_SSL_HEADER=HTTP_X_FORWARDED_PROTO,https
CSRF_COOKIE_SECURE=True
SESSION_COOKIE_SECURE=True
SECURE_HSTS_SECONDS=31536000

# Static Files
STATIC_ROOT=/var/www/eep/staticfiles
MEDIA_ROOT=/var/www/eep/media
EOF

# Set proper permissions
chmod 600 .env
```

### 5. Database Migration & Initial Data

```bash
# Activate virtual environment
source /var/www/eep/venv/bin/activate
cd /var/www/eep

# Run migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic --noinput

# Create static directory if it doesn't exist
mkdir -p /var/www/eep/static
mkdir -p /var/www/eep/staticfiles

# Set proper permissions
chown -R xeradb:xeradb /var/www/eep/static
chown -R xeradb:xeradb /var/www/eep/staticfiles

# Load initial data and setup models
python manage.py setup_initial_data

# Load historical draw data
python manage.py load_draw_data

# Populate enhanced historical data (economic, political, pool, PNP)
python manage.py populate_historical_data --clear

# Generate initial predictions with all enhanced features
python manage.py compute_predictions --force

# Create admin superuser
python manage.py createsuperuser
# Follow prompts to create admin account
```

### 6. Systemd Service Configuration

```bash
# Create systemd service file
sudo tee /etc/systemd/system/expressentry.service > /dev/null << EOF
[Unit]
Description=Express Entry Predictor Django App
After=network.target postgresql.service

[Service]
Type=exec
User=xeradb
Group=xeradb
WorkingDirectory=/var/www/eep
Environment=PATH=/var/www/eep/venv/bin
EnvironmentFile=/var/www/eep/.env
ExecStart=/var/www/eep/venv/bin/gunicorn --workers 3 --bind 127.0.0.1:8010 expressentry_predictor.wsgi:application
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable expressentry
sudo systemctl start expressentry

# Check service status
sudo systemctl status expressentry
```

### 7. Nginx Configuration

**Initial HTTP-only Configuration (before SSL):**
```bash
# Create initial Nginx configuration (HTTP only)
sudo tee /etc/nginx/sites-available/expressentry > /dev/null << EOF
server {
    listen 80;
    server_name expressentry.xeradb.com www.expressentry.xeradb.com;
    
    # Static files
    location /static/ {
        alias /var/www/eep/staticfiles/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    location /media/ {
        alias /var/www/eep/media/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Main application
    location / {
        proxy_pass http://127.0.0.1:8010;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_redirect off;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Health check endpoint
    location /health/ {
        proxy_pass http://127.0.0.1:8010;
        access_log off;
    }
}
EOF

# Enable site and remove default
sudo ln -sf /etc/nginx/sites-available/expressentry /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
sudo nginx -t

# Start nginx
sudo systemctl enable nginx
sudo systemctl restart nginx
```

**Note**: After SSL setup with Certbot (step 8), the configuration will be automatically updated to include SSL redirects and security headers.

### 8. SSL Certificate Setup (Let's Encrypt)

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d expressentry.xeradb.com

# Test auto-renewal
sudo certbot renew --dry-run

# The certbot will automatically update your nginx configuration
```

### 9. **Post-Deployment Verification** ✅

```bash
# 1. Verify application is running
sudo systemctl status expressentry
curl -I http://localhost:8010/

# 2. Test database connection
cd /var/www/eep
source venv/bin/activate
python manage.py shell -c "from django.db import connection; connection.ensure_connection(); print('✅ Database connected')"

# 3. Check predictions are working
python manage.py shell -c "from predictor.models import PreComputedPrediction; print(f'Active predictions: {PreComputedPrediction.objects.filter(is_active=True).count()}')"

# 4. Test website URLs
curl -I https://expressentry.xeradb.com/
curl -I https://expressentry.xeradb.com/api/stats/
curl -I https://expressentry.xeradb.com/admin/

# 5. Run security check
python manage.py check --deploy

# 6. Monitor logs
sudo journalctl -u expressentry -f
```

### 🎯 **Expected Results**:
- ✅ **Application Status**: Active and running
- ✅ **Database**: Connected and populated with 358+ draws
- ✅ **Predictions**: 150+ active predictions across categories  
- ✅ **Website**: Loads at https://expressentry.xeradb.com
- ✅ **API**: Returns stats and predictions
- ✅ **Security**: No critical warnings in deployment check
- ✅ **SSL**: A+ rating on SSL Labs test

---

## 📊 Data Update Procedures

### **Manual Data Updates**

When you have new draw data or need to refresh economic indicators:

```bash
# Basic data update (draw data + economic indicators)
./update_data.sh

# Full update including new predictions
./update_data.sh --with-predictions

# Clear and reload all data
./update_data.sh --clear-data --with-predictions

# View help
./update_data.sh --help
```

### **Automated Update Script**

The `update_data.sh` script handles all data updates automatically:

**What it does:**
- ✅ Loads new Express Entry draws from `data/draw_data.csv`
- ✅ Updates economic indicators (last 3 months)
- ✅ Syncs new IRCC draws automatically  
- ✅ Computes new predictions (optional)
- ✅ Restarts web services (on VPS)
- ✅ Clears cache

**Usage Examples:**
```bash
# Quick update (no predictions)
./update_data.sh

# Full update with new predictions
./update_data.sh --with-predictions

# Fresh start (clear existing data)
./update_data.sh --clear-data --with-predictions
```

### **Manual Command Reference**

If you need to run individual commands:

```bash
# Load new draw data
python manage.py load_draw_data --file data/draw_data.csv

# Update economic indicators  
python manage.py collect_economic_data --start-date 2024-05-01 --end-date 2024-08-01

# Sync latest IRCC draws
python manage.py sync_ircc_draws --days-back 30

# Compute new predictions
python manage.py compute_predictions --force

# Clear cache
python manage.py shell -c "from django.core.cache import cache; cache.clear()"
```

### 6. Database Backup Strategy

```bash
# Create backup script
cat > /var/www/eep/backup_db.sh << 'EOF'
#!/bin/bash

# Database backup script
BACKUP_DIR="/var/www/eep/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="eep_production_backup_$DATE.sql"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create backup
pg_dump -h localhost -U eep_user -d eep_production > "$BACKUP_DIR/$BACKUP_FILE"

# Compress backup
gzip "$BACKUP_DIR/$BACKUP_FILE"

# Keep only last 7 backups
find $BACKUP_DIR -name "eep_production_backup_*.sql.gz" -mtime +7 -delete

echo "✅ Database backup created: $BACKUP_FILE.gz"
EOF

# Make executable and setup cron
chmod +x /var/www/eep/backup_db.sh

# Add to crontab (daily backup at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /var/www/eep/backup_db.sh") | crontab -
```

---

## 🔄 Maintenance & Updates

### Daily Operations

**For ongoing maintenance, you only need:**

1. **Check website is running**: Visit your URL
2. **Update predictions when new draws published**:
   ```bash
   # Update CSV with new data, then:
   python manage.py load_draw_data
   python manage.py compute_predictions --force
   ```

### Weekly Tasks

```bash
# 1. Backup database
python manage.py dumpdata > backup_$(date +%Y%m%d).json

# 2. Update system and code
git pull origin main
pip install -r requirements.txt --upgrade
python manage.py migrate
python manage.py compute_predictions --force

# 3. Sync latest IRCC draws
python manage.py sync_ircc_draws --days-back 7
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
python manage.py shell -c "from predictor.models import ExpressEntryDraw; print(f'Draws: {ExpressEntryDraw.objects.count()}')"

# Check if categories exist
python manage.py shell -c "from predictor.models import DrawCategory; print(f'Categories: {DrawCategory.objects.count()}')"

# Regenerate predictions
python manage.py compute_predictions --force
```

**3. Database Issues:**
```bash
# Reset migrations (if needed)
rm predictor/migrations/000*.py
python manage.py makemigrations predictor
python manage.py migrate
```

**4. Performance Issues:**
```bash
# Reduce prediction load for testing
python manage.py compute_predictions --predictions 3 --force

# Generate predictions for specific category only
python manage.py compute_predictions --category "Canadian Experience Class" --force
```

### Performance Optimization

**Database Optimization:**
```bash
# Use PostgreSQL for production (already configured in this guide)
# Database settings are in .env file: DATABASE_URL=postgresql://...

# For Redis caching (optional enhancement):
sudo apt install redis-server
pip install django-redis

# Add to .env:
# CACHE_BACKEND=redis://127.0.0.1:6379/1
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
python manage.py load_draw_data           # Load historical data
python manage.py compute_predictions      # Generate all predictions
python manage.py setup_initial_data       # Setup ML models

# Enhanced Data Management
python manage.py sync_ircc_draws          # Sync latest IRCC draws
python manage.py populate_historical_data # Populate economic/political data
python manage.py collect_economic_data    # Update economic indicators

# System Management
python manage.py check                    # System health check
python manage.py migrate                  # Database updates
python manage.py collectstatic           # Collect static files

# Development
python manage.py runserver 8002          # Development server
python manage.py shell                   # Django shell
python manage.py createsuperuser         # Create admin user
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

---

## 🔧 VPS Monitoring & Maintenance

### 1. Application Monitoring

**Check Application Status:**
```bash
# Check if application is running
sudo systemctl status expressentry

# View application logs
sudo journalctl -u expressentry -f

# Check disk usage
df -h

# Check memory usage
free -h

# Check database connections
sudo -u postgres psql -c "SELECT count(*) FROM pg_stat_activity WHERE datname='eep_production';"
```

**Health Check Script:**
```bash
# Create monitoring script
cat > /var/www/eep/monitor.sh << 'EOF'
#!/bin/bash

echo "🔍 Express Entry Predictor Health Check"
echo "========================================"

# Check web service
if curl -sf http://localhost:8010/health/ > /dev/null; then
    echo "✅ Web service: Running"
else
    echo "❌ Web service: Down"
fi

# Check database
if sudo -u postgres psql -d eep_production -c '\q' 2>/dev/null; then
    echo "✅ Database: Connected"
else
    echo "❌ Database: Connection failed"
fi

# Check disk space
DISK_USAGE=$(df /var/www/eep | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -lt 80 ]; then
    echo "✅ Disk space: ${DISK_USAGE}% used"
else
    echo "⚠️ Disk space: ${DISK_USAGE}% used (High)"
fi

# Check predictions count
PRED_COUNT=$(sudo -u xeradb /var/www/eep/venv/bin/python /var/www/eep/manage.py shell -c "from predictor.models import PreComputedPrediction; print(PreComputedPrediction.objects.filter(is_active=True).count())" 2>/dev/null)
echo "📊 Active predictions: $PRED_COUNT"

# Check last prediction update
LAST_UPDATE=$(sudo -u xeradb /var/www/eep/venv/bin/python /var/www/eep/manage.py shell -c "from predictor.models import PreComputedPrediction; from django.utils import timezone; import datetime; latest = PreComputedPrediction.objects.filter(is_active=True).order_by('-created_at').first(); print((timezone.now() - latest.created_at).days if latest else 'N/A')" 2>/dev/null)
echo "📅 Last prediction update: $LAST_UPDATE days ago"

echo "========================================"
EOF

chmod +x /var/www/eep/monitor.sh
```

### 2. Performance Optimization

**Database Optimization:**
```bash
# Optimize PostgreSQL for production
sudo -u postgres psql -d eep_production << 'EOF'
-- Update table statistics
ANALYZE;

-- Vacuum to reclaim space
VACUUM;

-- Check slow queries
SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;
EOF

# Add database indices for better performance
sudo -u xeradb /var/www/eep/venv/bin/python /var/www/eep/manage.py shell -c "
from django.db import connection
cursor = connection.cursor()
cursor.execute('CREATE INDEX IF NOT EXISTS idx_draws_date ON predictor_expressentry_draw(date);')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_category ON predictor_precomputed_prediction(category_id);')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_active ON predictor_precomputed_prediction(is_active) WHERE is_active = true;')
print('Database indices created successfully')
"
```

### 3. Regular Maintenance Tasks

**Weekly Maintenance Script:**
```bash
# Create weekly maintenance script
cat > /var/www/eep/weekly_maintenance.sh << 'EOF'
#!/bin/bash

echo "🔧 Weekly Maintenance - $(date)"
echo "================================"

cd /var/www/eep
source venv/bin/activate

# 1. Clean up old predictions (keep last 100 per category)
echo "🧹 Cleaning old predictions..."
python manage.py shell -c "
from predictor.models import PreComputedPrediction, DrawCategory
for category in DrawCategory.objects.all():
    old_predictions = PreComputedPrediction.objects.filter(
        category=category, is_active=False
    ).order_by('-created_at')[100:]
    count = len(old_predictions)
    if count > 0:
        for pred in old_predictions:
            pred.delete()
        print(f'Cleaned {count} old predictions for {category.name}')
"

# 2. Update database statistics
echo "📊 Updating database statistics..."
sudo -u postgres psql -d eep_production -c "ANALYZE; VACUUM;"

# 3. Clean log files
echo "📝 Cleaning old logs..."
sudo journalctl --vacuum-time=30d

# 4. Check SSL certificate expiry
echo "🔒 Checking SSL certificate..."
openssl x509 -in /etc/letsencrypt/live/expressentry.xeradb.com/cert.pem -noout -dates 2>/dev/null || echo "SSL check skipped"

# 5. Restart services for fresh memory
echo "🔄 Restarting services..."
sudo systemctl restart expressentry
sudo systemctl reload nginx

echo "✅ Weekly maintenance completed"
EOF

chmod +x /var/www/eep/weekly_maintenance.sh

# Schedule weekly maintenance (Sundays at 3 AM)
(crontab -l 2>/dev/null; echo "0 3 * * 0 /var/www/eep/weekly_maintenance.sh >> /var/log/expressentry_maintenance.log 2>&1") | crontab -
```

### 4. Troubleshooting Common Issues

**Application Won't Start:**
```bash
# Check logs
sudo journalctl -u expressentry -n 50

# Check if port is occupied
sudo netstat -tlnp | grep :8010

# Check environment file
cat /var/www/eep/.env

# Test database connection manually
cd /var/www/eep
source venv/bin/activate
python manage.py check --database default
```

**Static Files Warning Fix:**
```bash
# If you see "The directory '/var/www/eep/static' does not exist" warning:
cd /var/www/eep
source venv/bin/activate

# Create missing static directories
mkdir -p /var/www/eep/static
mkdir -p /var/www/eep/staticfiles

# Set proper permissions
sudo chown -R xeradb:xeradb /var/www/eep/static
sudo chown -R xeradb:xeradb /var/www/eep/staticfiles

# Re-collect static files
python manage.py collectstatic --noinput

# Verify settings match directory structure
python manage.py shell -c "from django.conf import settings; print('STATIC_ROOT:', settings.STATIC_ROOT); print('STATICFILES_DIRS:', settings.STATICFILES_DIRS)"
```

**NaN Prediction Errors:**
```bash
# If you see "cannot convert float NaN to integer" errors:
cd /var/www/eep
source venv/bin/activate

# Clear problematic predictions for specific category
python manage.py clear_predictions --category "Education occupations (Version 1)" --confirm

# Regenerate with enhanced NaN handling
python manage.py compute_predictions --category "Education occupations (Version 1)" --force

# Check for remaining issues
python manage.py shell -c "from predictor.models import PreComputedPrediction; print(f'Active predictions: {PreComputedPrediction.objects.filter(is_active=True).count()}')"
```

**Database Connection Issues:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check database user permissions
sudo -u postgres psql -c "\du eep_user"

# Test connection
psql -h localhost -U eep_user -d eep_production -c "SELECT version();"
```

**SSL Certificate Issues:**
```bash
# Renew certificate manually
sudo certbot renew

# Check certificate status
sudo certbot certificates

# Test SSL configuration
sudo nginx -t
```

**Performance Issues:**
```bash
# Check resource usage
htop

# Check database performance
sudo -u postgres psql -d eep_production -c "
SELECT schemaname,tablename,attname,n_distinct,correlation 
FROM pg_stats WHERE tablename IN ('predictor_expressentry_draw', 'predictor_precomputed_prediction');
"

# Clear Django cache if using Redis
# redis-cli flushdb
```

---

## 📱 Quick Reference Commands

### Essential VPS Management Commands:

```bash
# Application Management
sudo systemctl start expressentry      # Start application
sudo systemctl stop expressentry       # Stop application  
sudo systemctl restart expressentry    # Restart application
sudo systemctl status expressentry     # Check status
sudo journalctl -u expressentry -f     # View live logs

# Database Operations
psql -h localhost -U eep_user -d eep_production  # Connect to database
sudo -u postgres pg_dump eep_production > backup.sql  # Manual backup
sudo systemctl restart postgresql      # Restart database

# Web Server
sudo nginx -t                         # Test configuration
sudo systemctl reload nginx           # Reload nginx
sudo systemctl restart nginx          # Restart nginx

# SSL Certificates
sudo certbot renew                    # Renew certificates
sudo certbot certificates             # List certificates

# Data Updates (after CSV changes)
cd /var/www/eep && source venv/bin/activate
./update_data.sh --with-predictions   # Full update with predictions
python manage.py compute_predictions --force  # Regenerate predictions only

# Monitoring
./monitor.sh                          # Health check
df -h                                 # Disk usage
free -h                               # Memory usage
```

### File Locations:
- **Application**: `/var/www/eep/`
- **Logs**: `sudo journalctl -u expressentry`
- **Database**: `eep_production` (PostgreSQL)
- **SSL Certs**: `/etc/letsencrypt/live/expressentry.xeradb.com/`
- **Nginx Config**: `/etc/nginx/sites-available/expressentry`

### Important URLs:
- **Website**: https://expressentry.xeradb.com
- **Admin Panel**: https://expressentry.xeradb.com/admin/
- **API Root**: https://expressentry.xeradb.com/api/
- **Health Check**: https://expressentry.xeradb.com/health/

---

## 🎯 Production Deployment Summary

Your Express Entry Predictor is now fully deployed on your VPS with:

✅ **Secure Production Setup**:
- PostgreSQL database (`eep_production`) with user `eep_user`
- Running on port 8010 as systemd service
- Nginx reverse proxy with SSL termination
- Let's Encrypt SSL certificates for HTTPS

✅ **Enhanced 87-Feature ML System**:
- Economic indicators integration
- Political context modeling  
- Pool composition analysis
- PNP activity tracking
- Advanced XGBoost and LSTM models

✅ **Data Management Workflow**:
- Automated update scripts for CSV data
- Database backup system (daily)
- Weekly maintenance automation
- Real-time monitoring tools

✅ **Production Monitoring**:
- Health check endpoints
- Performance monitoring
- Log management
- SSL certificate auto-renewal

### Next Steps:
1. **Deploy**: Follow the VPS setup instructions above
2. **Test**: Verify all endpoints work at https://expressentry.xeradb.com
3. **Monitor**: Run `./monitor.sh` regularly
4. **Update**: Use `./update_data.sh --with-predictions` when new draws are published

Your Express Entry Predictor is now enterprise-ready! 🇨🇦🚀

--- 