# Express Entry Predictor

A cutting-edge Django web application that uses advanced machine learning, temporal-priority scheduling, and dynamic interval calculation to predict Canadian Express Entry draw dates and CRS scores with unprecedented accuracy.

## 🚀 Revolutionary Features

- **🧠 Advanced AI-Powered Predictions**: Ensemble of 12+ ML models including ARIMA, Prophet, LSTM, XGBoost, Gaussian Process, and Bayesian Hierarchical models
- **⚡ Temporal-Priority System**: Intelligent scheduling based on category urgency and government policy alignment
- **🔄 Recursive Forecasting**: Each prediction builds on previous ones for enhanced accuracy
- **📊 Dynamic Interval Calculation**: Real-time analysis of historical patterns instead of hardcoded intervals  
- **🎯 Category Pooling**: Combines related category versions (e.g., Healthcare V1+V2) for unified predictions
- **📅 95% Confidence Intervals**: Provides statistical confidence for both CRS scores and predicted dates
- **📱 Mobile-Responsive UI**: Beautiful, modern interface optimized for all devices
- **🌐 RESTful API**: Complete API with caching and performance optimization
- **📈 Real-time Analytics**: Interactive dashboards with comprehensive visualizations
- **🏛️ Government Policy Integration**: Aligns with official 2025 Express Entry priorities

## 🤖 Advanced ML Architecture

### Core Prediction Models
1. **Prophet** - Advanced time series with seasonality detection
2. **LSTM Neural Networks** - Deep sequence learning with attention mechanisms  
3. **ARIMA/SARIMA** - Sophisticated temporal pattern analysis
4. **XGBoost** - High-performance gradient boosting with feature engineering
5. **Gaussian Process** - Probabilistic modeling with uncertainty quantification
6. **Bayesian Hierarchical** - Multi-level Bayesian inference
7. **Random Forest** - Robust ensemble with outlier resistance
8. **Holt-Winters** - Triple exponential smoothing for trends/seasonality
9. **VAR (Vector Autoregression)** - Multi-variate time series modeling
10. **Dynamic Linear Model** - State-space modeling with Kalman filtering
11. **Clean Linear Regression** - Outlier-robust linear modeling
12. **Small Dataset Predictor** - Specialized for categories with limited data

### Intelligent Model Selection
- **Automatic Best Model Selection**: Chooses optimal model per category based on cross-validation
- **Ensemble Weighting**: Combines multiple models using confidence-weighted averaging
- **Performance Monitoring**: Continuous evaluation and re-ranking of model accuracy

## 🛠️ Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/ExpressEntryPredictor.git
cd ExpressEntryPredictor
```

2. **Install dependencies**:
```bash
# Basic setup (SQLite, core Django features)
pip install django djangorestframework django-cors-headers pandas numpy

# For full ML capabilities (optional)
pip install -r requirements.txt
```

3. **Set up the database**:
```bash
python manage.py migrate
python manage.py setup_initial_data
python manage.py load_draw_data --file data/draw_data.csv
```

4. **Create admin user**:
```bash
python manage.py createsuperuser
```

5. **Run the development server**:
```bash
python manage.py runserver
```

6. **Access the application**:
   - Main app: http://127.0.0.1:8000/
   - Admin interface: http://127.0.0.1:8000/admin/
   - API: http://127.0.0.1:8000/api/

## 🏗️ Project Structure

```
ExpressEntryPredictor/
├── predictor/                 # Main prediction app
│   ├── models.py             # Data models
│   ├── views.py              # API views and web views
│   ├── ml_models.py          # Machine learning models
│   ├── serializers.py        # API serializers
│   └── management/commands/   # Custom Django commands
├── analytics/                # Analytics and visualization app
├── templates/                # HTML templates
├── static/                   # Static files (CSS, JS)
├── data/                     # Historical draw data
└── requirements.txt          # Python dependencies
```

## 📈 API Endpoints

### Core Endpoints

- `GET /api/stats/` - Dashboard statistics
- `GET /api/predict/` - Generate predictions for all categories
- `GET /api/predict/{category_id}/` - Generate predictions for specific category
- `GET /api/categories/` - List all draw categories
- `GET /api/draws/` - Historical draw data
- `GET /api/models/` - Available prediction models

### Analytics Endpoints

- `GET /analytics/api/charts/` - Chart data for visualizations
- `GET /analytics/api/trends/` - Trend analysis data

## 🎯 Usage Examples

### Getting Predictions via API

```python
import requests

# Get all predictions
response = requests.get('http://127.0.0.1:8000/api/predict/')
predictions = response.json()

# Get specific category predictions
response = requests.get('http://127.0.0.1:8000/api/predict/1/')
category_predictions = response.json()
```

### Using the Advanced Management Commands

```bash
# 🚀 Generate comprehensive predictions with temporal-priority system
python manage.py compute_predictions --force

# 🎯 Generate predictions for specific category (for testing)
python manage.py compute_predictions --category "Canadian Experience Class"

# 📊 Load new draw data with automatic validation
python manage.py load_draw_data --file path/to/new_data.csv

# 🔄 Clear prediction cache and force refresh
python manage.py shell -c "
from predictor.models import PredictionCache
PredictionCache.objects.all().delete()
print('All caches cleared')
"

# 📈 View comprehensive prediction logs
tail -f logs/prediction_computation_*.log
```

### Advanced Prediction Features

#### 🎯 **Temporal-Priority Methodology**
The system automatically calculates which categories are closest to their next expected draw:

```python
# Example: CEC averages 25.2 days between draws
# Last draw: July 7, 2025
# Expected next: August 1, 2025 (25 days later)
# Today: July 29, 2025
# Days until expected draw: 3 days -> HIGH PRIORITY
```

#### 📊 **Dynamic Interval Calculation** 
Instead of hardcoded intervals, the system analyzes real database patterns:

```python
# Calculates from actual draw history since 2022
intervals = {
    'Canadian Experience Class': 25.2,      # days (calculated from data)
    'Provincial Nominee Program': 29.4,     # days (calculated from data)
    'Healthcare': 35.0,                     # days (calculated from data)
    # ... dynamically calculated for all categories
}
```

#### 🔄 **Category Pooling System**
Related category versions are automatically combined:

```python
pooling_map = {
    'Healthcare occupations (Version 1)': 'Healthcare',
    'Healthcare and social services occupations (Version 1)': 'Healthcare', 
    'Healthcare and social services occupations (Version 2)': 'Healthcare',
    'French language proficiency (Version 1)': 'French',
    'French-language proficiency': 'French',
    # Results in unified "Healthcare" and "French" predictions
}
```

## 📊 Data Sources

The application uses data from:

- **IRCC Official Draw History**: canada.ca Express Entry rounds
- **Statistics Canada**: Economic indicators and employment data
- **Provincial Nominee Programs**: Individual provincial draw data
- **Historical Trends**: Pattern analysis from 2015-2024 data

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
DEBUG=True
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///db.sqlite3

# For production PostgreSQL
# DATABASE_URL=postgresql://username:password@localhost:5432/expressentry_db
```

### Production Deployment

For production deployment:

1. **Set environment variables**:
```bash
export DEBUG=False
export DATABASE_URL=postgresql://user:pass@localhost/dbname
```

2. **Collect static files**:
```bash
python manage.py collectstatic
```

3. **Use a proper web server** (e.g., gunicorn with nginx)

## 🤖 Machine Learning Details

### Model Training

Models are automatically trained when predictions are requested. To manually train:

```bash
python manage.py shell
```

```python
from predictor.ml_models import RandomForestPredictor
from predictor.models import ExpressEntryDraw
import pandas as pd

# Load data
draws = ExpressEntryDraw.objects.all()
df = pd.DataFrame([{...}])  # Convert to DataFrame

# Train model
model = RandomForestPredictor()
metrics = model.train(df)
print(f"Model accuracy: {metrics['r2']}")
```

### Revolutionary Prediction Pipeline

#### 🔍 **Phase 1: Dynamic Data Analysis**
1. **Real-time Interval Calculation**: Dynamically calculates draw intervals from database instead of hardcoded values
2. **Category Pooling**: Combines related category versions (Healthcare V1+V2, French variations) for unified analysis
3. **Temporal Urgency Assessment**: Analyzes days since last draw vs. average interval to determine category urgency

#### ⚡ **Phase 2: Temporal-Priority Scheduling**
1. **Urgency Ranking**: Categories ranked by proximity to next expected draw date
2. **Government Policy Integration**: Incorporates official 2025 Express Entry priorities (CEC prioritized, Transport eliminated)
3. **Day-of-Week Optimization**: Reserves optimal days (Wed/Thu) based on historical patterns per category
4. **Date Conflict Resolution**: Prevents unrealistic same-day draws through intelligent date reservation

#### 🤖 **Phase 3: Advanced ML Prediction**
1. **Multi-Model Training**: 12+ models trained simultaneously with automatic hyperparameter optimization
2. **Best Model Selection**: Automatic selection based on cross-validation performance per category
3. **Recursive Forecasting**: Each rank builds on previous predictions for compound accuracy
4. **Uncertainty Quantification**: 95% confidence intervals for both CRS scores and dates

#### 📊 **Phase 4: Intelligent Post-Processing**
1. **Domain-Aware Bounds**: CRS scores capped at realistic range (250-950)
2. **Date Constraint Enforcement**: No predictions before today or beyond 1 year
3. **Confidence Calibration**: Domain-specific confidence calculation combining statistical metrics with Express Entry knowledge
4. **Performance Optimization**: Cached results and optimized model selection for VPS deployment

## 📱 Frontend Features

### Home Page
- Latest predictions summary
- Quick statistics
- Recent draws table
- Feature highlights

### Predictions Page
- Detailed predictions by category
- Interactive timeline charts
- Model performance metrics
- Confidence intervals

### Dashboard
- Key performance indicators
- Historical trends
- Category breakdowns
- Interactive visualizations

### Analytics
- Deep trend analysis
- Seasonal patterns
- Moving averages
- Prediction accuracy tracking

## 🧪 Testing

```bash
# Run tests
python manage.py test

# Test specific app
python manage.py test predictor

# Test with coverage
pip install coverage
coverage run manage.py test
coverage report
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is for educational and informational purposes only. Predictions are based on historical data and statistical models and should not be considered as official immigration advice. Always consult official IRCC sources for the most current information.

## 📞 Support

For support and questions:

- Open an issue on GitHub
- Check the [EEP_guide.md](EEP_guide.md) for detailed documentation
- Review [EEP_data_source.md](EEP_data_source.md) for data source information

## ✅ Recent Major Enhancements (2025)

- ✅ **Temporal-Priority Prediction System**: Revolutionary scheduling based on category urgency
- ✅ **Dynamic Interval Calculation**: Real-time analysis replacing hardcoded values
- ✅ **Category Pooling**: Unified predictions for related category versions
- ✅ **95% Confidence Intervals**: Statistical confidence for dates and CRS scores
- ✅ **12+ Advanced ML Models**: Including Prophet, Gaussian Process, Bayesian Hierarchical
- ✅ **Government Policy Integration**: Aligned with official 2025 Express Entry priorities
- ✅ **Mobile-Responsive Design**: Optimized UI for all devices
- ✅ **Performance Optimization**: VPS-optimized with intelligent caching
- ✅ **Domain-Aware Confidence**: Express Entry specific confidence calculation
- ✅ **Comprehensive Logging**: Detailed prediction process monitoring

## 🚀 Future Enhancements

- [ ] Real-time data integration with IRCC APIs
- [ ] Advanced economic indicator integration (unemployment rates, GDP growth)
- [ ] Multi-language support (French, Mandarin, Hindi, Spanish)
- [ ] Email/SMS prediction alerts and notifications
- [ ] Social media integration for real-time updates
- [ ] Mobile app development (iOS/Android)
- [ ] Advanced visualization dashboard with interactive charts
- [ ] Integration with provincial immigration programs
- [ ] Machine learning model interpretability (SHAP/LIME)
- [ ] A/B testing framework for model performance

---

Built with ❤️ for the Canadian immigration community.

## 🚀 Development Workflow

### Automated GitHub Push

This project includes an automated script to push changes to GitHub after editing sessions:

```bash
# Push changes with auto-generated commit message
./auto_push.sh

# Push changes with custom commit message
./auto_push.sh "Your custom commit message here"
```

**Features:**
- ✅ Automatic staging of all changes
- ✅ Smart commit messages with file listings
- ✅ Automatic push to main branch
- ✅ Error handling and status reporting
- ✅ Summary of changes pushed

**Usage Examples:**
```bash
# Quick push after development session
./auto_push.sh

# Push with specific feature description
./auto_push.sh "✨ Add new prediction feature

• Enhanced ML models
• Improved frontend UI
• Fixed critical bugs"
```

The script will automatically detect changes, commit them, and push to GitHub with proper error handling.