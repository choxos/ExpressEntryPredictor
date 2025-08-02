# Express Entry Predictor

A sophisticated Django web application that uses machine learning and statistical models to predict Canadian Express Entry draw dates and CRS scores.

## 🚀 Features

- **AI-Powered Predictions**: Uses ensemble of ML models including ARIMA, Random Forest, XGBoost, and LSTM
- **Interactive Dashboard**: Real-time analytics and visualizations
- **Historical Data Analysis**: Comprehensive analysis of 350+ historical draws
- **Modern UI**: Beautiful, responsive interface with interactive charts
- **RESTful API**: Complete API for data access and predictions
- **Admin Interface**: Full Django admin for data management

## 📊 Models Used

1. **ARIMA Time Series** - For temporal pattern analysis
2. **Random Forest** - For robust ensemble predictions
3. **XGBoost** - For high-performance gradient boosting
4. **LSTM Neural Networks** - For deep sequence learning
5. **Linear Regression** - For baseline comparisons
6. **Ensemble Model** - Combines all models for optimal accuracy

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

### Using the Management Commands

```bash
# Load new draw data
python manage.py load_draw_data --file path/to/new_data.csv

# Train specific model
python manage.py train_model --model-type RF

# Generate predictions
python manage.py generate_predictions --category PNP
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

### Prediction Pipeline

1. **Data Preprocessing**: Historical draws are cleaned and features engineered
2. **Model Training**: Multiple models are trained on historical data
3. **Ensemble Prediction**: Models are combined using weighted voting
4. **Confidence Calculation**: Statistical confidence intervals are computed

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

## 🚀 Future Enhancements

- [ ] Real-time data integration with IRCC APIs
- [ ] Mobile app development
- [ ] Advanced economic indicator integration
- [ ] Machine learning model optimization
- [ ] Multi-language support
- [ ] Email prediction alerts
- [ ] Social media integration for updates

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