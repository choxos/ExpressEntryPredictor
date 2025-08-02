from django.core.management.base import BaseCommand
from predictor.models import PredictionModel


class Command(BaseCommand):
    help = 'Set up initial prediction models in the database'

    def handle(self, *args, **options):
        models_to_create = [
            {
                'name': 'ARIMA Time Series Model',
                'model_type': 'ARIMA',
                'version': '1.0',
                'description': 'Auto-Regressive Integrated Moving Average model for time series prediction',
                'is_active': True
            },
            {
                'name': 'Random Forest Ensemble',
                'model_type': 'RF',
                'version': '1.0',
                'description': 'Random Forest model using multiple decision trees',
                'is_active': True
            },
            {
                'name': 'XGBoost Gradient Boosting',
                'model_type': 'XGB',
                'version': '1.0',
                'description': 'Extreme Gradient Boosting model for high performance predictions',
                'is_active': True
            },
            {
                'name': 'LSTM Neural Network',
                'model_type': 'LSTM',
                'version': '1.0',
                'description': 'Long Short-Term Memory neural network for sequence prediction',
                'is_active': True
            },
            {
                'name': 'Linear Regression',
                'model_type': 'LR',
                'version': '1.0',
                'description': 'Simple linear regression baseline model',
                'is_active': True
            },
                                {
                        'name': 'Prophet Time Series',
                        'model_type': 'PROPHET',
                        'version': '1.0',
                        'description': 'Facebook Prophet model for seasonal time series prediction',
                        'is_active': True
                    },
                    {
                        'name': 'Ensemble Predictor',
                        'model_type': 'ENSEMBLE',
                        'version': '1.0',
                        'description': 'Ensemble of multiple models for robust predictions',
                        'is_active': True
                    }
        ]

        created_count = 0
        for model_data in models_to_create:
            model, created = PredictionModel.objects.get_or_create(
                name=model_data['name'],
                model_type=model_data['model_type'],
                defaults=model_data
            )
            
            if created:
                created_count += 1
                self.stdout.write(f'Created model: {model.name}')

        self.stdout.write(
            self.style.SUCCESS(f'Successfully created {created_count} prediction models')
        ) 