from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import date, timedelta
import pandas as pd

from predictor.models import (
    DrawCategory, ExpressEntryDraw, PreComputedPrediction, 
    PredictionModel, PredictionCache
)
from predictor.ml_models import (
    ARIMAPredictor, RandomForestPredictor, XGBoostPredictor, 
    LinearRegressionPredictor, NeuralNetworkPredictor
)


class Command(BaseCommand):
    help = 'Pre-compute predictions for all categories to avoid real-time calculations'

    def add_arguments(self, parser):
        parser.add_argument(
            '--category',
            type=str,
            help='Specific category name to compute predictions for',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force recomputation even if recent predictions exist',
        )
        parser.add_argument(
            '--predictions',
            type=int,
            default=10,
            help='Number of future predictions to generate (default: 10)',
        )

    def handle(self, *args, **options):
        category_filter = options.get('category')
        force_recompute = options.get('force')
        num_predictions = options.get('predictions')
        
        self.stdout.write(self.style.SUCCESS(
            f'🚀 Starting prediction computation for {num_predictions} future draws'
        ))
        
        # Get categories to process
        if category_filter:
            categories = DrawCategory.objects.filter(name__icontains=category_filter, is_active=True)
            if not categories.exists():
                self.stdout.write(self.style.ERROR(f'❌ Category "{category_filter}" not found'))
                return
        else:
            categories = DrawCategory.objects.filter(is_active=True)
        
        total_categories = categories.count()
        self.stdout.write(f'📊 Processing {total_categories} categories...')
        
        successful_predictions = 0
        failed_categories = []
        
        for i, category in enumerate(categories, 1):
            self.stdout.write(f'\n[{i}/{total_categories}] Processing: {category.name}')
            
            try:
                predictions_created = self.compute_category_predictions(
                    category, num_predictions, force_recompute
                )
                if predictions_created > 0:
                    successful_predictions += 1
                    self.stdout.write(self.style.SUCCESS(
                        f'✅ Created {predictions_created} predictions for {category.name}'
                    ))
                else:
                    self.stdout.write(self.style.WARNING(
                        f'⚠️  No predictions created for {category.name} (insufficient data or already computed)'
                    ))
            except Exception as e:
                failed_categories.append(category.name)
                self.stdout.write(self.style.ERROR(f'❌ Failed to process {category.name}: {str(e)}'))
        
        # Summary
        self.stdout.write(f'\n📈 PREDICTION COMPUTATION SUMMARY')
        self.stdout.write(f'✅ Successful categories: {successful_predictions}/{total_categories}')
        if failed_categories:
            self.stdout.write(f'❌ Failed categories: {", ".join(failed_categories)}')
        
        # Cache dashboard stats
        self.cache_dashboard_stats()
        
        self.stdout.write(self.style.SUCCESS('\n🎉 Prediction computation completed!'))

    def compute_category_predictions(self, category, num_predictions, force_recompute):
        """Compute predictions for a specific category"""
        
        # Check if we need to recompute
        if not force_recompute:
            existing_predictions = PreComputedPrediction.objects.filter(
                category=category, 
                is_active=True,
                created_at__gte=timezone.now() - timedelta(days=1)
            ).count()
            
            if existing_predictions >= num_predictions:
                return 0  # Already have recent predictions
        
        # Get historical data
        draws = ExpressEntryDraw.objects.filter(category=category).order_by('date')
        
        if draws.count() < 5:  # Need minimum data
            raise ValueError(f"Insufficient data: only {draws.count()} draws available")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'date': draw.date,
            'category': category.name,
            'lowest_crs_score': draw.lowest_crs_score,
            'invitations_issued': draw.invitations_issued,
            'days_since_last_draw': draw.days_since_last_draw or 14,
            'is_weekend': draw.is_weekend,
            'is_holiday': draw.is_holiday,
            'month': draw.month,
            'quarter': draw.quarter
        } for draw in draws])
        
        # Choose best model for this category
        best_model, confidence = self.select_best_model(df, category)
        
        if not best_model:
            raise ValueError("No suitable model found for this data")
        
        # Train the model
        try:
            metrics = best_model.train(df)
        except Exception as e:
            raise ValueError(f"Failed to train model: {str(e)}")
        
        # Generate predictions
        last_draw_date = draws.last().date
        current_date = last_draw_date
        
        # Clear old predictions if force recompute
        if force_recompute:
            PreComputedPrediction.objects.filter(category=category).update(is_active=False)
        
        predictions_created = 0
        
        for rank in range(1, num_predictions + 1):
            # Estimate next draw date (typically 2 weeks apart)
            next_date = current_date + timedelta(days=14 * rank)
            
            # Skip if too far in the future (more than 1 year)
            if next_date > last_draw_date + timedelta(days=365):
                break
            
            try:
                # Predict CRS score based on model type
                if hasattr(best_model, 'predict'):
                    # Different models have different predict interfaces
                    if best_model.name in ['ARIMA Time Series', 'Prophet Time Series']:
                        # Time series models can predict multiple steps
                        predicted_scores = best_model.predict(steps=rank)
                        if isinstance(predicted_scores, list) and len(predicted_scores) >= rank:
                            predicted_score = predicted_scores[rank-1]
                        else:
                            predicted_score = predicted_scores if not isinstance(predicted_scores, list) else predicted_scores[0]
                    else:
                        # ML models need feature data with same engineering as training
                        # Use the model's prepare_features method to get consistent features
                        features_df = best_model.prepare_features(df)
                        exclude_cols = ['date', 'lowest_crs_score', 'round_number', 'url', 'category']
                        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
                        X = features_df[feature_cols].fillna(0).tail(1)  # Use last row
                        predicted_score = best_model.predict(X)[0] if hasattr(best_model.predict(X), '__len__') else best_model.predict(X)
                else:
                    predicted_score = df['lowest_crs_score'].mean()  # Fallback
                
                # Ensure reasonable bounds
                predicted_score = max(300, min(900, int(predicted_score)))
                
                # Estimate invitations (based on historical average)
                avg_invitations = df['invitations_issued'].mean()
                predicted_invitations = max(100, int(avg_invitations * (0.8 + 0.4 * (rank/num_predictions))))
                
                # Calculate uncertainty range
                score_std = df['lowest_crs_score'].std()
                uncertainty_range = {
                    'min': max(300, int(predicted_score - score_std)),
                    'max': min(900, int(predicted_score + score_std))
                }
                
                # Create prediction
                PreComputedPrediction.objects.update_or_create(
                    category=category,
                    prediction_rank=rank,
                    defaults={
                        'predicted_date': next_date,
                        'predicted_crs_score': predicted_score,
                        'predicted_invitations': predicted_invitations,
                        'confidence_score': confidence,
                        'model_used': best_model.name,
                        'model_version': '1.0',
                        'uncertainty_range': uncertainty_range,
                        'is_active': True
                    }
                )
                
                predictions_created += 1
                
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'⚠️  Failed to create prediction {rank}: {str(e)}'))
                continue
        
        return predictions_created

    def select_best_model(self, df, category):
        """Select the best model for a category based on data characteristics"""
        
        models_to_try = [
            (LinearRegressionPredictor(), 0.70),  # Always available, good baseline
            (RandomForestPredictor(), 0.75),     # Good for non-linear patterns
        ]
        
        # Add more sophisticated models if enough data
        if len(df) >= 20:
            try:
                models_to_try.append((ARIMAPredictor(), 0.80))
            except ImportError:
                pass
            
            try:
                models_to_try.append((XGBoostPredictor(), 0.85))
            except ImportError:
                pass
        
        if len(df) >= 30:
            try:
                models_to_try.append((NeuralNetworkPredictor(), 0.78))
            except ImportError:
                pass
        
        # Try models and pick the best one
        best_model = None
        best_confidence = 0
        
        for model, base_confidence in models_to_try:
            try:
                # Quick validation
                if len(df) < 5:
                    continue
                
                # Adjust confidence based on data quality
                data_quality_score = min(1.0, len(df) / 50)  # More data = higher confidence
                score_variance = df['lowest_crs_score'].std()
                stability_score = max(0.5, min(1.0, 50 / score_variance)) if score_variance > 0 else 0.5
                
                confidence = base_confidence * data_quality_score * stability_score
                
                if confidence > best_confidence:
                    best_model = model
                    best_confidence = confidence
                    
            except Exception:
                continue
        
        return best_model, best_confidence

    def cache_dashboard_stats(self):
        """Cache expensive dashboard statistics"""
        
        try:
            # Calculate stats
            total_predictions = PreComputedPrediction.objects.filter(is_active=True).count()
            categories_with_predictions = PreComputedPrediction.objects.filter(
                is_active=True
            ).values('category').distinct().count()
            
            last_updated = timezone.now()
            
            stats = {
                'total_predictions': total_predictions,
                'categories_with_predictions': categories_with_predictions,
                'last_updated': last_updated.isoformat(),
                'next_predicted_draws': [
                    {
                        'category_name': pred['category__name'],
                        'predicted_date': pred['predicted_date'].isoformat(),
                        'predicted_crs_score': pred['predicted_crs_score'],
                        'confidence_score': pred['confidence_score']
                    }
                    for pred in PreComputedPrediction.objects.filter(
                        is_active=True, 
                        prediction_rank=1
                    ).select_related('category').values(
                        'category__name', 
                        'predicted_date', 
                        'predicted_crs_score',
                        'confidence_score'
                    )[:10]
                ]
            }
            
            # Cache for 24 hours
            PredictionCache.set_cache('dashboard_stats', stats, hours=24)
            
            self.stdout.write(f'📊 Cached dashboard stats: {total_predictions} predictions for {categories_with_predictions} categories')
            
        except Exception as e:
            self.stdout.write(self.style.WARNING(f'⚠️  Failed to cache dashboard stats: {str(e)}')) 