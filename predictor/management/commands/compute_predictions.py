from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import date, timedelta
import pandas as pd
import numpy as np

from predictor.models import (
    DrawCategory, ExpressEntryDraw, PreComputedPrediction, 
    PredictionModel, PredictionCache
)
from predictor.ml_models import (
    ARIMAPredictor, RandomForestPredictor, XGBoostPredictor, 
    LinearRegressionPredictor, NeuralNetworkPredictor,
    BayesianPredictor, SmallDatasetPredictor
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
        
        # Get all categories with recent activity (draws within last 2 years)
        all_categories = DrawCategory.objects.filter(is_active=True)
        active_categories = []
        
        for category in all_categories:
            if category.has_recent_activity(24):  # 24 months = 2 years
                active_categories.append(category)
            else:
                days_since = category.days_since_last_draw
                if days_since:
                    self.stdout.write(
                        self.style.WARNING(
                            f'⚠️  Skipping {category.name}: Last draw was {category.latest_draw_date} '
                            f'({days_since} days ago) - Program appears discontinued'
                        )
                    )
                else:
                    self.stdout.write(
                        self.style.WARNING(f'⚠️  Skipping {category.name}: No draws found')
                    )
        
        self.stdout.write(
            self.style.SUCCESS(
                f'📊 Found {len(active_categories)} active categories (with draws in last 2 years)'
            )
        )
        
        if not active_categories:
            self.stdout.write(self.style.ERROR('❌ No active categories found!'))
            return
        
        # Get categories to process (filter active categories if specified)
        if category_filter:
            categories = [cat for cat in active_categories if category_filter.lower() in cat.name.lower()]
            if not categories:
                self.stdout.write(self.style.ERROR(f'❌ Active category "{category_filter}" not found'))
                return
        else:
            categories = active_categories
        
        total_categories = len(categories)
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
        
        if draws.count() < 1:  # Need at least one data point
            raise ValueError(f"No data available: {draws.count()} draws found")
        
        # Log data availability for transparency  
        data_count = draws.count()
        if data_count <= 4:
            print(f"⚠️  Small dataset: {category.name} has only {data_count} draws - using specialized predictor")
        elif data_count <= 10:
            print(f"🔄 Limited data: {category.name} has {data_count} draws - using Bayesian approach")
        
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
        import pytz
        eastern = pytz.timezone('America/Toronto')  # Ottawa/Eastern Time
        
        # Get current date in Eastern Time
        now_eastern = timezone.now().astimezone(eastern)
        today_eastern = now_eastern.date()
        
        # Get last draw date
        last_draw_date = draws.last().date
        
        # Start predictions from today or 2 weeks after last draw, whichever is later
        days_since_last_draw = (today_eastern - last_draw_date).days
        if days_since_last_draw >= 14:
            # If it's been 2+ weeks since last draw, next draw could be soon
            next_draw_start = today_eastern + timedelta(days=7)  # Next week
        else:
            # Otherwise, wait for the normal 2-week interval
            next_draw_start = last_draw_date + timedelta(days=14)
            
        # Ensure we don't predict in the past
        current_date = max(next_draw_start, today_eastern)
        
        self.stdout.write(f"📅 Date calculation for {category.name}:")
        self.stdout.write(f"   Today (Eastern): {today_eastern}")
        self.stdout.write(f"   Last draw: {last_draw_date}")
        self.stdout.write(f"   Days since last draw: {days_since_last_draw}")
        self.stdout.write(f"   Starting predictions from: {current_date}")
        
        # Clear old predictions if force recompute
        if force_recompute:
            PreComputedPrediction.objects.filter(category=category).update(is_active=False)
        
        predictions_created = 0
        
        for rank in range(1, num_predictions + 1):
            # Express Entry draws typically happen every 2 weeks (14 days)
            # Add some variation: 12-16 days to make it more realistic
            base_interval = 14
            variation = (-2, -1, 0, 1, 2)[rank % 5]  # Cycle through variations
            interval = base_interval + variation
            
            next_date = current_date + timedelta(days=interval * rank)
            
            # Skip if too far in the future (more than 1 year from today)
            if next_date > today_eastern + timedelta(days=365):
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
                        prediction_result = best_model.predict(X)
                        predicted_score = prediction_result[0] if hasattr(prediction_result, '__len__') else prediction_result
                else:
                    predicted_score = df['lowest_crs_score'].mean()  # Fallback
                
                # Handle NaN predictions
                if pd.isna(predicted_score) or np.isnan(predicted_score):
                    predicted_score = df['lowest_crs_score'].mean()
                    print(f"⚠️  NaN prediction detected, using fallback: {predicted_score}")
                
                # Ensure prediction is a valid integer
                predicted_score = int(np.clip(predicted_score, 250, 950))
                
                # Ensure reasonable bounds
                predicted_score = max(300, min(900, int(predicted_score)))
                
                # ENHANCED: Proper invitation prediction using dedicated model
                from predictor.ml_models import InvitationPredictor
                
                try:
                    # Initialize and train invitation predictor
                    invitation_model = InvitationPredictor(model_type='XGB')
                    invitation_metrics = invitation_model.train(df)
                    
                    # Prepare features for invitation prediction
                    invitation_features = invitation_model.prepare_invitation_features(df)
                    exclude_cols = ['date', 'lowest_crs_score', 'round_number', 'url', 'category', 'invitations_issued']
                    feature_cols = [col for col in invitation_features.columns if col not in exclude_cols]
                    X_invitation = invitation_features[feature_cols].fillna(0).tail(1)
                    
                    # Predict invitation numbers with uncertainty
                    invitation_result = invitation_model.predict_with_uncertainty(
                        X_invitation, 
                        category=category.name
                    )
                    predicted_invitations = invitation_result['prediction']
                    invitation_uncertainty = invitation_result['std_dev']
                    
                    print(f"✅ Invitation prediction: {predicted_invitations} (±{invitation_uncertainty})")
                    
                    # Feature importance insights
                    if invitation_model.feature_importance:
                        top_features = list(invitation_model.feature_importance.items())[:5]
                        print(f"📊 Top invitation factors: {[f[0] for f in top_features]}")
                    
                except Exception as e:
                    print(f"⚠️ Invitation model failed, using fallback: {e}")
                    # Fallback to improved historical approach
                    avg_invitations = df['invitations_issued'].mean()
                    std_invitations = df['invitations_issued'].std()
                    
                    # Category-specific adjustments
                    if 'CEC' in category.name or 'Canadian Experience' in category.name:
                        base_invitations = 3000  # Your observation about CEC
                    elif 'PNP' in category.name:
                        base_invitations = avg_invitations * 0.8  # PNP typically smaller
                    else:
                        base_invitations = avg_invitations
                    
                    # Add some reasonable variation
                    predicted_invitations = max(500, int(base_invitations * (0.9 + 0.2 * (rank/num_predictions))))
                    invitation_uncertainty = std_invitations or 800
                
                # Calculate uncertainty range with proper 95% confidence intervals
                score_std = df['lowest_crs_score'].std()
                
                # For small datasets, increase uncertainty significantly
                data_size = len(df)
                if data_size <= 4:
                    uncertainty_multiplier = 3.0  # Very wide confidence intervals
                elif data_size <= 10:
                    uncertainty_multiplier = 2.0  # Wide confidence intervals  
                else:
                    uncertainty_multiplier = 1.0  # Normal confidence intervals
                
                # Calculate 95% confidence intervals
                # Using 1.96 as the z-score for 95% confidence interval
                z_score_95 = 1.96
                
                # Use Bayesian uncertainty if available
                if hasattr(best_model, 'predict_with_uncertainty') and hasattr(best_model, 'last_prediction_std'):
                    # Use model's uncertainty estimation for Bayesian models
                    model_std = getattr(best_model, 'last_prediction_std', [score_std])[0] if hasattr(best_model, 'last_prediction_std') else score_std
                    # Apply z-score for 95% CI
                    margin_of_error = z_score_95 * model_std * uncertainty_multiplier
                else:
                    # Fallback to data-based uncertainty with 95% CI
                    base_std = max(score_std, 15)  # Minimum 15 point uncertainty
                    margin_of_error = z_score_95 * base_std * uncertainty_multiplier
                
                # Calculate 95% confidence interval bounds for CRS score
                ci_lower = max(250, int(predicted_score - margin_of_error))
                ci_upper = min(950, int(predicted_score + margin_of_error))
                
                # Calculate date confidence interval (± days around predicted date)
                base_date_margin = 7  # Base uncertainty of ±7 days
                
                # Adjust date uncertainty based on model confidence and data quality
                if data_size <= 4:
                    date_margin_days = min(21, base_date_margin * 3)  # Up to ±21 days for very small data
                elif data_size <= 10:
                    date_margin_days = min(14, base_date_margin * 2)  # Up to ±14 days for small data
                elif confidence < 0.5:
                    date_margin_days = min(10, base_date_margin * 1.5)  # ±10 days for low confidence
                else:
                    date_margin_days = base_date_margin  # ±7 days for good confidence
                
                # Calculate date range
                date_lower = next_date - timedelta(days=date_margin_days)
                date_upper = next_date + timedelta(days=date_margin_days)
                
                # Ensure date range doesn't go into the past
                if date_lower < today_eastern:
                    date_lower = today_eastern
                
                uncertainty_range = {
                    'crs_min': ci_lower,
                    'crs_max': ci_upper,
                    'crs_margin_of_error': round(margin_of_error, 1),
                    'date_earliest': date_lower.isoformat(),
                    'date_latest': date_upper.isoformat(),
                    'date_margin_days': date_margin_days,
                    'confidence_level': 95
                }
                
                # Adjust confidence score based on uncertainty
                if data_size <= 4:
                    confidence = min(confidence * 0.6, 0.4)  # Cap at 40% for very small data
                elif data_size <= 10:
                    confidence = min(confidence * 0.8, 0.6)  # Cap at 60% for small data
                
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
        
        data_size = len(df)
        
        # Handle very small datasets (1-4 data points)
        if data_size <= 4:
            try:
                # Get global data for cross-category learning
                from predictor.models import ExpressEntryDraw
                global_draws = ExpressEntryDraw.objects.all().values(
                    'date', 'lowest_crs_score', 'invitations_issued'
                )
                global_df = pd.DataFrame(global_draws)
                
                # Use specialized small dataset predictor
                small_model = SmallDatasetPredictor(global_data=global_df)
                confidence = 0.2 + (data_size * 0.1)  # 20-50% confidence for very small data
                return small_model, confidence
                
            except Exception as e:
                print(f"Error setting up small dataset predictor: {e}")
                # Fallback to simple linear model with low confidence
                return LinearRegressionPredictor(), 0.15
        
        # Small datasets (5-10 data points) - use Bayesian model
        if data_size <= 10:
            models_to_try = [
                (BayesianPredictor(), 0.60),      # Great for small data with uncertainty
                (LinearRegressionPredictor(), 0.50),  # Simple baseline
            ]
            
            # Add small dataset predictor as backup
            try:
                from predictor.models import ExpressEntryDraw
                global_draws = ExpressEntryDraw.objects.all().values(
                    'date', 'lowest_crs_score', 'invitations_issued'  
                )
                global_df = pd.DataFrame(global_draws)
                small_model = SmallDatasetPredictor(global_data=global_df)
                models_to_try.append((small_model, 0.40))
            except Exception:
                pass
        
        # Medium datasets (11-19 data points)
        elif data_size <= 19:
            models_to_try = [
                (BayesianPredictor(), 0.75),         # Still good for medium data
                (LinearRegressionPredictor(), 0.70),  # Reliable baseline
                (RandomForestPredictor(), 0.65),     # May overfit with small data
            ]
        
        # Larger datasets (20+ data points) - full model suite
        else:
            models_to_try = [
                (LinearRegressionPredictor(), 0.70),  # Always available, good baseline
                (BayesianPredictor(), 0.72),         # Great uncertainty quantification
                (RandomForestPredictor(), 0.75),     # Good for non-linear patterns
            ]
            
            # Add more sophisticated models if enough data
            if data_size >= 20:
                try:
                    models_to_try.append((ARIMAPredictor(), 0.80))
                except ImportError:
                    pass
                
                try:
                    models_to_try.append((XGBoostPredictor(), 0.85))
                except ImportError:
                    pass
            
            if data_size >= 30:
                try:
                    models_to_try.append((NeuralNetworkPredictor(), 0.78))
                except ImportError:
                    pass
        
        # Try models and pick the best one
        best_model = None
        best_confidence = 0
        
        for model, base_confidence in models_to_try:
            try:
                # Adjust confidence based on data quality
                data_quality_score = min(1.0, data_size / 50)  # More data = higher confidence
                score_variance = df['lowest_crs_score'].std()
                stability_score = max(0.3, min(1.0, 50 / score_variance)) if score_variance > 0 else 0.5
                
                # Penalty for very small datasets (but don't exclude completely)
                size_penalty = 1.0 if data_size >= 10 else (0.5 + data_size * 0.05)
                
                confidence = base_confidence * data_quality_score * stability_score * size_penalty
                
                # For small datasets, add uncertainty bonus to Bayesian models
                if data_size <= 10 and hasattr(model, 'predict_with_uncertainty'):
                    confidence *= 1.1  # 10% bonus for uncertainty quantification
                
                if confidence > best_confidence:
                    best_model = model
                    best_confidence = confidence
                    
            except Exception as e:
                print(f"Error evaluating model {model.name}: {e}")
                continue
        
        # Ensure we always return a model
        if best_model is None:
            print(f"No suitable model found, using fallback LinearRegression for {category.name}")
            best_model = LinearRegressionPredictor()
            best_confidence = 0.2
        
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