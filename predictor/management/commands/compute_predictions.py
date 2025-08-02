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
        
        # ENHANCED: Get pooled data from related category versions
        pooled_draws, ircc_category, num_pooled_categories = category.get_pooled_data()
        
        if pooled_draws.count() < 1:  # Need at least one data point
            raise ValueError(f"No data available: {pooled_draws.count()} draws found")
        
        # Log data availability and pooling info
        individual_count = ExpressEntryDraw.objects.filter(category=category).count()
        pooled_count = pooled_draws.count()
        
        if num_pooled_categories > 1:
            print(f"📊 POOLED DATA: {category.name}")
            print(f"   ├─ Individual draws: {individual_count}")
            print(f"   ├─ Pooled with {num_pooled_categories} categories: {pooled_count} total draws")
            print(f"   └─ IRCC category: {ircc_category}")
        
        if pooled_count <= 4:
            print(f"⚠️  Small dataset: {ircc_category} has only {pooled_count} draws - using specialized predictor")
        elif pooled_count <= 10:
            print(f"🔄 Limited data: {ircc_category} has {pooled_count} draws - using Bayesian approach")
        else:
            print(f"✅ Good dataset: {ircc_category} has {pooled_count} draws - using advanced models")
        
        # Convert to DataFrame with pooled data
        df = pd.DataFrame([{
            'date': draw.date,
            'category': ircc_category,  # Use IRCC category name for consistency
            'lowest_crs_score': draw.lowest_crs_score,
            'invitations_issued': draw.invitations_issued,
            'days_since_last_draw': draw.days_since_last_draw or 14,
            'is_weekend': draw.is_weekend,
            'is_holiday': draw.is_holiday,
            'month': draw.month,
            'quarter': draw.quarter
        } for draw in pooled_draws])
        
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
        last_draw_date = pooled_draws.last().date
        
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
        
        # FIXED: Train invitation model ONCE outside the loop
        from predictor.ml_models import InvitationPredictor
        invitation_model = None
        invitation_trained = False
        
        try:
            invitation_model = InvitationPredictor(model_type='XGB')
            invitation_metrics = invitation_model.train(df)
            invitation_trained = True
            print(f"✅ Invitation model trained successfully with {len(df)} historical draws")
        except Exception as e:
            print(f"⚠️ Failed to train invitation model: {e}")
            invitation_trained = False
        
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
                    print(f"⚠️ NaN predicted_score detected, using fallback: {predicted_score}")
                
                # Ensure prediction is a valid integer
                predicted_score = int(np.clip(predicted_score, 250, 950))
                
                # Ensure reasonable bounds
                predicted_score = max(300, min(900, int(predicted_score)))
                
                # FIXED: Proper invitation prediction with future-date features
                if invitation_trained:
                    try:
                        # Generate features for the FUTURE prediction date (not historical data)
                        future_features = self.generate_future_features(
                            historical_df=df,
                            prediction_date=next_date,
                            last_draw_date=last_draw_date,
                            category_name=ircc_category,
                            rank=rank
                        )
                        
                        # Prepare features using the trained model's method
                        invitation_features = invitation_model.prepare_invitation_features(future_features)
                        exclude_cols = ['date', 'lowest_crs_score', 'round_number', 'url', 'category', 'invitations_issued']
                        feature_cols = [col for col in invitation_features.columns if col not in exclude_cols]
                        X_invitation = invitation_features[feature_cols].fillna(0).tail(1)
                        
                        # Predict invitation numbers with uncertainty scaling
                        invitation_result = invitation_model.predict_with_uncertainty(
                            X_invitation, 
                            category=category.name,
                            prediction_horizon=rank  # Scale uncertainty by prediction distance
                        )
                        predicted_invitations = invitation_result['prediction']
                        invitation_uncertainty = invitation_result['std_dev']
                        
                        print(f"✅ Invitation prediction (rank {rank}): {predicted_invitations} (±{invitation_uncertainty:.0f})")
                        
                        # Feature importance insights (only for first prediction to avoid spam)
                        if rank == 1 and invitation_model.feature_importance:
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
                        
                        # Add horizon-based variation (scientific improvement)
                        seasonal_factor = 1.0 + 0.05 * ((rank % 4) - 2)  # Seasonal variation
                        predicted_invitations = max(500, int(base_invitations * seasonal_factor))
                        invitation_uncertainty = (std_invitations or 800) * (1 + 0.1 * rank)  # Scale with horizon
                        
                        print(f"✅ Fallback invitation prediction (rank {rank}): {predicted_invitations} (±{invitation_uncertainty:.0f})")
                
                else:
                    # Invitation model not trained - use enhanced statistical fallback
                    print(f"⚠️ Invitation model not available, using statistical approach for rank {rank}")
                    
                    avg_invitations = df['invitations_issued'].mean()
                    std_invitations = df['invitations_issued'].std()
                    
                    # Category-aware baseline predictions
                    if 'CEC' in category.name or 'Canadian Experience' in category.name:
                        base_invitations = 3000  # CEC is very consistent
                        category_variation = 0.05  # Low variation
                    elif 'French' in category.name:
                        base_invitations = avg_invitations
                        category_variation = 0.3  # High variation for French draws  
                    elif 'Healthcare' in category.name:
                        base_invitations = avg_invitations * 1.1  # Slightly higher demand
                        category_variation = 0.15  # Moderate variation
                    elif 'PNP' in category.name:
                        base_invitations = avg_invitations * 0.7  # Typically smaller
                        category_variation = 0.2  # Moderate variation
                    else:
                        base_invitations = avg_invitations
                        category_variation = 0.2  # Default variation
                    
                    # Apply prediction horizon effects
                    horizon_uncertainty = 1 + (0.1 * rank)  # 10% increase per rank
                    seasonal_effect = 1.0 + 0.05 * ((rank % 4) - 2)  # Cyclic seasonal pattern
                    
                    predicted_invitations = max(500, int(base_invitations * seasonal_effect))
                    invitation_uncertainty = (std_invitations or 800) * horizon_uncertainty
                    
                    print(f"✅ Statistical invitation prediction (rank {rank}): {predicted_invitations} (±{invitation_uncertainty:.0f})")
                
                # Calculate uncertainty range with proper 95% confidence intervals
                score_std = df['lowest_crs_score'].std()
                
                # Calculate data size first for use in multiple places
                data_size = len(df)
                
                # Handle NaN std for single data points
                if pd.isna(score_std) or np.isnan(score_std):
                    # For single data points, use a reasonable default uncertainty
                    if data_size == 1:
                        score_std = 50  # Default 50-point uncertainty for single data point
                    else:
                        score_std = 25  # Default 25-point uncertainty for very small datasets
                    print(f"⚠️ NaN std detected, using fallback: {score_std}")
                
                # For small datasets, increase uncertainty significantly
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
                    
                    # Handle NaN in model_std
                    if pd.isna(model_std) or np.isnan(model_std):
                        model_std = score_std  # Fall back to score_std
                        print(f"⚠️ NaN model_std detected, using score_std: {model_std}")
                    
                    # Apply z-score for 95% CI
                    margin_of_error = z_score_95 * model_std * uncertainty_multiplier
                else:
                    # Fallback to data-based uncertainty with 95% CI
                    base_std = max(score_std, 15)  # Minimum 15 point uncertainty
                    margin_of_error = z_score_95 * base_std * uncertainty_multiplier
                
                # Calculate 95% confidence interval bounds for CRS score
                ci_lower = max(250, int(predicted_score - margin_of_error))
                ci_upper = min(950, int(predicted_score + margin_of_error))
                
                # Handle NaN in margin_of_error
                if pd.isna(margin_of_error) or np.isnan(margin_of_error):
                    margin_of_error = 50.0  # Default margin of error
                    ci_lower = max(250, int(predicted_score - margin_of_error))
                    ci_upper = min(950, int(predicted_score + margin_of_error))
                    print(f"⚠️ NaN margin_of_error detected, using fallback: {margin_of_error}")
                
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
                
                # FINAL NaN SAFETY CHECKS before database save
                # Ensure all values are valid numbers that can be saved to database
                if pd.isna(predicted_score) or np.isnan(predicted_score):
                    predicted_score = int(df['lowest_crs_score'].mean() or 450)
                    print(f"⚠️ NaN predicted_score detected, using fallback: {predicted_score}")
                
                if pd.isna(predicted_invitations) or np.isnan(predicted_invitations):
                    fallback_invitations = int(df['invitations_issued'].mean() or 2000)
                    predicted_invitations = fallback_invitations
                    print(f"⚠️ NaN predicted_invitations detected, using fallback: {predicted_invitations}")
                
                if pd.isna(confidence) or np.isnan(confidence):
                    confidence = 0.3  # Default 30% confidence
                    print(f"⚠️ NaN confidence detected, using fallback: {confidence}")
                
                # Ensure integer values are properly cast
                try:
                    predicted_score = int(float(predicted_score))
                    predicted_invitations = int(float(predicted_invitations))
                    confidence = float(confidence)
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Value conversion error: {e}, using safe fallbacks")
                    predicted_score = int(df['lowest_crs_score'].mean() or 450)
                    predicted_invitations = int(df['invitations_issued'].mean() or 2000)
                    confidence = 0.3
                
                # Final bounds checking
                predicted_score = max(250, min(950, predicted_score))
                predicted_invitations = max(100, min(10000, predicted_invitations))
                confidence = max(0.1, min(1.0, confidence))
                
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
        
        # Use pooled data count for model selection
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
                fallback_model = LinearRegressionPredictor()
                fallback_model.name = "Linear Regression (Fallback)"
                return fallback_model, 0.2
        
        # Handle small datasets (5-10 data points) with pooled benefits  
        elif data_size <= 10:
            try:
                # With pooled data, we might have enough for Bayesian approach
                bayesian_model = BayesianPredictor()
                bayesian_model.name = "Bayesian Regression"
                confidence = 0.4 + (data_size * 0.05)  # 40-90% confidence based on data size
                return bayesian_model, confidence
                
            except Exception as e:
                print(f"Error setting up Bayesian model: {e}")
                # Fallback to Random Forest which works well with small data
                rf_model = RandomForestPredictor()
                rf_model.name = "Random Forest (Small Data)"
                return rf_model, 0.3
        
        # Medium datasets (11-30 data points) - pooled data advantage
        elif data_size <= 30:
            try:
                # Random Forest for medium pooled data
                rf_model = RandomForestPredictor()
                rf_model.name = "Random Forest (Medium Data)"
                confidence = 0.6 + (data_size * 0.01)  # 60-90% confidence
                return rf_model, confidence
                
            except Exception as e:
                print(f"Error setting up Random Forest: {e}")
                linear_model = LinearRegressionPredictor()
                linear_model.name = "Linear Regression (Fallback)"
                return linear_model, 0.5
        
        # Large datasets (30+ data points) - full model capability
        else:
            try:
                # XGBoost for large pooled datasets
                from predictor.ml_models import XGBoostPredictor
                xgb_model = XGBoostPredictor()
                xgb_model.name = "XGBoost (Large Pooled Data)"
                confidence = 0.75 + min(data_size * 0.005, 0.2)  # 75-95% confidence
                return xgb_model, confidence
                
            except Exception as e:
                print(f"Error setting up XGBoost: {e}")
                # Fallback to Random Forest
                rf_model = RandomForestPredictor()
                rf_model.name = "Random Forest (Fallback)"
                return rf_model, 0.7

    def cache_dashboard_stats(self):
        """Cache basic prediction counts (removed conflicting dashboard stats cache)"""
        
        try:
            # Just cache basic counts - let DashboardStatsAPIView handle its own caching
            total_predictions = PreComputedPrediction.objects.filter(is_active=True).count()
            categories_with_predictions = PreComputedPrediction.objects.filter(
                is_active=True
            ).values('category').distinct().count()
            
            # Cache prediction counts only with different cache key
            prediction_counts = {
                'total_predictions': total_predictions,
                'categories_with_predictions': categories_with_predictions,
                'last_updated': timezone.now().isoformat(),
            }
            
            # Use different cache key to avoid conflict with DashboardStatsAPIView
            PredictionCache.set_cache('prediction_counts', prediction_counts, hours=24)
            
            self.stdout.write(f'📊 Cached prediction counts: {total_predictions} predictions for {categories_with_predictions} categories')
            
        except Exception as e:
            self.stdout.write(self.style.WARNING(f'⚠️  Failed to cache prediction counts: {str(e)}')) 

    def generate_future_features(self, historical_df, prediction_date, last_draw_date, category_name, rank):
        """
        Generate feature vector for a future prediction date.
        This is scientifically critical - features must represent the FUTURE state, not historical data.
        """
        import calendar
        
        # Create a future dataframe row based on the prediction date
        future_row = {
            'date': prediction_date,
            'category': category_name,
            'days_since_last_draw': (prediction_date - last_draw_date).days,
            'is_weekend': prediction_date.weekday() >= 5,
            'is_holiday': self.is_holiday_period(prediction_date),
            'month': prediction_date.month,
            'quarter': (prediction_date.month - 1) // 3 + 1
        }
        
        # Add recent historical context for lag/rolling features
        # Use the MOST RECENT values as baseline, then project forward
        recent_data = historical_df.tail(14).copy()  # Last 14 draws for context
        
        # Estimate future values based on trends and seasonality
        if len(recent_data) >= 3:
            # CRS Score projection (use recent trend)
            recent_crs_trend = recent_data['lowest_crs_score'].tail(3).mean()
            seasonal_adjustment = self.get_seasonal_crs_adjustment(prediction_date.month, category_name)
            future_row['lowest_crs_score'] = int(recent_crs_trend + seasonal_adjustment)
            
            # Invitations projection (use category-specific patterns)
            recent_invitations_avg = recent_data['invitations_issued'].tail(3).mean()
            category_adjustment = self.get_category_invitation_adjustment(category_name, prediction_date.month)
            future_row['invitations_issued'] = int(recent_invitations_avg * category_adjustment)
        else:
            # Fallback for categories with very little data
            future_row['lowest_crs_score'] = historical_df['lowest_crs_score'].mean()
            future_row['invitations_issued'] = historical_df['invitations_issued'].mean()
        
        # Create future dataframe (historical + projected future row)
        future_df = pd.concat([
            historical_df,
            pd.DataFrame([future_row])
        ], ignore_index=True)
        
        return future_df
    
    def is_holiday_period(self, date):
        """Check if date falls in a holiday period affecting Express Entry processing"""
        month = date.month
        day = date.day
        
        # Canadian holiday periods that typically affect EE processing
        holiday_periods = [
            (12, 15, 1, 15),   # Christmas/New Year period
            (7, 1, 8, 15),     # Summer holiday period
            (3, 15, 4, 15),    # Easter period (approximate)
        ]
        
        for start_month, start_day, end_month, end_day in holiday_periods:
            if start_month <= month <= end_month:
                if (month == start_month and day >= start_day) or \
                   (month == end_month and day <= end_day) or \
                   (start_month < month < end_month):
                    return True
        return False
    
    def get_seasonal_crs_adjustment(self, month, category_name):
        """Get seasonal CRS score adjustments based on historical patterns"""
        # Historical analysis shows certain patterns:
        seasonal_patterns = {
            # Winter months: slight increase due to fewer draws
            1: 5, 2: 3, 3: 0,
            # Spring: moderate activity
            4: -2, 5: -3, 6: -1,
            # Summer: variable (holidays vs. increased immigration targets)
            7: 2, 8: 4, 9: 1,
            # Fall: high activity
            10: -5, 11: -3, 12: 2
        }
        
        base_adjustment = seasonal_patterns.get(month, 0)
        
        # Category-specific adjustments
        if 'Healthcare' in category_name:
            # Healthcare draws are less seasonal
            base_adjustment *= 0.5
        elif 'French' in category_name:
            # French draws are more aggressive in certain periods
            if month in [3, 6, 9, 12]:  # Quarter ends
                base_adjustment -= 10
        elif 'CEC' in category_name or 'Canadian Experience' in category_name:
            # CEC is more stable
            base_adjustment *= 0.3
            
        return base_adjustment
    
    def get_category_invitation_adjustment(self, category_name, month):
        """Get category-specific invitation number adjustments"""
        base_multiplier = 1.0
        
        # Category-specific patterns
        if 'CEC' in category_name or 'Canadian Experience' in category_name:
            # CEC tends to be stable around 3000, slight seasonal variation
            if month in [3, 6, 9, 12]:  # Quarter ends
                base_multiplier = 1.1
            elif month in [7, 8]:  # Summer slowdown
                base_multiplier = 0.9
        elif 'Healthcare' in category_name:
            # Healthcare draws vary based on demand
            if month in [1, 2, 9, 10]:  # High demand periods
                base_multiplier = 1.2
        elif 'French' in category_name:
            # French draws can be very large in certain periods
            if month in [3, 6, 9]:
                base_multiplier = 1.5
        elif 'PNP' in category_name or 'Provincial' in category_name:
            # PNP is more consistent but smaller
            base_multiplier = 1.0
            
        return base_multiplier 