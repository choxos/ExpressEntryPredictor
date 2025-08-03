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
    SmallDatasetPredictor, ARIMAPredictor, LSTMPredictor, ProphetPredictor,
    CleanLinearRegressionPredictor, BayesianHierarchicalPredictor, GaussianProcessPredictor,
    SARIMAPredictor, VARPredictor, HoltWintersPredictor, DynamicLinearModelPredictor,
    ExponentialSmoothingPredictor, AdvancedEnsemblePredictor
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
        parser.add_argument(
            '--summary',
            action='store_true',
            help='Show detailed summary of successful and failed categories',
        )

    def handle(self, *args, **options):
        category_filter = options.get('category')
        force_recompute = options.get('force')
        num_predictions = options.get('predictions')
        show_summary = options.get('summary')
        
        # Track successful and failed categories for summary
        successful_categories = []
        failed_categories = []
        
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
        
        # Group categories by IRCC category to avoid duplicates
        ircc_groups = {}
        for category in categories:
            ircc_category, related_categories = DrawCategory.get_pooled_categories(category.name)
            
            # Use the IRCC category as the key
            if ircc_category not in ircc_groups:
                ircc_groups[ircc_category] = {
                    'representative_category': category,  # Use first category as representative
                    'related_categories': list(related_categories),
                    'total_draws': 0
                }
            
            # Update total draws count
            pooled_draws, _, _ = category.get_pooled_data()
            ircc_groups[ircc_category]['total_draws'] = pooled_draws.count()
        
        # Display grouping summary
        self.stdout.write(f'\n📊 Grouped into {len(ircc_groups)} unique IRCC categories:')
        for ircc_cat, group_info in ircc_groups.items():
            related_names = [cat.name for cat in group_info['related_categories']]
            if len(related_names) > 1:
                self.stdout.write(f'   🔗 {ircc_cat}: {len(related_names)} versions → {group_info["total_draws"]} total draws')
                for name in related_names:
                    self.stdout.write(f'      └─ {name}')
            else:
                self.stdout.write(f'   📋 {ircc_cat}: {group_info["total_draws"]} draws')
        
        total_groups = len(ircc_groups)
        self.stdout.write(f'\n📊 Processing {total_groups} unique IRCC categories...')
        
        successful_predictions = 0
        local_failed_categories = []
        
        for i, (ircc_category, group_info) in enumerate(ircc_groups.items(), 1):
            representative_category = group_info['representative_category']
            self.stdout.write(f'\n[{i}/{total_groups}] Processing IRCC Category: {ircc_category}')
            self.stdout.write(f'   📂 Using representative: {representative_category.name}')
            
            try:
                predictions_created = self.compute_category_predictions(
                    representative_category, num_predictions, force_recompute
                )
                
                if predictions_created > 0:
                    successful_predictions += 1
                    successful_categories.append({
                        'name': ircc_category,
                        'representative': representative_category.name,
                        'predictions': predictions_created
                    })
                    self.stdout.write(self.style.SUCCESS(
                        f'✅ Created {predictions_created} predictions for {ircc_category}'
                    ))
                else:
                    failed_categories.append({
                        'name': ircc_category,
                        'representative': representative_category.name,
                        'reason': 'No predictions created (insufficient data or already computed)',
                        'predictions': 0
                    })
                    self.stdout.write(self.style.WARNING(
                        f'⚠️  No predictions created for {ircc_category} (insufficient data or already computed)'
                    ))
                
            except Exception as e:
                local_failed_categories.append((ircc_category, str(e)))
                failed_categories.append({
                    'name': ircc_category,
                    'representative': representative_category.name,
                    'reason': f'Exception: {str(e)}',
                    'predictions': 0
                })
                self.stdout.write(
                    self.style.ERROR(f'❌ Failed to process {ircc_category}: {str(e)}')
                )
        
        # Summary
        self.stdout.write(f'\n📈 PREDICTION COMPUTATION SUMMARY')
        self.stdout.write(f'✅ Successful categories: {successful_predictions}/{total_groups}')
        if local_failed_categories:
            self.stdout.write(f'❌ Failed categories: {", ".join([f"{cat}: {err}" for cat, err in local_failed_categories])}')
        
        # Detailed summary if requested
        if show_summary:
            self.stdout.write(f'\n📊 DETAILED CATEGORY STATUS:')
            
            if successful_categories:
                self.stdout.write(f'\n✅ SUCCESSFUL CATEGORIES ({len(successful_categories)}):')
                for cat in successful_categories:
                    self.stdout.write(f'   • {cat["name"]:<40} | Rep: {cat["representative"]:<35} | Predictions: {cat["predictions"]}')
            
            if failed_categories:
                self.stdout.write(f'\n❌ FAILED CATEGORIES ({len(failed_categories)}):')
                for cat in failed_categories:
                    self.stdout.write(f'   • {cat["name"]:<40} | Rep: {cat["representative"]:<35} | Reason: {cat["reason"]}')
            
            self.stdout.write(f'\n📈 OVERALL SUCCESS RATE: {len(successful_categories)}/{len(successful_categories) + len(failed_categories)} ({len(successful_categories)/(len(successful_categories) + len(failed_categories))*100:.1f}%)')
        
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
        
        # Evaluate ALL models for this category
        all_models = self.select_best_model(df, category)
        
        if not all_models:
            raise ValueError("No suitable models found for this data")
        
        print(f"📊 Found {len(all_models)} successful models for {category.name}")
        
        # Clear old predictions if force recompute (for all models)
        if force_recompute:
            PreComputedPrediction.objects.filter(category=category).update(is_active=False)
        
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
        
        # Train invitation model ONCE (shared across all models)
        from predictor.ml_models import InvitationPredictor
        invitation_model = None
        invitation_trained = False
        
        try:
            invitation_model = InvitationPredictor(model_type='XGB')
            invitation_metrics = invitation_model.train(df)
            invitation_trained = True
            print(f"✅ Shared invitation model trained successfully with {len(df)} historical draws")
        except Exception as e:
            print(f"⚠️ Failed to train invitation model: {e}")
            invitation_trained = False
        
        total_predictions_created = 0
        
        # 🔄 MAIN LOOP: Create predictions for ALL models
        for model_idx, model_info in enumerate(all_models):
            current_model = model_info['model']
            model_confidence = model_info['confidence']
            model_name = model_info['name']
            
            print(f"\n🔧 Processing Model {model_idx + 1}/{len(all_models)}: {model_name} (confidence: {model_confidence:.3f})")
            
            # Train the current model
            try:
                metrics = current_model.train(df)
                print(f"  ✅ {model_name} trained successfully")
            except Exception as e:
                print(f"  ❌ {model_name} training failed: {e}")
                continue  # Skip this model and move to next
        
            # 🔄 PREDICTION CREATION LOOP for current model
            model_predictions_created = 0
        
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
                    if hasattr(current_model, 'predict'):
                        # Different models have different predict interfaces
                        if current_model.name == 'ARIMA Time Series':
                            # ARIMA models can predict multiple steps
                            predicted_scores = current_model.predict(steps=rank)
                            if isinstance(predicted_scores, list) and len(predicted_scores) >= rank:
                                predicted_score = predicted_scores[rank-1]
                            else:
                                predicted_score = predicted_scores if not isinstance(predicted_scores, list) else predicted_scores[0]
                        elif current_model.name == 'Prophet Time Series':
                            # Prophet uses 'periods' not 'steps'
                            predicted_scores = current_model.predict(periods=rank, freq='2W')
                            if isinstance(predicted_scores, list) and len(predicted_scores) >= rank:
                                predicted_score = predicted_scores[rank-1]
                            else:
                                predicted_score = predicted_scores if not isinstance(predicted_scores, list) else predicted_scores[0]
                        else:
                            # ML models need feature data with same engineering as training
                            # Use clean features for scientifically valid models
                            if hasattr(current_model, 'prepare_clean_features'):
                                features_df = current_model.prepare_clean_features(df)
                            else:
                                # Fallback for legacy models (with warning)
                                features_df = current_model.prepare_features(df)
                            
                            exclude_cols = ['date', 'lowest_crs_score', 'invitations_issued', 'round_number', 'url', 'category']
                            
                            # For Bayesian Hierarchical models, also exclude category dummy variables to match training
                            if hasattr(current_model, 'category_effects'):  # BayesianHierarchicalPredictor
                                feature_cols = [col for col in features_df.columns 
                                              if col not in exclude_cols and not col.startswith('category_')]
                            else:
                                feature_cols = [col for col in features_df.columns if col not in exclude_cols]
                            X = features_df[feature_cols].fillna(0).tail(1)  # Use last row
                            prediction_result = current_model.predict(X)
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
                    if hasattr(current_model, 'predict_with_uncertainty') and hasattr(current_model, 'last_prediction_std'):
                        # Use model's uncertainty estimation for Bayesian models
                        model_std = getattr(current_model, 'last_prediction_std', [score_std])[0] if hasattr(current_model, 'last_prediction_std') else score_std
                        
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
                    elif model_confidence < 0.5:
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
                        'date_earliest': str(date_lower.isoformat()),
                        'date_latest': str(date_upper.isoformat()),
                        'date_margin_days': date_margin_days,
                        'confidence_level': 95
                    }
                    
                    # Adjust confidence score based on uncertainty
                    if data_size <= 4:
                        model_confidence = min(model_confidence * 0.6, 0.4)  # Cap at 40% for very small data
                    elif data_size <= 10:
                        model_confidence = min(model_confidence * 0.8, 0.6)  # Cap at 60% for small data
                    
                    # FINAL NaN SAFETY CHECKS before database save
                    # Ensure all values are valid numbers that can be saved to database
                    if pd.isna(predicted_score) or np.isnan(predicted_score):
                        predicted_score = int(df['lowest_crs_score'].mean() or 450)
                        print(f"⚠️ NaN predicted_score detected, using fallback: {predicted_score}")
                    
                    if pd.isna(predicted_invitations) or np.isnan(predicted_invitations):
                        fallback_invitations = int(df['invitations_issued'].mean() or 2000)
                        predicted_invitations = fallback_invitations
                        print(f"⚠️ NaN predicted_invitations detected, using fallback: {predicted_invitations}")
                    
                    if pd.isna(model_confidence) or np.isnan(model_confidence):
                        model_confidence = 0.3  # Default 30% confidence
                        print(f"⚠️ NaN confidence detected, using fallback: {model_confidence}")
                    
                    # Ensure integer values are properly cast
                    try:
                        predicted_score = int(float(predicted_score))
                        predicted_invitations = int(float(predicted_invitations))
                        model_confidence = float(model_confidence)
                    except (ValueError, TypeError) as e:
                        print(f"⚠️ Value conversion error: {e}, using safe fallbacks")
                        predicted_score = int(df['lowest_crs_score'].mean() or 450)
                        predicted_invitations = int(df['invitations_issued'].mean() or 2000)
                        model_confidence = 0.3
                    
                    # Final bounds checking
                    predicted_score = max(250, min(950, predicted_score))
                    predicted_invitations = max(100, min(10000, predicted_invitations))
                    model_confidence = max(0.1, min(1.0, model_confidence))
                    
                    # Create prediction
                    PreComputedPrediction.objects.update_or_create(
                         category=category,
                         prediction_rank=rank,
                         model_used=str(model_name),
                         defaults={
                            'predicted_date': next_date,
                            'predicted_crs_score': predicted_score,
                            'predicted_invitations': predicted_invitations,
                            'confidence_score': model_confidence,
                            'model_used': str(model_name),
                            'model_version': '1.0',
                            'uncertainty_range': uncertainty_range,
                            'is_active': True
                        }
                    )
                    
                    model_predictions_created += 1
                    total_predictions_created += 1
                    
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f'⚠️  Failed to create prediction {rank}: {str(e)}'))
                    continue
            
            print(f"  📊 {model_name}: Created {model_predictions_created} predictions")
        
        print(f"\n🎉 Total predictions created across all models: {total_predictions_created}")
        return total_predictions_created

    def select_best_model(self, df, category):
        """Select the best model based on statistical performance criteria"""
        
        data_size = len(df)
        
        # For very small datasets, still use specialized predictor as fallback
        if data_size <= 3:
            try:
                from predictor.models import ExpressEntryDraw
                global_draws = ExpressEntryDraw.objects.all().values(
                    'date', 'lowest_crs_score', 'invitations_issued'
                )
                global_df = pd.DataFrame(global_draws)
                small_model = SmallDatasetPredictor(global_data=global_df)
                confidence = 0.2 + (data_size * 0.1)
                return small_model, confidence
            except Exception as e:
                fallback_model = CleanLinearRegressionPredictor()
                fallback_model.name = "Linear Regression (Fallback)"
                return fallback_model, 0.2
        
        # For all other cases, evaluate all models and select the best
        print(f"🔍 Evaluating all available models for {category} ({data_size} data points)...")
        
        try:
            all_models = self._evaluate_all_models_for_storage(df, category)
            print(f"✅ Evaluated {len(all_models)} successful models for storage")
            return all_models
            
        except Exception as e:
            print(f"❌ Model evaluation failed: {e}")
            # Fallback to single best model if evaluation fails
            try:
                best_model, confidence = self._evaluate_all_models(df, category)
                return [{'model': best_model, 'confidence': confidence, 'name': best_model.name}]
            except:
                fallback_model, fallback_confidence = self._fallback_model_selection(df, data_size)
                return [{'model': fallback_model, 'confidence': fallback_confidence, 'name': fallback_model.name}]
    
    def _evaluate_all_models_for_storage(self, df, category):
        """Evaluate all available models and return ALL successful models for storage"""
        
        # ✅ SCIENTIFICALLY VALID MODELS ONLY
        # Start with time series models (always valid)
        models_to_test = []
        data_size = len(df)
        
        # Time series models (no data leakage)
        if data_size >= 8:
            try:
                from predictor.ml_models import ARIMAPredictor
                models_to_test.append(('ARIMA', ARIMAPredictor()))
            except ImportError:
                pass
        
        if data_size >= 10:
            try:
                from predictor.ml_models import LSTMPredictor, ProphetPredictor
                models_to_test.extend([
                    ('LSTM', LSTMPredictor()),
                    ('Prophet', ProphetPredictor()),
                ])
            except ImportError:
                pass
        
        if data_size >= 12:
            try:
                from predictor.ml_models import ExponentialSmoothingPredictor, HoltWintersPredictor
                models_to_test.extend([
                    ('Exponential Smoothing', ExponentialSmoothingPredictor()),
                    ('Holt-Winters', HoltWintersPredictor()),
                ])
            except ImportError:
                pass
        
        # Advanced time series models
        if data_size >= 15:
            try:
                from predictor.ml_models import VARPredictor, DynamicLinearModelPredictor
                models_to_test.extend([
                    ('VAR', VARPredictor()),
                    ('Dynamic Linear Model', DynamicLinearModelPredictor()),
                ])
            except ImportError:
                pass
        
        if data_size >= 20:
            try:
                from predictor.ml_models import SARIMAPredictor
                models_to_test.append(('SARIMA', SARIMAPredictor()))
            except ImportError:
                pass
        
        # Advanced ensemble (use when we have enough models)
        if data_size >= 25:
            try:
                from predictor.ml_models import AdvancedEnsemblePredictor
                models_to_test.append(('Advanced Ensemble', AdvancedEnsemblePredictor()))
            except ImportError:
                pass
        
        # Clean ML models (no data leakage)
        if data_size >= 6:
            try:
                from predictor.ml_models import CleanLinearRegressionPredictor
                models_to_test.append(('Clean Linear Regression', CleanLinearRegressionPredictor()))
            except ImportError:
                pass
        
        if data_size >= 8:
            try:
                from predictor.ml_models import BayesianHierarchicalPredictor
                models_to_test.append(('Bayesian Hierarchical', BayesianHierarchicalPredictor()))
            except ImportError:
                pass
        
        if data_size >= 10:
            try:
                from predictor.ml_models import GaussianProcessPredictor
                models_to_test.append(('Gaussian Process', GaussianProcessPredictor()))
            except ImportError:
                pass
        
        print(f"📊 Testing {len(models_to_test)} models: {[name for name, _ in models_to_test]}")
        
        model_results = {}
        target_col = 'lowest_crs_score'
        
        for name, model in models_to_test:
            try:
                result = self._evaluate_single_model(model, df, target_col, name)
                if result:
                    model_results[name] = result
                    print(f"  ✅ {name}: CV Score={result['cv_score']:.3f}, MAE={result['mae']:.2f}")
                    
            except Exception as e:
                print(f"  ❌ {name}: Failed - {e}")
                continue
        
        if not model_results:
            raise ValueError("No models could be evaluated successfully")
        
        # Calculate confidence for all successful models
        successful_models = []
        for name, result in model_results.items():
            confidence = self._calculate_model_confidence(result, data_size)
            successful_models.append({
                'model': result['model'],
                'confidence': confidence,
                'name': name,
                'performance': {
                    'cv_score': result['cv_score'],
                    'mae': result['mae'],
                    'rmse': result.get('rmse', 0),
                    'r2': result.get('r2', 0)
                }
            })
        
        # Sort by confidence (best first)
        successful_models.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"🎯 All successful models ranked by confidence:")
        for i, model_info in enumerate(successful_models):
            print(f"  {i+1}. {model_info['name']}: {model_info['confidence']:.3f}")
        
        return successful_models

    def _evaluate_all_models(self, df, category):
        """Evaluate all available models and select the best based on statistical criteria"""
        
        # ✅ SCIENTIFICALLY VALID MODELS ONLY
        # Start with time series models (always valid)
        models_to_test = []
        data_size = len(df)
        
        # Time series models (no data leakage)
        if data_size >= 8:
            try:
                from predictor.ml_models import ARIMAPredictor
                models_to_test.append(('ARIMA', ARIMAPredictor()))
            except ImportError:
                pass
        
        if data_size >= 10:
            try:
                from predictor.ml_models import LSTMPredictor, ProphetPredictor
                models_to_test.extend([
                    ('LSTM', LSTMPredictor()),
                    ('Prophet', ProphetPredictor()),
                ])
            except ImportError:
                pass
        
        if data_size >= 12:
            try:
                from predictor.ml_models import ExponentialSmoothingPredictor, HoltWintersPredictor
                models_to_test.extend([
                    ('Exponential Smoothing', ExponentialSmoothingPredictor()),
                    ('Holt-Winters', HoltWintersPredictor()),
                ])
            except ImportError:
                pass
        
        # Advanced time series models
        if data_size >= 15:
            try:
                from predictor.ml_models import VARPredictor, DynamicLinearModelPredictor
                models_to_test.extend([
                    ('VAR', VARPredictor()),
                    ('Dynamic Linear Model', DynamicLinearModelPredictor()),
                ])
            except ImportError:
                pass
        
        if data_size >= 20:
            try:
                from predictor.ml_models import SARIMAPredictor
                models_to_test.append(('SARIMA', SARIMAPredictor()))
            except ImportError:
                pass
        
        # Advanced ensemble (use when we have enough models)
        if data_size >= 25:
            try:
                from predictor.ml_models import AdvancedEnsemblePredictor
                models_to_test.append(('Advanced Ensemble', AdvancedEnsemblePredictor()))
            except ImportError:
                pass
        
        # Clean ML models (no data leakage)
        if data_size >= 6:
            try:
                from predictor.ml_models import CleanLinearRegressionPredictor
                models_to_test.append(('Clean Linear Regression', CleanLinearRegressionPredictor()))
            except ImportError:
                pass
        
        if data_size >= 8:
            try:
                from predictor.ml_models import BayesianHierarchicalPredictor
                models_to_test.append(('Bayesian Hierarchical', BayesianHierarchicalPredictor()))
            except ImportError:
                pass
        
        if data_size >= 10:
            try:
                from predictor.ml_models import GaussianProcessPredictor
                models_to_test.append(('Gaussian Process', GaussianProcessPredictor()))
            except ImportError:
                pass
        
        print(f"📊 Testing {len(models_to_test)} models: {[name for name, _ in models_to_test]}")
        
        model_results = {}
        target_col = 'lowest_crs_score'
        
        for name, model in models_to_test:
            try:
                result = self._evaluate_single_model(model, df, target_col, name)
                if result:
                    model_results[name] = result
                    print(f"  ✅ {name}: CV Score={result['cv_score']:.3f}, MAE={result['mae']:.2f}")
                    
            except Exception as e:
                print(f"  ❌ {name}: Failed - {e}")
                continue
        
        if not model_results:
            raise ValueError("No models could be evaluated successfully")
        
        # Select best model using multi-criteria approach
        best_model_name, best_result = self._select_best_model_multi_criteria(model_results)
        
        # Calculate confidence based on model performance
        confidence = self._calculate_model_confidence(best_result, data_size)
        
        return best_result['model'], confidence
    
    def _evaluate_single_model(self, model, df, target_col, model_name):
        """Evaluate a single model using cross-validation and performance metrics"""
        
        if len(df) < 5:
            return None  # Need minimum data for evaluation
        
        try:
            # Calculate missing features that models may need
            df_enhanced = df.copy()
            df_enhanced['date'] = pd.to_datetime(df_enhanced['date'])
            df_enhanced = df_enhanced.sort_values('date')
            
            # Calculate days_since_last_draw (required by prepare_features)
            df_enhanced['days_since_last_draw'] = df_enhanced['date'].diff().dt.days.fillna(14)
            
            # Initialize CV scores
            cv_scores = []
            
            # All models are now scientifically valid - enable CV for ML models with clean features
            if model_name in ['Clean Linear Regression', 'Bayesian Hierarchical', 'Gaussian Process']:
                # These models use clean features - can do cross-validation
                features = model.prepare_clean_features(df_enhanced)
                exclude_cols = ['date', 'lowest_crs_score', 'invitations_issued', 'round_number', 'url', 'category']
                feature_cols = [col for col in features.columns if col not in exclude_cols]
                X = features[feature_cols].fillna(0)
                y = df_enhanced[target_col]
            else:
                # For time series models, use simple time-based features
                X = df_enhanced.drop(columns=[target_col]).select_dtypes(include=[np.number])
                y = df_enhanced[target_col]
                
                if X.empty or len(X.columns) == 0:
                    # For time series models, use index as feature
                    X = pd.DataFrame({'time_index': range(len(df_enhanced))})
            
            # Cross-validation evaluation (all models are now scientifically valid)
            n_folds = min(3, len(df) // 2)  # Adaptive fold count
            
            if n_folds >= 2:
                
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=n_folds, shuffle=False)  # No shuffle for time series
                
                for train_idx, val_idx in kf.split(X):
                    try:
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        
                        # Train model
                        train_df = df_enhanced.iloc[train_idx].copy()
                        model_copy = self._copy_model(model)
                        
                        # Handle different train() method signatures
                        if model_name in ['ARIMA', 'LSTM', 'Prophet', 'Exponential Smoothing', 'SARIMA', 'VAR', 'Holt-Winters', 'Dynamic Linear Model']:
                            model_copy.train(train_df)  # These models don't take target_col
                        else:
                            model_copy.train(train_df, target_col)
                        
                        # Predict
                        if hasattr(model_copy, 'predict'):
                            pred = model_copy.predict(X_val)
                            if isinstance(pred, (list, np.ndarray)):
                                pred = pred[0] if len(pred) > 0 else y_val.mean()
                            
                            # Calculate score (negative MAE for cross-validation)
                            mae = abs(pred - y_val.iloc[0]) if len(y_val) > 0 else 0
                            cv_scores.append(-mae)
                            
                    except Exception as e:
                        continue
            
            # Full model training for final metrics
            if model_name in ['ARIMA', 'LSTM', 'Prophet', 'Exponential Smoothing', 'SARIMA', 'VAR', 'Holt-Winters', 'Dynamic Linear Model', 'Advanced Ensemble']:
                model.train(df_enhanced)  # These models don't take target_col
            else:
                model.train(df_enhanced, target_col)
            
            # Calculate final metrics
            if hasattr(model, 'metrics') and model.metrics:
                mae = model.metrics.get('mae', np.inf)
                r2 = model.metrics.get('r2', -np.inf)
            else:
                mae = np.inf
                r2 = -np.inf
            
            # Information criteria approximation
            aic, bic = self._calculate_information_criteria(model, df, target_col)
            
            return {
                'model': model,
                'cv_score': np.mean(cv_scores) if cv_scores else -mae,
                'cv_std': np.std(cv_scores) if cv_scores else 0,
                'mae': mae,
                'r2': r2,
                'aic': aic,
                'bic': bic,
                'n_cv_folds': len(cv_scores)
            }
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            return None
    
    def _copy_model(self, model):
        """Create a copy of the model for cross-validation"""
        model_class = type(model)
        try:
            return model_class()
        except:
            # Fallback to the original model if copying fails
            return model
    
    def _calculate_information_criteria(self, model, df, target_col):
        """Calculate AIC and BIC for model comparison"""
        
        try:
            # For models with built-in criteria
            if hasattr(model, 'aic') and hasattr(model, 'bic'):
                return getattr(model, 'aic', np.inf), getattr(model, 'bic', np.inf)
            
            # Approximate for other models
            n = len(df)
            if hasattr(model, 'metrics') and model.metrics:
                mse = model.metrics.get('mse', 1.0)
            else:
                # Fallback calculation
                y_true = df[target_col]
                y_pred = model.predict(df.drop(columns=[target_col]).select_dtypes(include=[np.number]))
                if isinstance(y_pred, (list, np.ndarray)):
                    y_pred = y_pred[0] if len(y_pred) > 0 else y_true.mean()
                mse = ((y_true - y_pred) ** 2).mean()
            
            # Estimate number of parameters
            n_params = self._estimate_model_parameters(model)
            
            # Calculate AIC and BIC
            log_likelihood = -0.5 * n * (np.log(2 * np.pi * mse) + 1)
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + n_params * np.log(n)
            
            return aic, bic
            
        except Exception as e:
            return np.inf, np.inf
    
    def _estimate_model_parameters(self, model):
        """Estimate the number of parameters for a model"""
        
        if hasattr(model, 'n_estimators'):  # Random Forest, XGBoost
            return getattr(model, 'n_estimators', 100) * 2
        elif hasattr(model, 'coef_'):  # Linear models
            return len(getattr(model, 'coef_', [1]))
        elif 'ARIMA' in str(type(model)):
            return 5  # Typical ARIMA parameters
        elif 'LSTM' in str(type(model)):
            return 50  # Estimate for LSTM
        else:
            return 3  # Default fallback
    
    def _select_best_model_multi_criteria(self, model_results):
        """Select the best model using multiple statistical criteria"""
        
        if len(model_results) == 1:
            return list(model_results.items())[0]
        
        # Normalize scores for comparison
        scores = {}
        criteria = ['cv_score', 'r2', 'mae', 'aic', 'bic']
        
        # Extract values for normalization
        criteria_values = {criterion: [] for criterion in criteria}
        for result in model_results.values():
            for criterion in criteria:
                value = result.get(criterion, 0)
                if not np.isfinite(value):
                    value = 0 if criterion in ['cv_score', 'r2'] else 1000
                criteria_values[criterion].append(value)
        
        # Calculate composite scores
        for name, result in model_results.items():
            score = 0
            
            # CV Score (higher is better) - weight: 40% if available, otherwise redistribute
            cv_score = result.get('cv_score', 0)
            n_cv_folds = result.get('n_cv_folds', 0)
            
            if n_cv_folds > 0 and np.isfinite(cv_score):
                cv_norm = self._normalize_score(cv_score, criteria_values['cv_score'], higher_better=True)
                score += cv_norm * 0.4
                cv_weight_used = 0.4
            else:
                # No CV available, redistribute weight to other metrics
                cv_weight_used = 0.0
            
            # Redistribute weights if CV not available
            r2_weight = 0.25 + (cv_weight_used == 0.0) * 0.2  # 25% or 45% if no CV
            mae_weight = 0.2 + (cv_weight_used == 0.0) * 0.2   # 20% or 40% if no CV
            
            # R² (higher is better)
            r2 = result.get('r2', 0)
            if not np.isfinite(r2):
                r2 = min(criteria_values['r2'])
            r2_norm = self._normalize_score(r2, criteria_values['r2'], higher_better=True)
            score += r2_norm * r2_weight
            
            # MAE (lower is better)
            mae = result.get('mae', np.inf)
            if not np.isfinite(mae):
                mae = max(criteria_values['mae'])
            mae_norm = self._normalize_score(mae, criteria_values['mae'], higher_better=False)
            score += mae_norm * mae_weight
            
            # AIC (lower is better) - weight: 10%
            aic = result.get('aic', np.inf)
            if not np.isfinite(aic):
                aic = max(criteria_values['aic'])
            aic_norm = self._normalize_score(aic, criteria_values['aic'], higher_better=False)
            score += aic_norm * 0.1
            
            # BIC (lower is better) - weight: 5%
            bic = result.get('bic', np.inf)
            if not np.isfinite(bic):
                bic = max(criteria_values['bic'])
            bic_norm = self._normalize_score(bic, criteria_values['bic'], higher_better=False)
            score += bic_norm * 0.05
            
            scores[name] = score
        
        # Select model with highest score
        best_model_name = max(scores.items(), key=lambda x: x[1])[0]
        print(f"🏆 Model ranking: {dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))}")
        
        return best_model_name, model_results[best_model_name]
    
    def _normalize_score(self, value, all_values, higher_better=True):
        """Normalize a score between 0 and 1"""
        
        if len(all_values) <= 1:
            return 1.0
        
        min_val = min(all_values)
        max_val = max(all_values)
        
        if max_val == min_val:
            return 1.0
        
        if higher_better:
            return (value - min_val) / (max_val - min_val)
        else:
            return (max_val - value) / (max_val - min_val)
    
    def _calculate_model_confidence(self, result, data_size):
        """Calculate confidence score based on model performance and data size"""
        
        base_confidence = 0.5
        
        # CV score contribution (30%)
        cv_score = result.get('cv_score', -100)
        cv_confidence = max(0, min(1, (cv_score + 50) / 100)) * 0.3
        
        # R² contribution (25%)
        r2 = result.get('r2', 0)
        r2_confidence = max(0, min(1, r2)) * 0.25
        
        # Data size contribution (20%)
        size_confidence = min(1, data_size / 50) * 0.2
        
        # Cross-validation folds contribution (15%)
        n_folds = result.get('n_cv_folds', 0)
        cv_folds_confidence = min(1, n_folds / 3) * 0.15
        
        # MAE contribution (10%)
        mae = result.get('mae', 100)
        mae_confidence = max(0, min(1, (100 - mae) / 100)) * 0.1
        
        total_confidence = base_confidence + cv_confidence + r2_confidence + size_confidence + cv_folds_confidence + mae_confidence
        
        return min(0.95, max(0.1, total_confidence))
    
    def _fallback_model_selection(self, df, data_size):
        """Fallback to simple data-size based selection if evaluation fails"""
        
        print(f"🔄 Using fallback model selection for {data_size} data points")
        
        if data_size <= 10:
            model = CleanLinearRegressionPredictor()
            model.name = "Linear Regression (Fallback)"
            confidence = 0.3 + (data_size * 0.02)
        elif data_size <= 20:
            model = BayesianHierarchicalPredictor()
            model.name = "Bayesian Hierarchical (Fallback)"
            confidence = 0.5 + (data_size * 0.01)
        else:
            try:
                from predictor.ml_models import GaussianProcessPredictor
                model = GaussianProcessPredictor()
                model.name = "Gaussian Process (Fallback)"
                confidence = 0.7
            except ImportError:
                model = BayesianHierarchicalPredictor()
                model.name = "Bayesian Hierarchical (Fallback)"
                confidence = 0.6
        
        return model, confidence

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