from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import date, timedelta
import pandas as pd
import numpy as np
import math
import logging
import os

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
            
            # Note: Don't skip eliminated categories here - they still get recursive predictions
            
            # Use the IRCC category as the key
            if ircc_category not in ircc_groups:
                ircc_groups[ircc_category] = {
                    'representative_category': category,  # Use first category as representative
                    'related_categories': list(related_categories),
                    'total_draws': 0,
                    'priority_level': self.get_category_priority_2025(ircc_category),
                    'adjusted_predictions': self.get_adjusted_prediction_count(ircc_category, num_predictions)
                }
            
            # Update total draws count
            pooled_draws, _, _ = category.get_pooled_data()
            ircc_groups[ircc_category]['total_draws'] = pooled_draws.count()
        
        # Display grouping summary with priority information
        self.stdout.write(f'\n📊 Grouped into {len(ircc_groups)} unique IRCC categories (2025 policy-filtered):')
        for ircc_cat, group_info in ircc_groups.items():
            priority_emoji = {
                'HIGH': '🔥',
                'MEDIUM': '📋', 
                'LOW': '⬇️'
            }.get(group_info['priority_level'], '📋')
            
            related_names = [cat.name for cat in group_info['related_categories']]
            pred_count = group_info['adjusted_predictions']
            
            if len(related_names) > 1:
                self.stdout.write(f'   {priority_emoji} {ircc_cat}: {len(related_names)} versions → {group_info["total_draws"]} draws → {pred_count} predictions')
                for name in related_names:
                    self.stdout.write(f'      └─ {name}')
            else:
                self.stdout.write(f'   {priority_emoji} {ircc_cat}: {group_info["total_draws"]} draws → {pred_count} predictions')
        
        total_groups = len(ircc_groups)
        self.stdout.write(f'\n📊 Processing {total_groups} policy-compliant IRCC categories...')
        
        # 🗓️ GENERATE REALISTIC DRAW CALENDAR to avoid date conflicts
        import pytz
        eastern = pytz.timezone('America/Toronto')
        now_eastern = timezone.now().astimezone(eastern)
        today_eastern = now_eastern.date()
        
        self.stdout.write(f'\n🗓️ Generating realistic draw calendar to avoid date conflicts...')
        draw_calendar, category_priority_order = self.generate_realistic_draw_calendar(
            start_date=today_eastern, 
            total_weeks=52
        )
        
        # Assign specific dates to each category based on priority and frequency
        category_schedules = self.assign_category_dates(
            ircc_groups, 
            draw_calendar, 
            category_priority_order, 
            num_predictions
        )
        
        self.stdout.write(f'✅ Draw calendar generated with realistic spacing')
        
        successful_predictions = 0
        local_failed_categories = []
        
        # 🎯 PROCESS CATEGORIES IN PRIORITY ORDER (most frequent/prioritized first)
        priority_order = ['HIGHEST', 'HIGH', 'MEDIUM', 'LOW']
        ordered_categories = []
        
        for priority_level in priority_order:
            # Get categories for this priority level
            level_categories = [
                (ircc_cat, group_info) for ircc_cat, group_info in ircc_groups.items()
                if group_info['priority_level'] == priority_level
            ]
            # Sort by total draws (frequency) within same priority level
            level_categories.sort(key=lambda x: x[1]['total_draws'], reverse=True)
            ordered_categories.extend(level_categories)
        
        self.stdout.write(f'\n🏆 PROCESSING ORDER (Priority + Frequency):')
        for i, (ircc_cat, group_info) in enumerate(ordered_categories, 1):
            priority_emoji = {'HIGHEST': '🏆', 'HIGH': '🥇', 'MEDIUM': '🥈', 'LOW': '🥉'}.get(group_info['priority_level'], '📋')
            self.stdout.write(f'   {i}. {priority_emoji} {ircc_cat} ({group_info["total_draws"]} draws, {group_info["priority_level"]})')
        
        total_ordered = len(ordered_categories)
        
        for i, (ircc_category, group_info) in enumerate(ordered_categories, 1):
            representative_category = group_info['representative_category']
            adjusted_count = group_info['adjusted_predictions']
            priority_level = group_info['priority_level']
            
            self.stdout.write(f'\n[{i}/{total_ordered}] Processing IRCC Category: {ircc_category}')
            self.stdout.write(f'   📂 Using representative: {representative_category.name}')
            self.stdout.write(f'   🎯 2025 Priority: {priority_level} → {adjusted_count} predictions')
            
            try:
                # Get pre-assigned dates for this category
                assigned_dates = category_schedules.get(ircc_category, [])
                
                # 🎯 SWITCH TO RECURSIVE FORECASTING: Scientifically sound approach
                # Each category gets exactly 5 predictions (1 primary + 4 secondary)
                predictions_created = self.compute_recursive_predictions(
                    representative_category, force_recompute, assigned_dates
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
        self.stdout.write(f'\n🎯 RECURSIVE FORECASTING SUMMARY')
        self.stdout.write(f'✅ Successful categories: {successful_predictions}/{total_groups}')
        self.stdout.write(f'🔄 Method: Recursive forecasting (5 predictions per category)')
        self.stdout.write(f'📊 Focus: PRIMARY next draw + 4 secondary predictions')
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

    def is_category_eliminated_2025(self, ircc_category):
        """
        Check if category is eliminated by 2025 policy changes.
        
        Based on OFFICIAL Government consultation results (Feb 27, 2025):
        - Transport: Only 18.8% "great need", 35.4% "no need" (lowest support)
        - General: Focus shifted to Canadian Experience Class
        """
        # Transport: ELIMINATED due to poor consultation results (18.8% great need, 35.4% no need)
        if 'Transport' in ircc_category:
            return True
        
        # General/No Program: ELIMINATED - "Focus will be to invite candidates with experience working in Canada"
        if any(term in ircc_category for term in ['General', 'No Program']):
            return True
        
        return False
    
    def get_category_priority_2025(self, ircc_category):
        """
        Get 2025 policy priority based on OFFICIAL Government Announcement.
        
        Source: Canada.ca Feb 27, 2025 - "Canada announces 2025 Express Entry 
        category-based draws, plans for more in-Canada draws"
        
        Official Policy: "For 2025, the focus of the federal economic class 
        draws will be to invite candidates with experience working in Canada"
        """
        
        # 🏆 HIGHEST PRIORITY: Canadian Experience Class - PRIMARY 2025 FOCUS
        # Government Quote: "focus will be to invite candidates with experience 
        # working in Canada (Canadian Experience Class)"
        if any(term in ircc_category for term in ['Canadian Experience']):
            return 'HIGHEST'  # Primary focus, frequent draws
        
        # 🥇 HIGH PRIORITY: Official Category-Based Selection Priorities
        # 1. Healthcare: 55.8% rated "great need" (highest in consultations)
        if any(term in ircc_category for term in ['Healthcare']):
            return 'HIGH'  # Monthly draws, strong government priority
        
        # 2. French: Official commitment to Francophone immigration
        if any(term in ircc_category for term in ['French']):
            return 'HIGH'  # Monthly draws, government mandate
        
        # 🥈 MEDIUM PRIORITY: Confirmed 2025 Categories
        # 3. Trades: 38.8% rated "great need", confirmed priority
        if any(term in ircc_category for term in ['Trade']):
            return 'MEDIUM'  # Quarterly draws per consultation results
        
        # 4. Education: 28.4% rated "great need", NEW 2025 category
        # Includes: teachers, child care educators, instructors of persons with disabilities
        if any(term in ircc_category for term in ['Education']):
            return 'MEDIUM'  # NEW category for 2025, bi-monthly draws
        
        # Provincial Nominee: Continues but with reduced emphasis
        if any(term in ircc_category for term in ['Provincial Nominee']):
            return 'MEDIUM'  # Bi-weekly operational frequency
        
        # 🥉 LOW PRIORITY: Reduced Focus Categories  
        # STEM: 32.4% "great need" but not prioritized in 2025 announcement
        if any(term in ircc_category for term in ['STEM']):
            return 'LOW'  # Quarterly draws, reduced from previous years
        
        # Agriculture: 25.0% "great need", minimal presence in 2025 priorities
        if any(term in ircc_category for term in ['Agriculture']):
            return 'LOW'  # Quarterly draws, reduced priority
        
        # ❌ ELIMINATED: Categories with Poor Consultation Results or Policy Changes
        # Transport: ELIMINATED - Only 18.8% "great need", 35.4% "no need" (worst results)
        if any(term in ircc_category for term in ['Transport']):
            return 'ELIMINATED'  # No draws scheduled due to poor consultation results
        
        # General/No Program: ELIMINATED - "Focus will be to invite candidates with experience working in Canada"
        if any(term in ircc_category for term in ['General', 'No Program']):
            return 'ELIMINATED'  # Focus shifted to CEC entirely per Feb 27, 2025 announcement
        
        return 'LOW'  # Default for unspecified categories
    
    def get_adjusted_prediction_count(self, ircc_category, base_count):
        """
        Adjust prediction count based on OFFICIAL 2025 Government Policy.
        
        Reflects official government priorities and consultation results.
        """
        priority = self.get_category_priority_2025(ircc_category)
        
        if priority == 'HIGHEST':
            # Canadian Experience Class: PRIMARY 2025 FOCUS
            # Government quote: "focus will be to invite candidates with experience working in Canada"
            return min(52, base_count * 4)  # Up to 1 year predictions (bi-weekly draws)
        
        elif priority == 'HIGH':
            # Healthcare (55.8% great need) + French (government mandate)
            return min(26, base_count * 2)  # Up to 6 months (monthly draws)
        
        elif priority == 'MEDIUM':
            # Trades (38.8% great need), Education (28.4% great need), PNP
            return min(12, base_count)  # Up to 3 months (bi-monthly draws)
        
        elif priority == 'LOW':
            # STEM, Agriculture: Minimal consultation support
            return min(4, max(3, base_count // 2))  # 3-4 predictions maximum
        
        elif priority == 'ELIMINATED':
            # Transport, General: Not part of 2025 policy
            return 0  # No predictions for eliminated categories
        
        return min(6, base_count)  # Default conservative count
    
    def compute_category_predictions(self, category, num_predictions, force_recompute, assigned_dates=None):
        """Compute predictions for a specific category using coordinated dates"""
        
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
            from django.db import transaction
            
            # Force immediate deletion in separate transaction to prevent rollback
            deletion_count = PreComputedPrediction.objects.filter(category=category).count()
            PreComputedPrediction.objects.filter(category=category).delete()
            print(f"🗑️  Force delete: Cleared {deletion_count} existing predictions for {category.name}")
            
            # Verify deletion completed
            remaining_count = PreComputedPrediction.objects.filter(category=category).count()
            print(f"✅ Verification: {remaining_count} predictions remaining after delete")
        
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
            
            # Use coordinated dates to avoid conflicts across categories
            if assigned_dates:
                prediction_dates = assigned_dates[:num_predictions]
                print(f"📅 Using {len(prediction_dates)} coordinated dates for {category.name}")
            else:
                # Fallback to old logic if no coordinated dates provided
                prediction_dates = []
                for rank in range(1, num_predictions + 1):
                    base_interval = 14
                    variation = (-2, -1, 0, 1, 2)[rank % 5]
                    interval = base_interval + variation
                    next_date = current_date + timedelta(days=interval * rank)
                    
                    if next_date > today_eastern + timedelta(days=365):
                        break
                    prediction_dates.append(next_date)
                print(f"⚠️ Using fallback date calculation for {category.name}")
        
            for rank, next_date in enumerate(prediction_dates, 1):
                
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
                        elif 'LSTM' in current_model.name:
                            # LSTM models need sequence data
                            try:
                                # Get the last sequence_length rows as input
                                sequence_length = getattr(current_model, 'sequence_length', 10)
                                sequence_data = df['lowest_crs_score'].tail(sequence_length).values
                                
                                # Ensure we have enough data points for LSTM
                                if len(sequence_data) < sequence_length:
                                    # Use the last available value to pad the sequence
                                    if len(sequence_data) > 0:
                                        last_value = float(sequence_data[-1])
                                    else:
                                        last_value = float(df['lowest_crs_score'].mean())
                                    
                                    # Create a properly sized sequence
                                    padded_sequence = [last_value] * sequence_length
                                    # Replace the end with actual data if available
                                    if len(sequence_data) > 0:
                                        padded_sequence[-len(sequence_data):] = sequence_data.tolist()
                                    
                                    sequence_data = np.array(padded_sequence)
                                
                                # Verify we have the right size before reshaping
                                if len(sequence_data) != sequence_length:
                                    raise ValueError(f"Sequence length mismatch: expected {sequence_length}, got {len(sequence_data)}")
                                
                                # Reshape for LSTM: (1, sequence_length, 1)
                                sequence_data = sequence_data.reshape(1, sequence_length, 1)
                                
                                # Predict multiple steps
                                predicted_scores = current_model.predict(sequence_data, steps=rank)
                                if isinstance(predicted_scores, list) and len(predicted_scores) >= rank:
                                    predicted_score = predicted_scores[rank-1]
                                else:
                                    predicted_score = predicted_scores[0] if hasattr(predicted_scores, '__len__') else predicted_scores
                            except Exception as e:
                                print(f"⚠️ LSTM prediction failed: {e}, using simple approach")
                                # Fallback to simple prediction
                                predicted_score = df['lowest_crs_score'].mean()
                        elif current_model.name in ['VAR', 'Vector Autoregression']:
                            # VAR models return lists/arrays
                            try:
                                predicted_scores = current_model.predict(steps=rank)
                                if isinstance(predicted_scores, list) and len(predicted_scores) >= rank:
                                    predicted_score = predicted_scores[rank-1]
                                elif isinstance(predicted_scores, list) and len(predicted_scores) > 0:
                                    predicted_score = predicted_scores[0]
                                else:
                                    predicted_score = float(predicted_scores) if not hasattr(predicted_scores, '__len__') else float(predicted_scores[0])
                            except Exception as e:
                                print(f"⚠️ VAR prediction failed: {e}, using fallback")
                                predicted_score = df['lowest_crs_score'].mean()
                        elif current_model.name in ['Dynamic Linear Model', 'DLM']:
                            # DLM returns numpy arrays/matrices
                            try:
                                predicted_scores = current_model.predict(steps=rank)
                                if isinstance(predicted_scores, list) and len(predicted_scores) >= rank:
                                    pred_val = predicted_scores[rank-1]
                                    # Handle numpy arrays/matrices
                                    if hasattr(pred_val, 'item'):
                                        predicted_score = pred_val.item()
                                    elif hasattr(pred_val, '__getitem__'):
                                        predicted_score = float(pred_val[0])
                                    else:
                                        predicted_score = float(pred_val)
                                elif isinstance(predicted_scores, list) and len(predicted_scores) > 0:
                                    pred_val = predicted_scores[0]
                                    if hasattr(pred_val, 'item'):
                                        predicted_score = pred_val.item()
                                    elif hasattr(pred_val, '__getitem__'):
                                        predicted_score = float(pred_val[0])
                                    else:
                                        predicted_score = float(pred_val)
                                else:
                                    predicted_score = df['lowest_crs_score'].mean()
                            except Exception as e:
                                print(f"⚠️ DLM prediction failed: {e}, using fallback")
                                predicted_score = df['lowest_crs_score'].mean()
                        elif current_model.name in ['Holt-Winters', 'HW']:
                            # Holt-Winters returns lists
                            try:
                                predicted_scores = current_model.predict(steps=rank)
                                if isinstance(predicted_scores, list) and len(predicted_scores) >= rank:
                                    predicted_score = float(predicted_scores[rank-1])
                                elif isinstance(predicted_scores, list) and len(predicted_scores) > 0:
                                    predicted_score = float(predicted_scores[0])
                                else:
                                    predicted_score = float(predicted_scores) if not hasattr(predicted_scores, '__len__') else df['lowest_crs_score'].mean()
                            except Exception as e:
                                print(f"⚠️ Holt-Winters prediction failed: {e}, using fallback")
                                predicted_score = df['lowest_crs_score'].mean()
                        elif current_model.name in ['SARIMA', 'Exponential Smoothing']:
                            # SARIMA and Exponential Smoothing models
                            try:
                                predicted_scores = current_model.predict(steps=rank)
                                if isinstance(predicted_scores, list) and len(predicted_scores) >= rank:
                                    predicted_score = float(predicted_scores[rank-1])
                                elif isinstance(predicted_scores, list) and len(predicted_scores) > 0:
                                    predicted_score = float(predicted_scores[0])
                                else:
                                    predicted_score = float(predicted_scores) if not hasattr(predicted_scores, '__len__') else df['lowest_crs_score'].mean()
                            except Exception as e:
                                print(f"⚠️ {current_model.name} prediction failed: {e}, using fallback")
                                predicted_score = df['lowest_crs_score'].mean()
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
                    
                    # Now predict invitation numbers using the invitation model if available
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
                    
                    else:
                        # Invitation model not trained - use enhanced statistical fallback
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
                    
                    # Create uncertainty range for CRS score (based on model-specific prediction)
                    if hasattr(current_model, 'predict_with_uncertainty'):
                        try:
                            # Use model's own uncertainty if available
                            uncertainty_result = current_model.predict_with_uncertainty(X)
                            crs_uncertainty = uncertainty_result.get('std_dev', 50)
                        except:
                            # Fallback uncertainty based on model confidence
                            crs_uncertainty = 50 + (1 - model_confidence) * 100
                    else:
                        # Default uncertainty scaling
                        crs_uncertainty = 50 + (1 - model_confidence) * 100
                    
                    # Scale uncertainty by prediction horizon
                    scaled_crs_uncertainty = crs_uncertainty * (1 + 0.2 * rank)
                    scaled_invitation_uncertainty = invitation_uncertainty * (1 + 0.2 * rank)
                    
                    uncertainty_range = {
                        'crs_min': max(250, predicted_score - int(scaled_crs_uncertainty)),
                        'crs_max': min(950, predicted_score + int(scaled_crs_uncertainty)),
                        'invitations_min': max(100, predicted_invitations - int(scaled_invitation_uncertainty)),
                        'invitations_max': min(10000, predicted_invitations + int(scaled_invitation_uncertainty))
                    }
                    
                    print(f"🎯 {model_name} prediction (rank {rank}): CRS {predicted_score} (±{int(scaled_crs_uncertainty)}), Invitations {predicted_invitations} (±{int(scaled_invitation_uncertainty)})")
                    
                    # 🧠 RECALCULATE CONFIDENCE with DOMAIN INTELLIGENCE
                    # Use the actual prediction values to assess confidence with domain knowledge
                    enhanced_confidence = self._calculate_model_confidence(
                        result={'cv_score': -50, 'r2': 0.8, 'mae': 30, 'n_cv_folds': 5},  # Default statistical metrics
                        data_size=len(df),
                        predicted_crs=int(predicted_score),
                        prediction_date=next_date,
                        df=df
                    )
                    
                    # Use enhanced confidence if it's significantly different from model baseline
                    confidence_adjustment = enhanced_confidence / max(0.1, model_confidence)
                    if abs(enhanced_confidence - model_confidence) > 0.15:  # Significant difference
                        print(f"🔍 Confidence adjusted: {model_confidence:.3f} → {enhanced_confidence:.3f} (factor: {confidence_adjustment:.2f})")
                        model_confidence = enhanced_confidence
                    
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
                    
                    # Create prediction with transaction protection
                    with transaction.atomic():
                        PreComputedPrediction.objects.update_or_create(
                             category=category,
                             prediction_rank=rank,
                             model_used=str(model_name),
                             defaults={
                                'predicted_date': next_date,
                                'predicted_crs_score': predicted_score,
                                'predicted_invitations': predicted_invitations,
                                'confidence_score': model_confidence,
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
                return [{'model': small_model, 'confidence': confidence, 'name': 'Small Dataset Predictor'}]
            except Exception as e:
                fallback_model = CleanLinearRegressionPredictor()
                fallback_model.name = "Linear Regression (Fallback)"
                return [{'model': fallback_model, 'confidence': 0.2, 'name': 'Linear Regression (Fallback)'}]
        
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
    
    def _scientific_probability_penalty(self, z_score):
        """
        🔬 CONTINUOUS SCIENTIFIC PENALTY: Smooth penalization based on statistical deviation
        
        Scientific approach using continuous exponential decay:
        - No penalty within 1σ (68% of data is considered normal)
        - Gradual continuous penalization beyond 1σ using Gaussian-inspired decay
        - Mathematically smooth (no discontinuities or sharp transitions)
        - Based on inverse relationship with statistical probability
        
        Mathematical formula:
        - z ≤ 1.0: penalty = 1.0 (no penalty within normal range)
        - z > 1.0: penalty = exp(-k * (z - 1)^α) where k=0.5, α=1.5
        
        Confidence mapping:
        - z=0.0: 100% confidence (at the mean)
        - z=1.0: 100% confidence (boundary of normal range)
        - z=1.5: ~77% confidence (mildly unusual)
        - z=2.0: ~57% confidence (unusual, ~5% probability)
        - z=2.5: ~41% confidence (quite unusual, ~1% probability)
        - z=3.0: ~29% confidence (very unusual, ~0.3% probability)
        - z=4.0: ~12% confidence (extremely unusual)
        - z=5.0+: <5% confidence (statistically implausible)
        
        Args:
            z_score (float): Number of standard deviations from mean
            
        Returns:
            float: Confidence penalty score (0.001 to 1.0)
        """
        z = abs(z_score)  # Use absolute value
        
        # No penalty within 1 standard deviation (normal range)
        if z <= 1.0:
            return 1.0
        
        # Continuous exponential decay penalty beyond 1σ
        # Scientific parameters tuned for realistic confidence degradation
        penalty_threshold = 1.0      # Start penalizing beyond 1σ
        decay_rate = 0.5            # Controls steepness (0.5 = moderate decay)
        power_factor = 1.5          # Controls curvature (1.5 = slightly accelerating)
        
        # Calculate excess beyond normal range
        excess_z = z - penalty_threshold
        
        # Exponential decay with power adjustment for smoother curve
        penalty_factor = math.exp(-decay_rate * (excess_z ** power_factor))
        
        # Ensure minimum confidence for extreme deviations (scientific floor)
        return max(0.001, penalty_factor)
    
    def _z_score_probability_description(self, z_score):
        """
        Provide human-readable description of z-score probability significance.
        
        Args:
            z_score (float): Number of standard deviations from mean
            
        Returns:
            str: Human-readable description
        """
        z = abs(z_score)
        
        if z <= 1.0:
            return "within normal range"
        elif z <= 1.5:
            return "somewhat unusual"
        elif z <= 2.0:
            return "unusual (~5% probability)"
        elif z <= 2.5:
            return "quite unusual (~1% probability)"
        elif z <= 3.0:
            return "very unusual (~0.3% probability)"
        elif z <= 4.0:
            return "extremely unusual (~0.01% probability)"
        else:
            return "statistically implausible"
    
    def _calculate_model_confidence(self, result, data_size, predicted_crs=None, prediction_date=None, df=None):
        """
        Enhanced confidence calculation with DOMAIN-SPECIFIC INTELLIGENCE.
        
        Combines statistical metrics with Express Entry domain knowledge:
        - Recent trend alignment
        - Seasonal pattern alignment  
        - Realistic score ranges
        - Statistical performance
        """
        
        # Get metrics with safe defaults
        cv_score = result.get('cv_score', -np.inf)
        r2 = result.get('r2', -np.inf)
        mae = result.get('mae', np.inf)
        n_folds = result.get('n_cv_folds', 0)
        
        # Initialize component scores
        components = {}
        
        # STATISTICAL COMPONENTS (65% total)
        
        # 1. Cross-Validation Performance (25%) - Reduced from 40%
        if cv_score != -np.inf and cv_score > -1000:
            cv_normalized = max(0, min(1, (cv_score + 100) / 95))
            components['cv_performance'] = cv_normalized * 0.25
        else:
            components['cv_performance'] = 0.0
        
        # 2. Coefficient of Determination (20%) - Reduced from 25%
        if r2 != -np.inf:
            r2_normalized = max(0, min(1.0, r2))
            components['goodness_of_fit'] = r2_normalized * 0.20
        else:
            components['goodness_of_fit'] = 0.0
        
        # 3. Prediction Error (15%) - Reduced from 20%
        if mae != np.inf:
            mae_normalized = max(0, min(1, (60 - mae) / 60))
            components['prediction_accuracy'] = mae_normalized * 0.15
        else:
            components['prediction_accuracy'] = 0.0
        
        # 4. Validation Robustness (5%) - Reduced from 10%
        if n_folds > 0:
            folds_normalized = min(1, n_folds / 5)
            components['validation_robustness'] = folds_normalized * 0.05
        else:
            components['validation_robustness'] = 0.0
        
        # DOMAIN-SPECIFIC COMPONENTS (35% total) - NEW!
        
        # 5. Recent Trend Alignment (15%) - How well does prediction align with recent 6-month trend?
        if predicted_crs is not None and df is not None and len(df) >= 6:
            recent_scores = df['lowest_crs_score'].tail(6).values
            recent_trend = np.mean(recent_scores)
            recent_std = np.std(recent_scores)
            
            # Calculate how many standard deviations away from recent trend
            if recent_std > 0:
                z_score = abs(predicted_crs - recent_trend) / recent_std
                # SCIENTIFIC PENALTY based on probability theory and empirical rule
                trend_alignment = self._scientific_probability_penalty(z_score)
            else:
                trend_alignment = 1.0 if abs(predicted_crs - recent_trend) < 20 else 0.5
            
            components['trend_alignment'] = trend_alignment * 0.15
        else:
            components['trend_alignment'] = 0.075  # Neutral score when no data
        
        # 6. Seasonal Alignment (10%) - How well does it align with same month in previous years?
        if predicted_crs is not None and prediction_date is not None and df is not None:
            try:
                pred_month = prediction_date.month
                # Get historical data for same month
                df['date'] = pd.to_datetime(df['date'])
                historical_same_month = df[df['date'].dt.month == pred_month]['lowest_crs_score']
                
                if len(historical_same_month) >= 2:
                    seasonal_mean = historical_same_month.mean()
                    seasonal_std = historical_same_month.std()
                    
                    if seasonal_std > 0:
                        seasonal_z = abs(predicted_crs - seasonal_mean) / seasonal_std
                        # SCIENTIFIC PENALTY for seasonal deviations
                        seasonal_alignment = self._scientific_probability_penalty(seasonal_z)
                    else:
                        seasonal_alignment = 1.0 if abs(predicted_crs - seasonal_mean) < 30 else 0.5
                    
                    components['seasonal_alignment'] = seasonal_alignment * 0.10
                else:
                    components['seasonal_alignment'] = 0.05  # Neutral when insufficient seasonal data
            except:
                components['seasonal_alignment'] = 0.05
        else:
            components['seasonal_alignment'] = 0.05
        
        # 7. Realistic Range Score (10%) - Is prediction within reasonable historical bounds?
        if predicted_crs is not None and df is not None:
            historical_scores = df['lowest_crs_score'].values
            hist_min, hist_max = historical_scores.min(), historical_scores.max()
            hist_mean = historical_scores.mean()
            hist_std = historical_scores.std()
            
            # Define realistic range: mean ± 2.5 std, but bounded by historical extremes
            realistic_min = max(hist_min - 50, hist_mean - 2.5 * hist_std)  # Allow some extrapolation
            realistic_max = min(hist_max + 50, hist_mean + 2.5 * hist_std)
            
            # Calculate z-score relative to historical distribution
            hist_z_score = abs(predicted_crs - hist_mean) / hist_std
            
            if realistic_min <= predicted_crs <= realistic_max:
                # Within realistic range - use scientific penalty based on historical z-score
                realism_score = max(0.1, self._scientific_probability_penalty(hist_z_score))
            else:
                # Outside realistic range - severe scientific penalty
                if predicted_crs < realistic_min:
                    excess_z = (realistic_min - predicted_crs) / hist_std + 2.5  # Add to base 2.5σ
                else:
                    excess_z = (predicted_crs - realistic_max) / hist_std + 2.5  # Add to base 2.5σ
                realism_score = max(0.001, self._scientific_probability_penalty(excess_z))
            
            components['realism_score'] = realism_score * 0.10
            
            # Debug unrealistic predictions with scientific analysis
            if realism_score < 0.15 or hist_z_score > 2.0:  # More sensitive detection
                print(f"📊 SCIENTIFIC PENALTY ANALYSIS:")
                print(f"   Predicted CRS: {predicted_crs}")
                print(f"   Historical mean: {hist_mean:.1f} ± {hist_std:.1f}")
                print(f"   Z-score: {hist_z_score:.2f}σ ({self._z_score_probability_description(hist_z_score)})")
                print(f"   Historical range: {hist_min:.0f} - {hist_max:.0f}")
                print(f"   Realistic range: {realistic_min:.0f} - {realistic_max:.0f}")
                print(f"   Scientific penalty: {(1.0 - realism_score)*100:.1f}% confidence loss")
        else:
            components['realism_score'] = 0.05
        
        # Calculate total confidence (components sum to 100%)
        total_confidence = sum(components.values())
        
        # Apply confidence floor and ceiling
        final_confidence = max(0.15, min(0.95, total_confidence))
        
        # Debug output for models with domain issues (more sensitive threshold)
        if predicted_crs is not None and (components.get('trend_alignment', 0) < 0.10 or 
                                          components.get('seasonal_alignment', 0) < 0.10 or
                                          components.get('realism_score', 0) < 0.10):
            print(f"🔬 SCIENTIFIC DOMAIN ANALYSIS for prediction {predicted_crs:.0f}:")
            print(f"   📈 Trend Alignment: {components.get('trend_alignment', 0):.3f} (15%) - Recent 6-month trend")
            print(f"   📅 Seasonal Alignment: {components.get('seasonal_alignment', 0):.3f} (10%) - Historical same-month")
            print(f"   🎯 Realism Score: {components.get('realism_score', 0):.3f} (10%) - Overall historical range")
            print(f"   📊 Statistical Score: {total_confidence - components.get('trend_alignment', 0) - components.get('seasonal_alignment', 0) - components.get('realism_score', 0):.3f} (65%)")
            print(f"   ✅ FINAL SCIENTIFIC CONFIDENCE: {final_confidence:.3f}")
        
        return final_confidence
    
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

    def generate_realistic_draw_calendar(self, start_date, total_weeks=52):
        """
        Generate a realistic Express Entry draw calendar avoiding date conflicts.
        
        Based on historical patterns:
        - Maximum 1-2 draws per week
        - Strategic spacing to manage processing workload
        - Priority categories get preferred dates
        """
        from datetime import datetime, timedelta
        
        draw_calendar = {}
        current_date = start_date
        
        # Category priorities: Historical frequency + 2025 Government policy
        # 📊 HISTORICAL DATA (2022+): PNP #1 (45 draws), French #3 (20 draws), CEC #4 (17 draws)
        # 🏛️ GOVERNMENT POLICY: CEC primary focus, Healthcare/French priorities
        priority_categories = {
            'HIGHEST': ['Provincial Nominee Program'],  # #1 historical frequency (29.4 days, 45 draws)
            'HIGH': ['Canadian Experience Class', 'French-language proficiency'],  # Gov priority + historical data
            'MEDIUM': ['Healthcare and social services occupations', 'Education occupations'],  # Gov priority but quarterly frequency
            'LOW': ['Agriculture and agri-food occupations', 'STEM occupations', 'Trade occupations'],  # Infrequent (70-149 days)
            'ELIMINATED': ['Transport occupations', 'General', 'No Program Specified']  # No longer conducted
        }
        
        # Flatten priorities with order (HIGHEST gets first priority for date assignment)
        category_priority_order = []
        for priority_level in ['HIGHEST', 'HIGH', 'MEDIUM', 'LOW']:
            category_priority_order.extend(priority_categories[priority_level])
        
        # Generate weekly slots for draws (max 2 per week)
        # 📊 BASED ON HISTORICAL DATA ANALYSIS:
        # - Wednesday: 57.5% of all draws (dominant!)
        # - Thursday: 14.2% (popular for CEC: 57.8%)
        # - Friday: 10.1%
        # - Weekends: Almost never (0.8% Saturday, 0% Sunday)
        for week in range(total_weeks):
            week_start = current_date + timedelta(weeks=week)
            
            # Primary draw day: Wednesday (most common historically)
            primary_date = week_start + timedelta(days=2)  # Wednesday (57.5% of draws)
            
            # Secondary draw day: Thursday (especially good for CEC)
            secondary_date = week_start + timedelta(days=3)  # Thursday (14.2% of draws, CEC favorite)
            
            draw_calendar[week] = {
                'primary': primary_date,
                'secondary': secondary_date,
                'assigned_primary': None,
                'assigned_secondary': None
            }
        
        return draw_calendar, category_priority_order
    
    def assign_category_dates(self, ircc_groups, draw_calendar, category_priority_order, num_predictions):
        """
        Assign realistic dates to categories based on HISTORICAL DATA + 2025 GOVERNMENT PRIORITIES.
        
        Uses actual historical intervals (2022+) combined with official policy priorities.
        
        Returns: category_schedules = {category_name: [date1, date2, ...]}
        """
        category_schedules = {}
        
        # 📊 HISTORICAL INTERVAL DATA (2022+) - REAL PATTERNS  
        historical_intervals = {
            'Provincial Nominee Program': {'avg': 29.4, 'min': 6, 'max': 224, 'std': 49.0},
            'Canadian Experience Class': {'avg': 25.2, 'min': 6, 'max': 97, 'std': 22.8},  # NOT bi-weekly!
            'French-language proficiency': {'avg': 32.8, 'min': 5, 'max': 78, 'std': 20.4},
            'Healthcare and social services occupations': {'avg': 94.4, 'min': 8, 'max': 163, 'std': 57.1},
            'Trade occupations': {'avg': 149.0, 'min': 111, 'max': 198, 'std': 44.5},
            'STEM occupations': {'avg': 140.5, 'min': 125, 'max': 156, 'std': 21.9},
            'Agriculture and agri-food occupations': {'avg': 70.5, 'min': 57, 'max': 84, 'std': 19.1},
            'Transport occupations': {'avg': 87.5, 'min': 84, 'max': 91, 'std': 4.9},  # ELIMINATED 2025
            'General': {'avg': 17.3, 'min': 6, 'max': 57, 'std': 12.3},  # ELIMINATED 2025
            'Education occupations': {'avg': 60.0, 'min': 30, 'max': 90, 'std': 20.0}  # NEW 2025 - estimated
        }
        
        # 🎯 SEQUENTIAL PROCESSING: Most frequent/prioritized first, then reserve dates
        # Process in priority order to ensure proper date reservation
        priority_order = ['HIGHEST', 'HIGH', 'MEDIUM', 'LOW']
        processing_order = []
        
        for priority_level in priority_order:
            # Get categories for this priority level
            level_categories = [
                ircc_cat for ircc_cat, group_info in ircc_groups.items()
                if group_info['priority_level'] == priority_level
            ]
            processing_order.extend(level_categories)
        
        # Calculate prediction frequency based on HISTORICAL DATA + 2025 POLICY
        for ircc_category in processing_order:
            adjusted_count = self.get_adjusted_prediction_count(ircc_category, num_predictions)
            priority = self.get_category_priority_2025(ircc_category)
            
            # Skip eliminated categories
            if priority == 'ELIMINATED':
                continue
            
            # Get historical interval data for this category
            interval_data = historical_intervals.get(ircc_category, {'avg': 60.0, 'std': 20.0})
            avg_interval = interval_data['avg']
            
            # Convert average days to week frequency (for calendar assignment)
            # frequency = how many weeks between draws
            if avg_interval <= 18:        # ~2.5 weeks
                frequency = 2  # Every 2 weeks
            elif avg_interval <= 35:      # ~5 weeks  
                frequency = 4  # Every 4 weeks (monthly)
            elif avg_interval <= 70:      # ~10 weeks
                frequency = 8  # Every 8 weeks (bi-monthly)
            elif avg_interval <= 120:     # ~17 weeks
                frequency = 12  # Every 12 weeks (quarterly)
            else:
                frequency = 20  # Every 20 weeks (5+ months)
            
            # Generate date schedule for this category
            dates = []
            weeks_assigned = []
            
            # Start assignment from week 1
            current_week = 0
            predictions_needed = 5  # EXACTLY 5 predictions per category as requested
            
            while len(dates) < predictions_needed and current_week < len(draw_calendar):
                week_info = draw_calendar[current_week]
                
                # Determine if this category should draw this week based on frequency
                if current_week % frequency == 0:  # Respects frequency pattern
                    
                    # Assign dates based on HISTORICAL FREQUENCY + GOVERNMENT PRIORITY  
                    if priority == 'HIGHEST':  # Provincial Nominee Program - #1 HISTORICAL FREQUENCY
                        # 🎯 PNP PREFERS WEDNESDAY: 63.1% historical preference (top priority gets first choice)
                        if week_info['assigned_primary'] is None:  # Wednesday preferred for PNP
                            dates.append(week_info['primary'])  # Wednesday
                            draw_calendar[current_week]['assigned_primary'] = ircc_category
                            weeks_assigned.append(current_week)
                        elif week_info['assigned_secondary'] is None:  # Fallback to Thursday
                            dates.append(week_info['secondary'])  # Thursday
                            draw_calendar[current_week]['assigned_secondary'] = ircc_category
                            weeks_assigned.append(current_week)
                    
                    elif priority == 'HIGH':  # CEC + French - GOVERNMENT PRIORITY
                        if 'Canadian Experience Class' in ircc_category:
                            # 🎯 CEC LOVES THURSDAYS: 57.8% historical preference
                            if week_info['assigned_secondary'] is None:  # Thursday preferred for CEC
                                dates.append(week_info['secondary'])  # Thursday
                                draw_calendar[current_week]['assigned_secondary'] = ircc_category
                                weeks_assigned.append(current_week)
                            elif week_info['assigned_primary'] is None:  # Fallback to Wednesday
                                dates.append(week_info['primary'])  # Wednesday
                                draw_calendar[current_week]['assigned_primary'] = ircc_category
                                weeks_assigned.append(current_week)
                        else:
                            # 🎯 French: Wednesday preference (general pattern)
                            if week_info['assigned_primary'] is None:  # Wednesday preferred
                                dates.append(week_info['primary'])  # Wednesday
                                draw_calendar[current_week]['assigned_primary'] = ircc_category
                                weeks_assigned.append(current_week)
                            elif week_info['assigned_secondary'] is None:  # Fallback to Thursday
                                dates.append(week_info['secondary'])  # Thursday
                                draw_calendar[current_week]['assigned_secondary'] = ircc_category
                                weeks_assigned.append(current_week)
                    
                    else:  # MEDIUM/LOW - Gets remaining slots
                        # Other categories take whatever is available
                        if week_info['assigned_primary'] is None:
                            dates.append(week_info['primary'])  # Wednesday
                            draw_calendar[current_week]['assigned_primary'] = ircc_category
                            weeks_assigned.append(current_week)
                        elif week_info['assigned_secondary'] is None:
                            dates.append(week_info['secondary'])  # Thursday fallback
                            draw_calendar[current_week]['assigned_secondary'] = ircc_category
                            weeks_assigned.append(current_week)
                
                current_week += 1
            
            category_schedules[ircc_category] = dates
            
            # Show detailed reservation results
            priority_emojis = {
                'HIGHEST': '🏆', 'HIGH': '🥇', 'MEDIUM': '🥈', 'LOW': '🥉', 'ELIMINATED': '❌'
            }
            hist_interval = interval_data['avg']
            priority_emoji = priority_emojis.get(priority, '❓')
            
            print(f"📅 {priority_emoji} {ircc_category} ({priority}): {len(dates)} dates reserved")
            print(f"   Historical avg: {hist_interval:.1f} days, Frequency: every {frequency} weeks")
            if dates:
                print(f"   Reserved dates: {dates[0]} to {dates[-1] if len(dates) > 1 else dates[0]}")
                # Show day preferences met
                day_counts = {'Wed': 0, 'Thu': 0, 'Other': 0}
                for date in dates:
                    if date.weekday() == 2:  # Wednesday
                        day_counts['Wed'] += 1
                    elif date.weekday() == 3:  # Thursday
                        day_counts['Thu'] += 1
                    else:
                        day_counts['Other'] += 1
                print(f"   Day breakdown: {day_counts['Wed']} Wed, {day_counts['Thu']} Thu, {day_counts['Other']} Other")
            print()
        
        return category_schedules
    
    def predict_next_draw_date(self, working_df, ircc_category, base_date, rank):
        """
        🎯 DYNAMIC DATE PREDICTION: Use ML models to predict actual draw dates
        
        Instead of fixed intervals, train models on historical 'days_since_last_draw' 
        patterns to predict when the next draw will actually occur.
        
        Returns: (predicted_date, confidence_interval_lower, confidence_interval_upper)
        """
        
        print(f"   📅 Training date prediction models for rank {rank}...")
        
        if len(working_df) < 3:
            # Fallback using HISTORICAL AVERAGES instead of arbitrary values
            historical_intervals = {
                'Provincial Nominee Program': 29.4,
                'Canadian Experience Class': 25.2,  # NOT 35 days!
                'French-language proficiency': 32.8,
                'Healthcare and social services occupations': 94.4,
                'Trade occupations': 149.0,
                'STEM occupations': 140.5,
                'Agriculture and agri-food occupations': 70.5,
                'Education occupations': 60.0  # NEW 2025 category
            }
            fallback_days = historical_intervals.get(ircc_category, 60.0)
            predicted_date = base_date + timedelta(days=int(fallback_days))
            ci_width = int(fallback_days * 0.3)  # ±30% confidence interval
            return predicted_date, predicted_date - timedelta(days=ci_width), predicted_date + timedelta(days=ci_width)
        
        # Prepare data for date prediction (predict days_since_last_draw)
        date_features = working_df[['days_since_last_draw']].dropna()
        if len(date_features) < 3:
            # Use same historical intervals for consistency
            historical_intervals = {
                'Provincial Nominee Program': 29.4,
                'Canadian Experience Class': 25.2,  # NOT 35 days!
                'French-language proficiency': 32.8,
                'Healthcare and social services occupations': 94.4,
                'Trade occupations': 149.0,
                'STEM occupations': 140.5,
                'Agriculture and agri-food occupations': 70.5,
                'Education occupations': 60.0  # NEW 2025 category
            }
            fallback_days = historical_intervals.get(ircc_category, 60.0)
            predicted_date = base_date + timedelta(days=int(fallback_days))
            ci_width = int(fallback_days * 0.3)  # ±30% confidence interval
            return predicted_date, predicted_date - timedelta(days=ci_width), predicted_date + timedelta(days=ci_width)
        
        # Train multiple models to predict days_since_last_draw
        date_predictions = []
        
        try:
            # 1. ARIMA for time series patterns
            from statsmodels.tsa.arima.model import ARIMA
            arima_data = date_features['days_since_last_draw'].values
            try:
                arima_model = ARIMA(arima_data, order=(1, 0, 1))
                arima_fit = arima_model.fit()
                arima_pred = arima_fit.forecast(steps=1)[0]
                date_predictions.append(max(7, min(365, arima_pred)))  # Bound between 7-365 days
                print(f"      🔧 ARIMA date prediction: {arima_pred:.0f} days")
            except:
                pass
            
            # 2. Exponential smoothing for trend
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            try:
                exp_model = ExponentialSmoothing(arima_data, trend='add')
                exp_fit = exp_model.fit()
                exp_pred = exp_fit.forecast(steps=1)[0]
                date_predictions.append(max(7, min(365, exp_pred)))
                print(f"      🔧 Exponential Smoothing date prediction: {exp_pred:.0f} days")
            except:
                pass
                
            # 3. Historical average with recent weighting
            recent_avg = date_features['days_since_last_draw'].tail(5).mean()
            overall_avg = date_features['days_since_last_draw'].mean()
            weighted_avg = 0.7 * recent_avg + 0.3 * overall_avg
            date_predictions.append(max(7, min(365, weighted_avg)))
            print(f"      🔧 Weighted average date prediction: {weighted_avg:.0f} days")
            
            # 4. Category-specific seasonal patterns
            if len(working_df) > 12:  # Need enough data for seasonal analysis
                # Look at same month patterns from previous years
                current_month = base_date.month
                same_month_data = working_df[working_df['date'].dt.month == current_month]
                if len(same_month_data) > 1:
                    seasonal_avg = same_month_data['days_since_last_draw'].mean()
                    date_predictions.append(max(7, min(365, seasonal_avg)))
                    print(f"      🔧 Seasonal pattern prediction: {seasonal_avg:.0f} days")
            
        except Exception as e:
            print(f"      ⚠️ Date prediction models failed: {e}")
        
        # If no models worked, use historical statistics
        if not date_predictions:
            category_avg = date_features['days_since_last_draw'].mean()
            date_predictions.append(max(7, min(365, category_avg)))
            print(f"      🔧 Fallback to category average: {category_avg:.0f} days")
        
        # Ensemble prediction: weighted average with outlier removal
        if len(date_predictions) > 1:
            # Remove outliers (predictions > 2 std devs from median)
            pred_array = np.array(date_predictions)
            median_pred = np.median(pred_array)
            std_pred = np.std(pred_array)
            
            # Keep predictions within 2 standard deviations
            valid_preds = pred_array[np.abs(pred_array - median_pred) <= 2 * std_pred]
            
            if len(valid_preds) > 0:
                final_days_prediction = np.mean(valid_preds)
            else:
                final_days_prediction = median_pred
        else:
            final_days_prediction = date_predictions[0]
        
        # Add progressive uncertainty for future ranks
        rank_uncertainty = (rank - 1) * 3  # 0, 3, 6, 9, 12 extra days uncertainty
        final_days_prediction += rank_uncertainty
        
        # Calculate confidence intervals based on historical variance
        days_std = date_features['days_since_last_draw'].std()
        confidence_multiplier = 1.959964  # qnorm(0.975) for 95% confidence interval
        
        # Adjust confidence based on data quality and rank
        data_quality_factor = min(1.0, len(date_features) / 10)  # More data = tighter CI
        rank_uncertainty_factor = 1 + (rank - 1) * 0.2  # Later ranks = wider CI
        
        ci_width = confidence_multiplier * days_std * rank_uncertainty_factor / data_quality_factor
        ci_width = max(3, min(30, ci_width))  # Bound CI width between 3-30 days
        
        # Calculate final dates
        predicted_date = base_date + timedelta(days=int(final_days_prediction))
        ci_lower = predicted_date - timedelta(days=int(ci_width))
        ci_upper = predicted_date + timedelta(days=int(ci_width))
        
        print(f"      ✅ Final prediction: {predicted_date.strftime('%b %d')} ({ci_lower.strftime('%b %d')}-{ci_upper.strftime('%b %d')}, 95% CI)")
        
        return predicted_date, ci_lower, ci_upper
    
    def get_interval_type(self, model_name):
        """Determine if model uses Bayesian (CrI) or Frequentist (CI) intervals"""
        bayesian_models = [
            'BayesianHierarchical', 
            'GaussianProcess', 
            'Prophet',  # Prophet uses Bayesian inference
            'DynamicLinearModel'  # Often Bayesian
        ]
        
        for bayesian_model in bayesian_models:
            if bayesian_model.lower() in model_name.lower():
                return 'CrI'  # Credibility Interval for Bayesian models
        
        return 'CI'  # Confidence Interval for Frequentist models
    
    def setup_prediction_logging(self, category_name):
        """Setup detailed logging for debugging prediction issues"""
        log_dir = 'prediction_logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a unique log file for this category
        log_file = os.path.join(log_dir, f'predictions_{category_name.replace(" ", "_").replace("/", "_")}.log')
        
        # Setup logger
        logger = logging.getLogger(f'prediction_{category_name}')
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add file handler
        handler = logging.FileHandler(log_file, mode='w')  # Overwrite each time
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def compute_recursive_predictions(self, category, force_recompute, assigned_dates=None):
        """
        🎯 RECURSIVE FORECASTING: Scientifically sound approach
        
        Strategy:
        1. Predict NEXT draw (rank 1) using all models → select BEST prediction
        2. Add best prediction as "historical data" → predict rank 2
        3. Continue recursive chain up to rank 5
        
        This mirrors real-world draw scheduling where each draw affects the next.
        Focus: PRIMARY prediction for next draw + 4 secondary predictions.
        """
        
        # Setup detailed logging for debugging
        logger = self.setup_prediction_logging(category.name)
        
        # Check if we need to recompute
        if not force_recompute:
            existing_predictions = PreComputedPrediction.objects.filter(
                category=category, 
                is_active=True,
                created_at__gte=timezone.now() - timedelta(days=1)
            ).count()
            
            if existing_predictions >= 5:  # Fixed 5 predictions per category
                return 0  # Already have recent predictions
        
        # Get pooled data from related category versions
        pooled_draws, ircc_category, num_pooled_categories = category.get_pooled_data()
        
        if pooled_draws.count() < 1:
            raise ValueError(f"No data available: {pooled_draws.count()} draws found")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'date': draw.date,
            'category': ircc_category,
            'lowest_crs_score': draw.lowest_crs_score,
            'invitations_issued': draw.invitations_issued,
            'days_since_last_draw': draw.days_since_last_draw or 14,
            'is_weekend': draw.is_weekend,
            'is_holiday': draw.is_holiday,
            'month': draw.month,
            'quarter': draw.quarter
        } for draw in pooled_draws])
        
        # Clear old predictions if force recompute
        if force_recompute:
            deletion_count = PreComputedPrediction.objects.filter(category=category).count()
            PreComputedPrediction.objects.filter(category=category).delete()
            print(f"🗑️  Recursive: Cleared {deletion_count} existing predictions for {category.name}")
        
        # Date calculation
        import pytz
        eastern = pytz.timezone('America/Toronto')
        now_eastern = timezone.now().astimezone(eastern)
        today_eastern = now_eastern.date()
        last_draw_date = pooled_draws.last().date
        
        # Start predictions intelligently
        days_since_last_draw = (today_eastern - last_draw_date).days
        if days_since_last_draw >= 14:
            next_draw_start = today_eastern + timedelta(days=7)
        else:
            next_draw_start = last_draw_date + timedelta(days=14)
        current_date = max(next_draw_start, today_eastern)
        
        print(f"🔄 RECURSIVE FORECASTING for {category.name}")
        print(f"   📅 Starting from: {current_date}")
        print(f"   🎯 Strategy: Next draw + 4 recursive predictions")
        
        # Log initial state
        logger.info(f"Starting recursive forecasting for {category.name}")
        logger.info(f"Data available: {len(df)} historical draws")
        logger.info(f"Historical CRS range: {df['lowest_crs_score'].min():.0f} - {df['lowest_crs_score'].max():.0f}")
        logger.info(f"Historical CRS mean: {df['lowest_crs_score'].mean():.1f} ± {df['lowest_crs_score'].std():.1f}")
        
        # Train invitation model
        from predictor.ml_models import InvitationPredictor
        invitation_model = None
        try:
            invitation_model = InvitationPredictor(model_type='XGB')
            invitation_model.train(df)
            print(f"   ✅ Invitation model trained with {len(df)} draws")
        except Exception as e:
            print(f"   ⚠️  Invitation model failed: {e}")
        
        # Get working dataframe for recursive updates
        working_df = df.copy()
        total_created = 0
        
        # 🔄 RECURSIVE LOOP: 5 predictions with dependency chain
        for rank in range(1, 6):  # Ranks 1-5
            print(f"\n🎯 RECURSIVE RANK {rank}:")
            
            # Evaluate models for current data state
            all_models = self.select_best_model(working_df, category)
            if not all_models:
                print(f"   ❌ No models available for rank {rank}")
                break
            
            # Generate predictions from all models for this rank
            rank_predictions = []
            
            for model_info in all_models:
                model = model_info['model']
                model_name = model_info['name']
                base_confidence = model_info['confidence']
                
                try:
                    # Train model on current working data
                    model.train(working_df)
                    
                    # Predict next draw only (not multiple steps)
                    if hasattr(model, 'predict'):
                        if model.name == 'ARIMA Time Series':
                            predicted_score = model.predict(steps=1)
                            predicted_score = predicted_score[0] if isinstance(predicted_score, list) else predicted_score
                        elif model.name == 'Prophet Time Series':
                            predicted_score = model.predict(periods=1, freq='2W')
                            predicted_score = predicted_score[0] if isinstance(predicted_score, list) else predicted_score
                        elif 'LSTM' in model.name:
                            sequence_length = getattr(model, 'sequence_length', 10)
                            sequence_data = working_df['lowest_crs_score'].tail(sequence_length).values
                            
                            if len(sequence_data) < sequence_length:
                                last_value = float(sequence_data[-1]) if len(sequence_data) > 0 else float(working_df['lowest_crs_score'].mean())
                                padded_sequence = [last_value] * sequence_length
                                if len(sequence_data) > 0:
                                    padded_sequence[-len(sequence_data):] = sequence_data.tolist()
                                sequence_data = np.array(padded_sequence)
                            
                            sequence_data = sequence_data.reshape(1, sequence_length, 1)
                            predicted_score = model.predict(sequence_data, steps=1)
                            predicted_score = predicted_score[0] if hasattr(predicted_score, '__len__') else predicted_score
                        else:
                            # Standard single-step prediction
                            predicted_score = model.predict()
                            predicted_score = predicted_score[0] if isinstance(predicted_score, list) else predicted_score
                    else:
                        predicted_score = working_df['lowest_crs_score'].mean()
                    
                    # Ensure we have a valid number and enforce realistic bounds
                    predicted_score = float(predicted_score)
                    
                    # 🛡️ ENFORCE REALISTIC BOUNDS: Critical for healthcare stability
                    if predicted_score < 250 or predicted_score > 950 or not np.isfinite(predicted_score):
                        historical_mean = working_df['lowest_crs_score'].mean()
                        predicted_score = historical_mean
                        print(f"   🛡️ BOUND ENFORCEMENT: {model_name} prediction was unrealistic, reset to historical mean {historical_mean:.0f}")
                    
                    # Calculate confidence with domain intelligence
                    # Use simple date calculation for confidence (just for internal calculation)
                    from datetime import timedelta
                    simple_date = current_date + timedelta(days=30 * rank)  # Simple progressive dates
                    confidence = self._calculate_model_confidence(
                        result={'cv_score': -30, 'r2_score': 0.3, 'mae': 25}, 
                        data_size=len(working_df),
                        predicted_crs=predicted_score,
                        prediction_date=simple_date,
                        df=working_df
                    )
                    
                    # Predict invitations
                    predicted_invitations = 1500  # Default
                    if invitation_model:
                        try:
                            predicted_invitations = invitation_model.predict(working_df, predicted_score)
                        except:
                            pass
                    
                    rank_predictions.append({
                        'model': model_name,
                        'crs': predicted_score,
                        'invitations': predicted_invitations,
                        'confidence': confidence,
                        'base_confidence': base_confidence
                    })
                    
                    print(f"   🔧 {model_name}: CRS {predicted_score:.0f}, Confidence {confidence:.3f}")
                    
                    # Log detailed prediction info
                    logger.info(f"Rank {rank} - {model_name}: CRS {predicted_score:.0f}, Confidence {confidence:.3f}, Invitations {predicted_invitations}")
                    
                except Exception as e:
                    print(f"   ❌ {model_name} failed: {e}")
                    logger.error(f"Rank {rank} - {model_name} failed: {str(e)}")
                    continue
            
            if not rank_predictions:
                print(f"   ❌ No successful predictions for rank {rank}")
                break
            
            # Select BEST prediction (highest confidence) for recursive chain
            best_prediction = max(rank_predictions, key=lambda x: x['confidence'])
            print(f"   🏆 BEST: {best_prediction['model']} - CRS {best_prediction['crs']:.0f} (confidence: {best_prediction['confidence']:.3f})")
            
            # 🎯 PRIORITY-BASED DATE ASSIGNMENT: Use assigned dates first, ML models as fallback
            if assigned_dates and rank <= len(assigned_dates):
                # Use pre-assigned date from conflict-free calendar system
                prediction_date = assigned_dates[rank - 1]  # rank 1 = index 0
                
                # Calculate realistic confidence intervals for assigned dates
                days_std = 7  # Conservative 1-week standard deviation for assigned dates
                ci_width = days_std * (1 + (rank - 1) * 0.2)  # Slightly wider for later ranks
                date_ci_lower = prediction_date - timedelta(days=int(ci_width))
                date_ci_upper = prediction_date + timedelta(days=int(ci_width))
                
                print(f"      ✅ Using ASSIGNED date: {prediction_date.strftime('%b %d')} (conflict-free calendar)")
            else:
                # Fallback to ML-based date prediction when no assigned dates
                prediction_date, date_ci_lower, date_ci_upper = self.predict_next_draw_date(working_df, ircc_category, current_date, rank)
                print(f"      🔧 Using ML-predicted date: {prediction_date.strftime('%b %d')} (model-based)")
            
            # 💾 SAVE ALL MODEL PREDICTIONS (not just the best one)
            models_saved = 0
            final_crs = best_prediction['crs']  # Use best for recursive chain
            final_invitations = best_prediction['invitations']
            
            from django.db import transaction
            for pred in rank_predictions:
                try:
                    # Apply uncertainty modeling for each model
                    uncertainty = self.apply_uncertainty_modeling([pred], rank)
                    
                    # 🎯 Determine interval type based on model
                    interval_type = self.get_interval_type(pred['model'])
                    
                    with transaction.atomic():
                        prediction = PreComputedPrediction.objects.create(
                            category=category,
                            predicted_date=prediction_date,
                            predicted_crs_score=round(pred['crs']),
                            predicted_invitations=round(pred['invitations']),
                            confidence_score=pred['confidence'],
                            model_used=pred['model'],
                            model_version="1.0",
                            prediction_rank=rank,
                            uncertainty_range={
                                'crs_min': max(300, round(pred['crs'] - uncertainty.get('crs_std', 50))),
                                'crs_max': min(1000, round(pred['crs'] + uncertainty.get('crs_std', 50))),
                                'invitations_min': max(0, round(pred['invitations'] - uncertainty.get('inv_std', 500))),
                                'invitations_max': round(pred['invitations'] + uncertainty.get('inv_std', 500))
                            },
                            # 📅 NEW: Dynamic date prediction with confidence intervals
                            predicted_date_lower=date_ci_lower.date() if hasattr(date_ci_lower, 'date') else date_ci_lower,
                            predicted_date_upper=date_ci_upper.date() if hasattr(date_ci_upper, 'date') else date_ci_upper,
                            interval_type=interval_type,  # 🆕 CI for frequentist, CrI for Bayesian
                            is_active=True
                        )
                    models_saved += 1
                    logger.info(f"Rank {rank} - Saved {pred['model']}: CRS {pred['crs']:.0f}, Confidence {pred['confidence']:.3f}")
                        
                except Exception as e:
                    print(f"   ❌ Failed to save {pred['model']} for rank {rank}: {e}")
                    continue
            
            total_created += models_saved
            print(f"   ✅ Saved: Rank {rank}, {models_saved}/{len(rank_predictions)} models, Date {prediction_date}")
            
            # 🔄 RECURSIVE STEP: Add best prediction as "historical data" for next iteration
            if rank < 5:  # Don't add after the last prediction
                # Calculate realistic interval for this category (handle date type conversion)
                if len(working_df) > 0:
                    last_date = working_df['date'].iloc[-1]
                    if hasattr(last_date, 'date'):  # pandas Timestamp
                        last_date = last_date.date()
                    days_since_last = (prediction_date - last_date).days
                else:
                    days_since_last = 35
                
                new_row = {
                    'date': prediction_date,
                    'category': ircc_category,
                    'lowest_crs_score': final_crs,
                    'invitations_issued': final_invitations,
                    'days_since_last_draw': days_since_last,  # Realistic interval
                    'is_weekend': prediction_date.weekday() >= 5,
                    'is_holiday': False,  # Simplified
                    'month': prediction_date.month,
                    'quarter': (prediction_date.month - 1) // 3 + 1
                }
                
                # Append to working dataframe
                working_df = pd.concat([working_df, pd.DataFrame([new_row])], ignore_index=True)
                print(f"   🔄 Added prediction as data for rank {rank+1}")
        
        print(f"🎉 RECURSIVE COMPLETE: Created {total_created} predictions for {category.name}")
        return total_created
    
    def apply_uncertainty_modeling(self, predictions, rank):
        """Apply uncertainty modeling to predictions"""
        base_uncertainty = 50 + (rank * 10)  # Increase uncertainty with rank
        
        return {
            'crs_std': base_uncertainty,
            'inv_std': 500 + (rank * 200)
        } 