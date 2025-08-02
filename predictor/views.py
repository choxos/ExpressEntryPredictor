from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.utils import timezone
from django.db.models import Avg, Count, Q
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from datetime import datetime, date, timedelta
import pandas as pd

from .models import (
    DrawCategory, ExpressEntryDraw, EconomicIndicator,
    PredictionModel, DrawPrediction, PredictionAccuracy
)
from .serializers import (
    DrawCategorySerializer, ExpressEntryDrawSerializer, EconomicIndicatorSerializer,
    PredictionModelSerializer, DrawPredictionSerializer, PredictionAccuracySerializer,
    DrawStatsSerializer, PredictionResultSerializer, ModelTrainingResultSerializer
)
from .ml_models import (
    ARIMAPredictor, RandomForestPredictor, XGBoostPredictor,
    LinearRegressionPredictor, EnsemblePredictor
)


class DrawCategoryViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Draw Categories"""
    queryset = DrawCategory.objects.filter(is_active=True)
    serializer_class = DrawCategorySerializer
    
    @action(detail=True, methods=['get'])
    def draws(self, request, pk=None):
        """Get all draws for a specific category"""
        category = self.get_object()
        draws = ExpressEntryDraw.objects.filter(category=category).order_by('-date')
        
        # Pagination
        page = int(request.query_params.get('page', 1))
        page_size = int(request.query_params.get('page_size', 20))
        start = (page - 1) * page_size
        end = start + page_size
        
        paginated_draws = draws[start:end]
        serializer = ExpressEntryDrawSerializer(paginated_draws, many=True)
        
        return Response({
            'count': draws.count(),
            'results': serializer.data,
            'page': page,
            'page_size': page_size
        })
    
    @action(detail=True, methods=['get'])
    def stats(self, request, pk=None):
        """Get statistics for a specific category"""
        category = self.get_object()
        draws = ExpressEntryDraw.objects.filter(category=category)
        
        if not draws.exists():
            return Response({'message': 'No draws found for this category'})
        
        stats = {
            'total_draws': draws.count(),
            'avg_crs_score': draws.aggregate(avg=Avg('lowest_crs_score'))['avg'],
            'avg_invitations': draws.aggregate(avg=Avg('invitations_issued'))['avg'],
            'min_crs_score': draws.order_by('lowest_crs_score').first().lowest_crs_score,
            'max_crs_score': draws.order_by('-lowest_crs_score').first().lowest_crs_score,
            'latest_draw': ExpressEntryDrawSerializer(draws.order_by('-date').first()).data,
            'earliest_draw': ExpressEntryDrawSerializer(draws.order_by('date').first()).data,
        }
        
        return Response(stats)


class ExpressEntryDrawViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Express Entry Draws"""
    queryset = ExpressEntryDraw.objects.all().order_by('-date')
    serializer_class = ExpressEntryDrawSerializer
    filterset_fields = ['category', 'month', 'quarter', 'is_weekend', 'is_holiday']
    search_fields = ['round_number', 'category__name']
    ordering_fields = ['date', 'lowest_crs_score', 'invitations_issued']
    
    @action(detail=False, methods=['get'])
    def recent(self, request):
        """Get recent draws (last 30 days)"""
        thirty_days_ago = timezone.now().date() - timedelta(days=30)
        recent_draws = self.queryset.filter(date__gte=thirty_days_ago)
        serializer = self.get_serializer(recent_draws, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def trends(self, request):
        """Get draw trends over time"""
        # Get draws from last 2 years
        two_years_ago = timezone.now().date() - timedelta(days=730)
        draws = self.queryset.filter(date__gte=two_years_ago)
        
        # Group by month
        monthly_data = {}
        for draw in draws:
            month_key = f"{draw.date.year}-{draw.date.month:02d}"
            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    'month': month_key,
                    'draws': [],
                    'avg_crs': 0,
                    'total_invitations': 0,
                    'categories': set()
                }
            
            monthly_data[month_key]['draws'].append(draw)
            monthly_data[month_key]['total_invitations'] += draw.invitations_issued
            monthly_data[month_key]['categories'].add(draw.category.name)
        
        # Calculate averages
        for month_data in monthly_data.values():
            draws_in_month = month_data['draws']
            month_data['avg_crs'] = sum(d.lowest_crs_score for d in draws_in_month) / len(draws_in_month)
            month_data['draw_count'] = len(draws_in_month)
            month_data['categories'] = list(month_data['categories'])
            del month_data['draws']  # Remove draws to reduce payload size
        
        return Response(sorted(monthly_data.values(), key=lambda x: x['month']))


class PredictionModelViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Prediction Models"""
    queryset = PredictionModel.objects.filter(is_active=True).order_by('-created_at')
    serializer_class = PredictionModelSerializer
    
    @action(detail=True, methods=['post'])
    def train(self, request, pk=None):
        """Train a specific model"""
        model_obj = self.get_object()
        
        try:
            # Get training data
            draws = ExpressEntryDraw.objects.all().order_by('date')
            if not draws.exists():
                return Response(
                    {'error': 'No training data available'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'date': draw.date,
                'category': draw.category.name,
                'lowest_crs_score': draw.lowest_crs_score,
                'invitations_issued': draw.invitations_issued,
                'days_since_last_draw': draw.days_since_last_draw or 14,
                'is_weekend': draw.is_weekend,
                'is_holiday': draw.is_holiday,
                'month': draw.month,
                'quarter': draw.quarter
            } for draw in draws])
            
            # Initialize model based on type
            start_time = timezone.now()
            
            if model_obj.model_type == 'ARIMA':
                predictor = ARIMAPredictor()
            elif model_obj.model_type == 'RF':
                predictor = RandomForestPredictor()
            elif model_obj.model_type == 'XGB':
                predictor = XGBoostPredictor()
            elif model_obj.model_type == 'LR':
                predictor = LinearRegressionPredictor()
            else:
                return Response(
                    {'error': f'Unknown model type: {model_obj.model_type}'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Train model
            metrics = predictor.train(df)
            training_time = (timezone.now() - start_time).total_seconds()
            
            # Update model object with results
            model_obj.mae_score = metrics.get('mae')
            model_obj.mse_score = metrics.get('mse')
            model_obj.r2_score = metrics.get('r2')
            model_obj.feature_importance = getattr(predictor, 'feature_importance', {})
            model_obj.trained_on = timezone.now()
            model_obj.save()
            
            result = ModelTrainingResultSerializer({
                'model_name': model_obj.name,
                'model_type': model_obj.model_type,
                'training_metrics': metrics,
                'feature_importance': model_obj.feature_importance,
                'training_time': training_time,
                'success': True
            })
            
            return Response(result.data)
            
        except Exception as e:
            result = ModelTrainingResultSerializer({
                'model_name': model_obj.name,
                'model_type': model_obj.model_type,
                'training_metrics': {},
                'feature_importance': {},
                'training_time': 0,
                'success': False,
                'error_message': str(e)
            })
            
            return Response(result.data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DrawPredictionViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Draw Predictions"""
    queryset = DrawPrediction.objects.filter(is_published=True).order_by('predicted_date')
    serializer_class = DrawPredictionSerializer
    filterset_fields = ['category', 'model']
    
    @action(detail=False, methods=['get'])
    def upcoming(self, request):
        """Get upcoming predictions"""
        today = date.today()
        upcoming = self.queryset.filter(predicted_date__gte=today)
        serializer = self.get_serializer(upcoming, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def by_category(self, request):
        """Get predictions grouped by category"""
        today = date.today()
        end_of_year = date(today.year, 12, 31)
        
        predictions_by_category = {}
        
        for category in DrawCategory.objects.filter(is_active=True):
            predictions = self.queryset.filter(
                category=category,
                predicted_date__gte=today,
                predicted_date__lte=end_of_year
            )
            
            if predictions.exists():
                predictions_by_category[category.name] = {
                    'category': DrawCategorySerializer(category).data,
                    'predictions': self.get_serializer(predictions, many=True).data
                }
        
        return Response(predictions_by_category)


class PredictionAPIView(APIView):
    """Main prediction API endpoint"""
    
    def get(self, request, category_id=None):
        """Generate predictions for a specific category or all categories"""
        
        if category_id:
            try:
                category = DrawCategory.objects.get(id=category_id, is_active=True)
                categories = [category]
            except DrawCategory.DoesNotExist:
                return Response(
                    {'error': 'Category not found'}, 
                    status=status.HTTP_404_NOT_FOUND
                )
        else:
            categories = DrawCategory.objects.filter(is_active=True)
        
        results = []
        
        for category in categories:
            try:
                # Get historical data for this category
                draws = ExpressEntryDraw.objects.filter(category=category).order_by('date')
                
                if draws.count() < 10:  # Need minimum data for predictions
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame([{
                    'date': draw.date,
                    'category': draw.category.name,
                    'lowest_crs_score': draw.lowest_crs_score,
                    'invitations_issued': draw.invitations_issued,
                    'days_since_last_draw': draw.days_since_last_draw or 14,
                    'is_weekend': draw.is_weekend,
                    'is_holiday': draw.is_holiday,
                    'month': draw.month,
                    'quarter': draw.quarter
                } for draw in draws])
                
                # Create ensemble of models
                ensemble = EnsemblePredictor()
                
                # Add models to ensemble
                try:
                    ensemble.add_model(ARIMAPredictor())
                except:
                    pass
                
                try:
                    ensemble.add_model(RandomForestPredictor())
                except:
                    pass
                
                try:
                    ensemble.add_model(XGBoostPredictor())
                except:
                    pass
                
                try:
                    ensemble.add_model(LinearRegressionPredictor())
                except:
                    pass
                
                if not ensemble.models:
                    continue
                
                # Train ensemble
                ensemble.train(df)
                
                # Generate predictions for next few draws
                predictions = []
                last_draw_date = draws.last().date
                
                # Predict next 5 draws until end of year
                current_date = last_draw_date
                year_end = date(current_date.year, 12, 31)
                
                for i in range(5):
                    if current_date >= year_end:
                        break
                    
                    # Estimate next draw date (typically 2 weeks apart)
                    next_date = current_date + timedelta(days=14)
                    
                    # Predict CRS score
                    try:
                        predicted_score = ensemble.predict(steps=1)[0]
                        predicted_score = max(300, min(900, int(predicted_score)))  # Reasonable bounds
                    except:
                        predicted_score = 450  # Fallback
                    
                    predictions.append({
                        'predicted_date': next_date,
                        'predicted_crs_score': predicted_score,
                        'confidence': 75,  # Default confidence
                        'model_used': 'Ensemble'
                    })
                    
                    current_date = next_date
                
                result_data = {
                    'category': DrawCategorySerializer(category).data,
                    'predictions': predictions,
                    'ensemble_prediction': {
                        'next_draw_date': predictions[0]['predicted_date'] if predictions else None,
                        'next_crs_score': predictions[0]['predicted_crs_score'] if predictions else None,
                        'confidence': predictions[0]['confidence'] if predictions else None
                    },
                    'model_performance': {
                        'r2_score': ensemble.metrics.get('r2', 0),
                        'mae': ensemble.metrics.get('mae', 0)
                    },
                    'confidence_metrics': {
                        'overall_confidence': 75,
                        'data_quality': 'Good' if draws.count() > 50 else 'Limited'
                    },
                    'prediction_timeline': predictions
                }
                
                results.append(PredictionResultSerializer(result_data).data)
                
            except Exception as e:
                print(f"Error predicting for category {category.name}: {e}")
                continue
        
        return Response(results)


class DashboardStatsAPIView(APIView):
    """API endpoint for dashboard statistics"""
    
    def get(self, request):
        """Get comprehensive dashboard statistics"""
        
        # Basic stats
        total_draws = ExpressEntryDraw.objects.count()
        categories_count = DrawCategory.objects.filter(is_active=True).count()
        
        # Date range
        earliest_draw = ExpressEntryDraw.objects.order_by('date').first()
        latest_draw = ExpressEntryDraw.objects.order_by('-date').first()
        
        date_range = {
            'start': earliest_draw.date if earliest_draw else None,
            'end': latest_draw.date if latest_draw else None
        }
        
        # Averages
        avg_stats = ExpressEntryDraw.objects.aggregate(
            avg_crs=Avg('lowest_crs_score'),
            avg_invitations=Avg('invitations_issued')
        )
        
        # Category breakdown
        category_breakdown = []
        for category in DrawCategory.objects.filter(is_active=True):
            draws = ExpressEntryDraw.objects.filter(category=category)
            if draws.exists():
                category_breakdown.append({
                    'category': category.name,
                    'count': draws.count(),
                    'avg_crs': draws.aggregate(avg=Avg('lowest_crs_score'))['avg'],
                    'latest_draw': draws.order_by('-date').first().date
                })
        
        # Monthly trends (last 12 months)
        twelve_months_ago = timezone.now().date() - timedelta(days=365)
        recent_draws = ExpressEntryDraw.objects.filter(date__gte=twelve_months_ago)
        
        monthly_trends = []
        current_date = twelve_months_ago
        end_date = timezone.now().date()
        
        while current_date <= end_date:
            month_draws = recent_draws.filter(
                date__year=current_date.year,
                date__month=current_date.month
            )
            
            if month_draws.exists():
                monthly_trends.append({
                    'month': f"{current_date.year}-{current_date.month:02d}",
                    'draw_count': month_draws.count(),
                    'avg_crs': month_draws.aggregate(avg=Avg('lowest_crs_score'))['avg'],
                    'total_invitations': sum(d.invitations_issued for d in month_draws)
                })
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        # Recent draws
        recent_draws_qs = ExpressEntryDraw.objects.order_by('-date')[:10]
        recent_draws_data = ExpressEntryDrawSerializer(recent_draws_qs, many=True).data
        
        stats_data = {
            'total_draws': total_draws,
            'categories_count': categories_count,
            'date_range': date_range,
            'avg_crs_score': avg_stats['avg_crs'],
            'avg_invitations': avg_stats['avg_invitations'],
            'category_breakdown': category_breakdown,
            'monthly_trends': monthly_trends,
            'recent_draws': recent_draws_data
        }
        
        serializer = DrawStatsSerializer(stats_data)
        return Response(serializer.data)


def home_view(request):
    """Home page view"""
    return render(request, 'predictor/home.html')


def dashboard_view(request):
    """Dashboard view"""
    return render(request, 'predictor/dashboard.html')


def analytics_view(request):
    """Analytics view"""
    return render(request, 'predictor/analytics.html')


def predictions_view(request):
    """Predictions view"""
    return render(request, 'predictor/predictions.html')
