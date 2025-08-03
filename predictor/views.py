import pandas as pd
from datetime import date, timedelta
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.db.models import Avg, Count, Max, Min, Q, Sum
from django.utils import timezone
from django.core.paginator import Paginator

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import (
    DrawCategory, ExpressEntryDraw, PredictionModel, 
    DrawPrediction, PredictionAccuracy, PreComputedPrediction, PredictionCache
)
from .serializers import (
    DrawCategorySerializer, ExpressEntryDrawSerializer, 
    PredictionModelSerializer, DrawPredictionSerializer,
    DashboardStatsSerializer, PredictionSummarySerializer
)


# =================== VIEWSETS ===================

class DrawCategoryViewSet(viewsets.ModelViewSet):
    """API for managing draw categories"""
    queryset = DrawCategory.objects.filter(is_active=True)
    serializer_class = DrawCategorySerializer

    @action(detail=True, methods=['get'])
    def statistics(self, request, pk=None):
        """Get statistics for a specific category"""
        category = self.get_object()
        draws = ExpressEntryDraw.objects.filter(category=category)
        
        stats = {
            'total_draws': draws.count(),
            'avg_crs_score': draws.aggregate(Avg('lowest_crs_score'))['lowest_crs_score__avg'],
            'min_crs_score': draws.aggregate(Min('lowest_crs_score'))['lowest_crs_score__min'],
            'max_crs_score': draws.aggregate(Max('lowest_crs_score'))['lowest_crs_score__max'],
            'total_invitations': draws.aggregate(total=Sum('invitations_issued'))['total'],
            'latest_draw': draws.order_by('-date').first()
        }
        
        return Response(stats)

    @action(detail=True, methods=['get'])
    def predictions(self, request, pk=None):
        """Get pre-computed predictions for a category"""
        category = self.get_object()
        predictions = PreComputedPrediction.objects.filter(
            category=category, 
            is_active=True
        ).order_by('prediction_rank')[:10]
        
        prediction_data = [{
            'rank': pred.prediction_rank,
            'predicted_date': pred.predicted_date,
            'predicted_crs_score': pred.predicted_crs_score,
            'predicted_invitations': pred.predicted_invitations,
            'confidence_score': pred.confidence_score,
            'model_used': pred.model_used,
            'uncertainty_range': pred.uncertainty_range,
            'created_at': pred.created_at
        } for pred in predictions]
        
        return Response({
            'category': category.name,
            'predictions': prediction_data,
            'total_predictions': len(prediction_data)
        })


class ExpressEntryDrawViewSet(viewsets.ModelViewSet):
    """API for managing Express Entry draws"""
    queryset = ExpressEntryDraw.objects.all().order_by('-date')
    serializer_class = ExpressEntryDrawSerializer
    pagination_class = None  # Disable pagination for complete historical data
    
    def get_queryset(self):
        """Filter draws by category and date range"""
        queryset = super().get_queryset()
        
        # Filter by category
        category_id = self.request.query_params.get('category', None)
        if category_id:
            queryset = queryset.filter(category_id=category_id)
        
        # Filter by date range
        start_date = self.request.query_params.get('start_date', None)
        end_date = self.request.query_params.get('end_date', None)
        
        if start_date:
            queryset = queryset.filter(date__gte=start_date)
        if end_date:
            queryset = queryset.filter(date__lte=end_date)
            
        return queryset

    @action(detail=False, methods=['get'])
    def recent(self, request):
        """Get recent draws (last 30 days)"""
        thirty_days_ago = timezone.now().date() - timedelta(days=30)
        recent_draws = self.get_queryset().filter(date__gte=thirty_days_ago)
        serializer = self.get_serializer(recent_draws, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def trends(self, request):
        """Get CRS score trends by category"""
        category_id = request.query_params.get('category')
        days = int(request.query_params.get('days', 180))  # Default 6 months
        
        start_date = timezone.now().date() - timedelta(days=days)
        queryset = self.get_queryset().filter(date__gte=start_date)
        
        if category_id:
            queryset = queryset.filter(category_id=category_id)
        
        # Group by month and calculate averages
        trends = queryset.extra(
            select={'month': "EXTRACT(month FROM date)", 'year': "EXTRACT(year FROM date)"}
        ).values('month', 'year', 'category__name').annotate(
            avg_crs=Avg('lowest_crs_score'),
            total_invitations=Sum('invitations_issued'),
            draw_count=Count('id')
        ).order_by('year', 'month')
        
        return Response(list(trends))


class PredictionModelViewSet(viewsets.ModelViewSet):
    """API for managing prediction models"""
    queryset = PredictionModel.objects.filter(is_active=True)
    serializer_class = PredictionModelSerializer


class DrawPredictionViewSet(viewsets.ModelViewSet):
    """API for managing predictions"""
    queryset = DrawPrediction.objects.all()
    serializer_class = DrawPredictionSerializer


# =================== CUSTOM API VIEWS ===================

class PredictionAPIView(APIView):
    """Fast prediction API using pre-computed predictions with IRCC category grouping"""
    
    def _group_predictions_by_date(self, predictions):
        """Group predictions by date and show all models for each date"""
        grouped = {}
        
        for pred in predictions:
            date_key = pred.predicted_date.isoformat()
            
            if date_key not in grouped:
                grouped[date_key] = {
                    'predicted_date': date_key,
                    'rank': pred.prediction_rank,
                    'models': []
                }
            
            # Add this model's prediction to the date group
            grouped[date_key]['models'].append({
                'model_name': pred.model_used,
                'predicted_crs_score': pred.predicted_crs_score,
                'predicted_invitations': pred.predicted_invitations,
                'confidence_score': round(pred.confidence_score * 100, 1),
                'uncertainty_range': pred.uncertainty_range
            })
        
        # Sort models within each date by confidence (highest first)
        for date_group in grouped.values():
            date_group['models'].sort(key=lambda x: x['confidence_score'], reverse=True)
        
        # Convert to list and sort by date
        result = list(grouped.values())
        result.sort(key=lambda x: x['predicted_date'])
        
        return result
    
    def get(self, request, category_id=None):
        """Get pre-computed predictions grouped by IRCC categories"""
        
        try:
            # Get cached predictions first
            cache_key = f"predictions_api_{category_id or 'all'}"
            cached_data = PredictionCache.get_cached(cache_key)
            
            if cached_data:
                return Response(cached_data)
            
            # Get categories with predictions (regardless of recent draw activity)
            if category_id:
                categories = DrawCategory.objects.filter(id=category_id, is_active=True)
            else:
                # Get all active categories that have predictions available
                categories_with_predictions = PreComputedPrediction.objects.filter(
                    is_active=True
                ).values_list('category', flat=True).distinct()
                
                categories = DrawCategory.objects.filter(
                    id__in=categories_with_predictions,
                    is_active=True
                )
            
            # Group categories by IRCC category to avoid duplicates (like compute_predictions does)
            ircc_groups = {}
            for category in categories:
                ircc_category, related_categories = DrawCategory.get_pooled_categories(category.name)
                
                # Use the IRCC category as the key
                if ircc_category not in ircc_groups:
                    # Find a representative category that has predictions
                    representative_category = None
                    for related_cat in related_categories:
                        if PreComputedPrediction.objects.filter(category=related_cat, is_active=True).exists():
                            representative_category = related_cat
                            break
                    
                    # If no related category has predictions, use the first one
                    if not representative_category and related_categories:
                        representative_category = related_categories.first()
                    
                    if representative_category:
                        # Get pooled data for better statistics
                        pooled_draws, _, num_pooled = representative_category.get_pooled_data()
                        
                        ircc_groups[ircc_category] = {
                            'representative_category': representative_category,
                            'related_categories': list(related_categories),
                            'pooled_draws': pooled_draws,
                            'total_draws': pooled_draws.count(),
                            'num_pooled': num_pooled
                        }
            
            results = []
            
            for ircc_category, group_info in ircc_groups.items():
                representative_category = group_info['representative_category']
                pooled_draws = group_info['pooled_draws']
                
                # Get pre-computed predictions (stored under representative category)
                predictions = PreComputedPrediction.objects.filter(
                    category=representative_category,
                    is_active=True
                ).order_by('prediction_rank')[:20]  # Next 20 draws for full year coverage
                
                if not predictions.exists():
                    continue
                
                # Get recent draws from ALL pooled categories for better context
                recent_draws = pooled_draws.order_by('-date')[:3]
                
                # Build category data using IRCC category name
                category_data = {
                    'category_id': representative_category.id,
                    'category_name': ircc_category,  # Use IRCC category name instead of individual
                    'category_description': f"Pooled data from {group_info['num_pooled']} category version(s)",
                    'total_draws': group_info['total_draws'],
                    'pooled_info': {
                        'num_versions': group_info['num_pooled'],
                        'related_categories': [cat.name for cat in group_info['related_categories']]
                    } if group_info['num_pooled'] > 1 else None,
                    'last_updated': predictions.first().updated_at.isoformat() if predictions else None,
                    'recent_draws': [{
                        'date': draw.date.isoformat(),
                        'crs_score': draw.lowest_crs_score,
                        'invitations': draw.invitations_issued
                    } for draw in recent_draws],
                    'predictions': self._group_predictions_by_date(predictions)
                }
                
                results.append(category_data)
            
            response_data = {
                'success': True,
                'data': results,
                'total_categories': len(results),
                'timestamp': timezone.now().isoformat(),
                'cache_duration': 3600  # Cache for 1 hour
            }
            
            # Cache the response
            PredictionCache.set_cache(cache_key, response_data, hours=1)
            
            return Response(response_data)
            
        except Exception as e:
            return Response({
                'success': False,
                'error': str(e),
                'data': []
            })


class DashboardStatsAPIView(APIView):
    """Optimized dashboard statistics API"""
    
    def get(self, request):
        """Get dashboard statistics using cached data"""
        
        try:
            # Try to get cached stats first
            cached_stats = PredictionCache.get_cached('dashboard_stats')
            
            if cached_stats:
                return Response(cached_stats)
            
            # Calculate fresh stats
            total_draws = ExpressEntryDraw.objects.count()
            
            # Count unique IRCC categories instead of individual categories
            all_active_categories = DrawCategory.objects.filter(is_active=True)
            active_with_recent_draws = [cat for cat in all_active_categories if cat.has_recent_activity(24)]
            
            # Group by IRCC category to get accurate count
            ircc_categories = set()
            for category in active_with_recent_draws:
                ircc_category, _ = DrawCategory.get_pooled_categories(category.name)
                ircc_categories.add(ircc_category)
            
            total_categories = len(ircc_categories)  # Count unique IRCC categories
            total_predictions = PreComputedPrediction.objects.filter(is_active=True).count()
            
            # Recent draws with IRCC category names
            recent_draws_raw = ExpressEntryDraw.objects.select_related('category').order_by('-date')[:10]
            
            # Next predicted draws with IRCC category names
            next_predictions_raw = PreComputedPrediction.objects.filter(
                is_active=True,
                prediction_rank=1  # Only next draw for each category
            ).select_related('category').order_by('predicted_date')[:10]
            
            # CRS score and invitation statistics
            crs_stats = ExpressEntryDraw.objects.aggregate(
                avg_crs=Avg('lowest_crs_score'),
                min_crs=Min('lowest_crs_score'),
                max_crs=Max('lowest_crs_score'),
                avg_invitations=Avg('invitations_issued')
            )
            
            # Date range statistics
            date_range = ExpressEntryDraw.objects.aggregate(
                start_date=Min('date'),
                end_date=Max('date')
            )
            
            stats = {
                # Frontend expects these exact property names
                'total_draws': total_draws,
                'categories_count': total_categories,
                'total_predictions': total_predictions,
                'avg_crs_score': round(crs_stats['avg_crs'] or 0, 1),
                'avg_invitations': round(crs_stats['avg_invitations'] or 0),
                'min_crs_score': crs_stats['min_crs'] or 0,
                'max_crs_score': crs_stats['max_crs'] or 0,
                'date_range': {
                    'start': date_range['start_date'].isoformat() if date_range['start_date'] else None,
                    'end': date_range['end_date'].isoformat() if date_range['end_date'] else None
                },
                'recent_draws': [{
                    'id': draw.id,
                    'round_number': draw.round_number,
                    'date': draw.date.isoformat(),
                    'category_name': DrawCategory.get_pooled_categories(draw.category.name)[0],  # Use IRCC category name
                    'lowest_crs_score': draw.lowest_crs_score,
                    'invitations_issued': draw.invitations_issued
                } for draw in recent_draws_raw],
                'next_predictions': [{
                    'category': DrawCategory.get_pooled_categories(pred.category.name)[0],  # Use IRCC category name
                    'predicted_date': pred.predicted_date.isoformat(),
                    'predicted_crs_score': pred.predicted_crs_score,
                    'confidence_score': round(pred.confidence_score * 100, 1)
                } for pred in next_predictions_raw],
                'last_updated': timezone.now().isoformat()
            }
            
            # Cache for 30 minutes
            PredictionCache.set_cache('dashboard_stats', stats, hours=0.5)
            
            return Response(stats)
            
        except Exception as e:
            return Response({
                'error': str(e),
                'totals': {'draws': 0, 'categories': 0, 'predictions': 0}
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class APIHealthCheckView(APIView):
    """Simple health check endpoint for API status"""
    
    def get(self, request):
        """Check API health and return system status"""
        try:
            # Quick database check
            total_draws = ExpressEntryDraw.objects.count()
            total_categories = DrawCategory.objects.filter(is_active=True).count()
            total_predictions = PreComputedPrediction.objects.filter(is_active=True).count()
            
            return Response({
                'status': 'healthy',
                'timestamp': timezone.now().isoformat(),
                'version': '1.0',
                'database': {
                    'connected': True,
                    'draws_count': total_draws,
                    'categories_count': total_categories,
                    'predictions_count': total_predictions
                },
                'endpoints': {
                    'predictions': '/api/predict/',
                    'categories': '/api/categories/',
                    'draws': '/api/draws/',
                    'statistics': '/api/stats/',
                    'documentation': '/api-docs/'
                }
            })
            
        except Exception as e:
            return Response({
                'status': 'unhealthy',
                'timestamp': timezone.now().isoformat(),
                'error': str(e)
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)


# =================== WEB VIEWS ===================

def home(request):
    """Home page with overview"""
    return render(request, 'predictor/home.html')


def predictions_page(request):
    """Predictions page"""
    return render(request, 'predictor/predictions.html')


def analytics_page(request):
    """Analytics page"""
    return render(request, 'predictor/analytics.html')


def api_docs(request):
    """API documentation page"""
    return render(request, 'predictor/api_docs.html')

def technical_docs(request):
    """Technical documentation page explaining datasets, models, and methodologies"""
    return render(request, 'predictor/technical_docs.html')

def draw_calculator(request):
    """Draw time calculator page for users to estimate their ITA timing"""
    return render(request, 'predictor/draw_calculator.html')


def category_detail(request, category_id):
    """Category detail page"""
    category = get_object_or_404(DrawCategory, id=category_id, is_active=True)
    
    # Get recent draws
    recent_draws = ExpressEntryDraw.objects.filter(
        category=category
    ).order_by('-date')[:20]
    
    # Get predictions
    predictions = PreComputedPrediction.objects.filter(
        category=category,
        is_active=True
    ).order_by('prediction_rank')[:10]
    
    context = {
        'category': category,
        'recent_draws': recent_draws,
        'predictions': predictions,
        'total_draws': recent_draws.count()
    }
    
    return render(request, 'predictor/category_detail.html', context)
