from django.shortcuts import render
from django.http import JsonResponse
from django.db.models import Avg, Count, Min, Max
from rest_framework.views import APIView
from rest_framework.response import Response
from datetime import datetime, timedelta, date
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

from predictor.models import ExpressEntryDraw, DrawCategory, PredictionModel, DrawPrediction


def analytics_home(request):
    """Analytics home page"""
    return render(request, 'analytics/home.html')


def trends_view(request):
    """Trends analysis page"""
    return render(request, 'analytics/trends.html')


def statistics_view(request):
    """Statistics page"""
    return render(request, 'analytics/statistics.html')


class ChartsAPIView(APIView):
    """API for generating charts data"""
    
    def get(self, request):
        """Generate charts data for the dashboard"""
        
        charts = {}
        
        # 1. CRS Score Trends Over Time
        draws = ExpressEntryDraw.objects.order_by('date')
        dates = [draw.date for draw in draws]
        crs_scores = [draw.lowest_crs_score for draw in draws]
        categories = [draw.category.name for draw in draws]
        
        fig_crs_trend = go.Figure()
        
        # Group by category for better visualization
        for category in DrawCategory.objects.filter(is_active=True):
            cat_draws = draws.filter(category=category)
            if cat_draws.exists():
                fig_crs_trend.add_trace(go.Scatter(
                    x=[d.date for d in cat_draws],
                    y=[d.lowest_crs_score for d in cat_draws],
                    mode='lines+markers',
                    name=category.name,
                    line=dict(width=2)
                ))
        
        fig_crs_trend.update_layout(
            title='CRS Score Trends by Category',
            xaxis_title='Date',
            yaxis_title='CRS Score',
            hovermode='x unified'
        )
        
        charts['crs_trends'] = json.loads(json.dumps(fig_crs_trend, cls=PlotlyJSONEncoder))
        
        # 2. Monthly Draw Count
        monthly_data = {}
        for draw in draws:
            month_key = f"{draw.date.year}-{draw.date.month:02d}"
            if month_key not in monthly_data:
                monthly_data[month_key] = 0
            monthly_data[month_key] += 1
        
        fig_monthly = go.Figure(data=[
            go.Bar(
                x=list(monthly_data.keys()),
                y=list(monthly_data.values()),
                name='Draw Count'
            )
        ])
        
        fig_monthly.update_layout(
            title='Monthly Draw Count',
            xaxis_title='Month',
            yaxis_title='Number of Draws'
        )
        
        charts['monthly_draws'] = json.loads(json.dumps(fig_monthly, cls=PlotlyJSONEncoder))
        
        # 3. Category Distribution (Pie Chart)
        category_counts = {}
        for category in DrawCategory.objects.filter(is_active=True):
            count = ExpressEntryDraw.objects.filter(category=category).count()
            if count > 0:
                category_counts[category.name] = count
        
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=list(category_counts.keys()),
                values=list(category_counts.values()),
                hole=0.3
            )
        ])
        
        fig_pie.update_layout(title='Draw Distribution by Category')
        charts['category_distribution'] = json.loads(json.dumps(fig_pie, cls=PlotlyJSONEncoder))
        
        # 4. CRS Score Distribution (Histogram)
        all_scores = [draw.lowest_crs_score for draw in draws]
        
        fig_hist = go.Figure(data=[
            go.Histogram(
                x=all_scores,
                nbinsx=20,
                name='CRS Score Distribution'
            )
        ])
        
        fig_hist.update_layout(
            title='CRS Score Distribution',
            xaxis_title='CRS Score',
            yaxis_title='Frequency'
        )
        
        charts['score_distribution'] = json.loads(json.dumps(fig_hist, cls=PlotlyJSONEncoder))
        
        # 5. Invitations vs CRS Score Scatter Plot
        invitations = [draw.invitations_issued for draw in draws]
        
        fig_scatter = go.Figure(data=[
            go.Scatter(
                x=crs_scores,
                y=invitations,
                mode='markers',
                marker=dict(
                    color=categories,
                    size=8,
                    opacity=0.7
                ),
                text=[f"{cat}<br>Date: {date}<br>CRS: {crs}<br>Invitations: {inv}" 
                      for cat, date, crs, inv in zip(categories, dates, crs_scores, invitations)],
                hovertemplate='%{text}<extra></extra>'
            )
        ])
        
        fig_scatter.update_layout(
            title='CRS Score vs Invitations Issued',
            xaxis_title='CRS Score',
            yaxis_title='Invitations Issued'
        )
        
        charts['score_vs_invitations'] = json.loads(json.dumps(fig_scatter, cls=PlotlyJSONEncoder))
        
        return Response(charts)


class TrendsAPIView(APIView):
    """API for trends analysis"""
    
    def get(self, request):
        """Get trend analysis data"""
        
        trends = {}
        
        # Get recent data (last 2 years)
        two_years_ago = date.today() - timedelta(days=730)
        recent_draws = ExpressEntryDraw.objects.filter(date__gte=two_years_ago).order_by('date')
        
        # 1. Moving averages
        window_size = 10
        moving_averages = []
        
        for i, draw in enumerate(recent_draws):
            if i >= window_size - 1:
                window_draws = recent_draws[i-window_size+1:i+1]
                avg_crs = sum(d.lowest_crs_score for d in window_draws) / window_size
                avg_invitations = sum(d.invitations_issued for d in window_draws) / window_size
                
                moving_averages.append({
                    'date': draw.date,
                    'avg_crs': round(avg_crs, 2),
                    'avg_invitations': round(avg_invitations, 2)
                })
        
        trends['moving_averages'] = moving_averages
        
        # 2. Seasonal patterns
        seasonal_data = {}
        for draw in recent_draws:
            month = draw.date.month
            if month not in seasonal_data:
                seasonal_data[month] = {
                    'crs_scores': [],
                    'invitations': [],
                    'count': 0
                }
            
            seasonal_data[month]['crs_scores'].append(draw.lowest_crs_score)
            seasonal_data[month]['invitations'].append(draw.invitations_issued)
            seasonal_data[month]['count'] += 1
        
        seasonal_summary = []
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month in range(1, 13):
            if month in seasonal_data:
                data = seasonal_data[month]
                seasonal_summary.append({
                    'month': month_names[month-1],
                    'avg_crs': sum(data['crs_scores']) / len(data['crs_scores']),
                    'avg_invitations': sum(data['invitations']) / len(data['invitations']),
                    'draw_count': data['count']
                })
            else:
                seasonal_summary.append({
                    'month': month_names[month-1],
                    'avg_crs': 0,
                    'avg_invitations': 0,
                    'draw_count': 0
                })
        
        trends['seasonal_patterns'] = seasonal_summary
        
        # 3. Category trends
        category_trends = {}
        for category in DrawCategory.objects.filter(is_active=True):
            cat_draws = recent_draws.filter(category=category)
            if cat_draws.exists():
                # Calculate trend (simple linear regression slope)
                n = len(cat_draws)
                x_values = list(range(n))
                y_values = [d.lowest_crs_score for d in cat_draws]
                
                if n > 1:
                    x_mean = sum(x_values) / n
                    y_mean = sum(y_values) / n
                    
                    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
                    denominator = sum((x - x_mean) ** 2 for x in x_values)
                    
                    slope = numerator / denominator if denominator != 0 else 0
                    
                    category_trends[category.name] = {
                        'trend_slope': round(slope, 3),
                        'avg_crs': round(sum(y_values) / n, 1),
                        'draw_count': n,
                        'trend_direction': 'increasing' if slope > 0.1 else 'decreasing' if slope < -0.1 else 'stable'
                    }
        
        trends['category_trends'] = category_trends
        
        # 4. Prediction accuracy (if available)
        accuracy_data = []
        predictions = DrawPrediction.objects.filter(is_published=True)
        
        for prediction in predictions:
            # Try to find actual draw that matches the prediction
            actual_draws = ExpressEntryDraw.objects.filter(
                category=prediction.category,
                date__gte=prediction.predicted_date - timedelta(days=7),
                date__lte=prediction.predicted_date + timedelta(days=7)
            )
            
            if actual_draws.exists():
                actual_draw = actual_draws.first()
                date_error = abs((actual_draw.date - prediction.predicted_date).days)
                score_error = abs(actual_draw.lowest_crs_score - prediction.predicted_crs_score)
                
                accuracy_data.append({
                    'prediction_date': prediction.predicted_date,
                    'actual_date': actual_draw.date,
                    'predicted_score': prediction.predicted_crs_score,
                    'actual_score': actual_draw.lowest_crs_score,
                    'date_error_days': date_error,
                    'score_error': score_error,
                    'model': prediction.model.name,
                    'category': prediction.category.name
                })
        
        trends['prediction_accuracy'] = accuracy_data
        
        return Response(trends)
