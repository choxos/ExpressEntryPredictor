from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    path('', views.analytics_home, name='home'),
    path('trends/', views.trends_view, name='trends'),
    path('statistics/', views.statistics_view, name='statistics'),
    path('api/charts/', views.ChartsAPIView.as_view(), name='api-charts'),
    path('api/trends/', views.TrendsAPIView.as_view(), name='api-trends'),
] 