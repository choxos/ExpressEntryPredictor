from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# API Router
router = DefaultRouter()
router.register(r'categories', views.DrawCategoryViewSet)
router.register(r'draws', views.ExpressEntryDrawViewSet)
router.register(r'models', views.PredictionModelViewSet)
router.register(r'predictions', views.DrawPredictionViewSet)

app_name = 'predictor'

urlpatterns = [
    # Web views
    path('', views.home_view, name='home'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('analytics/', views.analytics_view, name='analytics'),
    path('predictions/', views.predictions_view, name='predictions'),
    
    # API endpoints
    path('api/', include(router.urls)),
    path('api/predict/', views.PredictionAPIView.as_view(), name='api-predict'),
    path('api/predict/<int:category_id>/', views.PredictionAPIView.as_view(), name='api-predict-category'),
    path('api/stats/', views.DashboardStatsAPIView.as_view(), name='api-stats'),
] 