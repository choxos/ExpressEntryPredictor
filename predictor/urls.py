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
    path('', views.home, name='home'),
    path('analytics/', views.analytics_page, name='analytics'),
    path('predictions/', views.predictions_page, name='predictions'),
    path('api-docs/', views.api_docs, name='api-docs'),
    path('category/<int:category_id>/', views.category_detail, name='category-detail'),
    
    # API endpoints
    path('api/', include(router.urls)),
    path('api/health/', views.APIHealthCheckView.as_view(), name='api-health'),
    path('api/predict/', views.PredictionAPIView.as_view(), name='api-predict'),
    path('api/predict/<int:category_id>/', views.PredictionAPIView.as_view(), name='api-predict-category'),
    path('api/stats/', views.DashboardStatsAPIView.as_view(), name='api-stats'),
] 