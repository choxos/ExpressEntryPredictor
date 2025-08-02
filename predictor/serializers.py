from rest_framework import serializers
from .models import (
    DrawCategory, ExpressEntryDraw, EconomicIndicator,
    PredictionModel, DrawPrediction, PredictionAccuracy
)


class DrawCategorySerializer(serializers.ModelSerializer):
    draw_count = serializers.SerializerMethodField()
    avg_crs_score = serializers.SerializerMethodField()
    
    class Meta:
        model = DrawCategory
        fields = ['id', 'name', 'code', 'description', 'is_active', 'draw_count', 'avg_crs_score']
    
    def get_draw_count(self, obj):
        return obj.expressentry_draws.count()
    
    def get_avg_crs_score(self, obj):
        draws = obj.expressentry_draws.all()
        if draws:
            return round(sum(draw.lowest_crs_score for draw in draws) / len(draws), 1)
        return 0


class ExpressEntryDrawSerializer(serializers.ModelSerializer):
    category_name = serializers.CharField(source='category.name', read_only=True)
    category_code = serializers.CharField(source='category.code', read_only=True)
    
    class Meta:
        model = ExpressEntryDraw
        fields = [
            'id', 'round_number', 'date', 'category', 'category_name', 'category_code',
            'invitations_issued', 'lowest_crs_score', 'url', 'days_since_last_draw',
            'is_weekend', 'is_holiday', 'month', 'quarter', 'created_at', 'updated_at'
        ]


class EconomicIndicatorSerializer(serializers.ModelSerializer):
    class Meta:
        model = EconomicIndicator
        fields = [
            'id', 'date', 'unemployment_rate', 'job_vacancy_rate',
            'gdp_growth', 'immigration_target', 'created_at', 'updated_at'
        ]


class PredictionModelSerializer(serializers.ModelSerializer):
    predictions_count = serializers.SerializerMethodField()
    
    class Meta:
        model = PredictionModel
        fields = [
            'id', 'name', 'model_type', 'version', 'description',
            'mae_score', 'mse_score', 'r2_score', 'date_accuracy',
            'parameters', 'feature_importance', 'is_active',
            'trained_on', 'created_at', 'predictions_count'
        ]
    
    def get_predictions_count(self, obj):
        return obj.drawprediction_set.count()


class DrawPredictionSerializer(serializers.ModelSerializer):
    category_name = serializers.CharField(source='category.name', read_only=True)
    category_code = serializers.CharField(source='category.code', read_only=True)
    model_name = serializers.CharField(source='model.name', read_only=True)
    model_type = serializers.CharField(source='model.model_type', read_only=True)
    days_until_prediction = serializers.SerializerMethodField()
    
    class Meta:
        model = DrawPrediction
        fields = [
            'id', 'category', 'category_name', 'category_code',
            'model', 'model_name', 'model_type',
            'predicted_date', 'predicted_crs_score', 'predicted_invitations',
            'date_confidence', 'score_confidence',
            'crs_score_lower', 'crs_score_upper',
            'date_range_start', 'date_range_end',
            'prediction_date', 'is_published', 'notes',
            'days_until_prediction'
        ]
    
    def get_days_until_prediction(self, obj):
        from datetime import date
        today = date.today()
        if obj.predicted_date > today:
            return (obj.predicted_date - today).days
        return 0


class PredictionAccuracySerializer(serializers.ModelSerializer):
    prediction_details = DrawPredictionSerializer(source='prediction', read_only=True)
    actual_draw_details = ExpressEntryDrawSerializer(source='actual_draw', read_only=True)
    
    class Meta:
        model = PredictionAccuracy
        fields = [
            'id', 'prediction', 'actual_draw',
            'date_error_days', 'score_error',
            'date_accuracy_score', 'score_accuracy_score',
            'evaluated_on', 'prediction_details', 'actual_draw_details'
        ]


class DrawStatsSerializer(serializers.Serializer):
    """Serializer for draw statistics"""
    total_draws = serializers.IntegerField()
    categories_count = serializers.IntegerField()
    date_range = serializers.DictField()
    avg_crs_score = serializers.FloatField()
    avg_invitations = serializers.FloatField()
    category_breakdown = serializers.ListField()
    monthly_trends = serializers.ListField()
    recent_draws = ExpressEntryDrawSerializer(many=True)


class PredictionResultSerializer(serializers.Serializer):
    """Serializer for prediction results"""
    category = DrawCategorySerializer()
    predictions = serializers.ListField()
    ensemble_prediction = serializers.DictField()
    model_performance = serializers.ListField()
    confidence_metrics = serializers.DictField()
    prediction_timeline = serializers.ListField()


class ModelTrainingResultSerializer(serializers.Serializer):
    """Serializer for model training results"""
    model_name = serializers.CharField()
    model_type = serializers.CharField()
    training_metrics = serializers.DictField()
    feature_importance = serializers.DictField()
    training_time = serializers.FloatField()
    success = serializers.BooleanField()
    error_message = serializers.CharField(required=False) 