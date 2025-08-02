from django.contrib import admin
from django.db.models import Avg, Count
from .models import (
    DrawCategory, ExpressEntryDraw, PredictionModel, 
    DrawPrediction, PredictionAccuracy, PreComputedPrediction, PredictionCache
)


@admin.register(DrawCategory)
class DrawCategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'code', 'is_active', 'draw_count', 'avg_crs_score')
    list_filter = ('is_active',)
    search_fields = ('name', 'code')
    readonly_fields = ('draw_count', 'avg_crs_score')
    
    def draw_count(self, obj):
        return obj.expressentrydraw_set.count()
    draw_count.short_description = 'Total Draws'
    
    def avg_crs_score(self, obj):
        avg = obj.expressentrydraw_set.aggregate(avg=Avg('lowest_crs_score'))['avg']
        return round(avg, 1) if avg else 0
    avg_crs_score.short_description = 'Avg CRS Score'


@admin.register(ExpressEntryDraw)
class ExpressEntryDrawAdmin(admin.ModelAdmin):
    list_display = (
        'round_number', 'date', 'category', 'lowest_crs_score', 
        'invitations_issued', 'days_since_last_draw'
    )
    list_filter = ('category', 'is_weekend', 'is_holiday', 'month', 'quarter')
    search_fields = ('round_number', 'category__name')
    ordering = ('-date',)
    date_hierarchy = 'date'
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('round_number', 'date', 'category', 'url')
        }),
        ('Draw Details', {
            'fields': ('invitations_issued', 'lowest_crs_score', 'days_since_last_draw')
        }),
        ('Metadata', {
            'fields': ('is_weekend', 'is_holiday', 'month', 'quarter'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    readonly_fields = ('created_at', 'updated_at', 'month', 'quarter', 'is_weekend')



@admin.register(PredictionModel)
class PredictionModelAdmin(admin.ModelAdmin):
    list_display = (
        'name', 'model_type', 'version', 'is_active', 
        'mae_score', 'r2_score', 'date_accuracy', 'trained_on'
    )
    list_filter = ('model_type', 'is_active')
    search_fields = ('name', 'description')
    ordering = ('-created_at',)
    
    fieldsets = (
        ('Model Information', {
            'fields': ('name', 'model_type', 'version', 'description', 'is_active')
        }),
        ('Performance Metrics', {
            'fields': ('mae_score', 'mse_score', 'r2_score', 'date_accuracy')
        }),
        ('Model Data', {
            'fields': ('parameters', 'feature_importance'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('trained_on', 'created_at'),
            'classes': ('collapse',)
        }),
    )
    readonly_fields = ('created_at',)


@admin.register(DrawPrediction)
class DrawPredictionAdmin(admin.ModelAdmin):
    list_display = (
        'category', 'model', 'predicted_date', 'predicted_crs_score',
        'date_confidence', 'score_confidence', 'is_published'
    )
    list_filter = ('category', 'model', 'is_published', 'predicted_date')
    search_fields = ('category__name', 'model__name')
    ordering = ('predicted_date',)
    date_hierarchy = 'predicted_date'
    
    fieldsets = (
        ('Prediction Details', {
            'fields': ('category', 'model', 'predicted_date', 'predicted_crs_score', 'predicted_invitations')
        }),
        ('Confidence & Ranges', {
            'fields': (
                'date_confidence', 'score_confidence',
                'crs_score_lower', 'crs_score_upper',
                'date_range_start', 'date_range_end'
            )
        }),
        ('Metadata', {
            'fields': ('is_published', 'notes', 'prediction_date')
        }),
    )
    readonly_fields = ('prediction_date',)


@admin.register(PredictionAccuracy)
class PredictionAccuracyAdmin(admin.ModelAdmin):
    list_display = (
        'model', 'actual_draw', 'predicted_score', 'actual_score', 'error', 'created_at'
    )
    list_filter = ('model', 'actual_draw__category')
    ordering = ('-created_at',)
    readonly_fields = ('error', 'created_at')
    
    def save_model(self, request, obj, form, change):
        # Calculate error automatically
        obj.error = abs(obj.predicted_score - obj.actual_score)
        super().save_model(request, obj, form, change)


@admin.register(PreComputedPrediction)
class PreComputedPredictionAdmin(admin.ModelAdmin):
    list_display = (
        'category', 'prediction_rank', 'predicted_date', 'predicted_crs_score', 
        'confidence_score', 'model_used', 'is_active', 'created_at'
    )
    list_filter = ('category', 'model_used', 'is_active', 'created_at')
    search_fields = ('category__name',)
    ordering = ('category', 'prediction_rank', 'predicted_date')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        ('Prediction Details', {
            'fields': ('category', 'prediction_rank', 'predicted_date', 'predicted_crs_score', 'predicted_invitations')
        }),
        ('Model Information', {
            'fields': ('model_used', 'model_version', 'confidence_score')
        }),
        ('Additional Data', {
            'fields': ('uncertainty_range', 'is_active'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(PredictionCache)
class PredictionCacheAdmin(admin.ModelAdmin):
    list_display = ('cache_key', 'expires_at', 'created_at')
    list_filter = ('expires_at', 'created_at')
    search_fields = ('cache_key',)
    ordering = ('-created_at',)
    readonly_fields = ('created_at',)


# Admin site customization
admin.site.site_header = "Express Entry Predictor Admin"
admin.site.site_title = "EEP Admin Portal"
admin.site.index_title = "Welcome to Express Entry Predictor Administration"
