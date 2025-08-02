from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from datetime import datetime, date
import json


class DrawCategory(models.Model):
    """Categories for Express Entry draws"""
    name = models.CharField(max_length=100, unique=True)
    code = models.CharField(max_length=20, unique=True)
    description = models.TextField(blank=True, null=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        verbose_name = "Draw Category"
        verbose_name_plural = "Draw Categories"
        ordering = ['name']
    
    def __str__(self):
        return self.name


class ExpressEntryDraw(models.Model):
    """Historical Express Entry draw data"""
    round_number = models.IntegerField(unique=True)
    date = models.DateField()
    category = models.ForeignKey(DrawCategory, on_delete=models.CASCADE)
    invitations_issued = models.IntegerField(validators=[MinValueValidator(1)])
    lowest_crs_score = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(1200)]
    )
    url = models.URLField(blank=True, null=True)
    
    # Additional derived fields
    days_since_last_draw = models.IntegerField(blank=True, null=True)
    is_weekend = models.BooleanField(default=False)
    is_holiday = models.BooleanField(default=False)
    month = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(12)])
    quarter = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(4)])
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-date']
        unique_together = ['date', 'category']
    
    def save(self, *args, **kwargs):
        # Auto-populate derived fields
        self.month = self.date.month
        self.quarter = (self.date.month - 1) // 3 + 1
        self.is_weekend = self.date.weekday() >= 5
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"Round {self.round_number} - {self.category.name} ({self.date})"


class EconomicIndicator(models.Model):
    """Economic indicators that might influence draw patterns"""
    date = models.DateField()
    unemployment_rate = models.FloatField(blank=True, null=True)
    job_vacancy_rate = models.FloatField(blank=True, null=True)
    gdp_growth = models.FloatField(blank=True, null=True)
    immigration_target = models.IntegerField(blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-date']
        unique_together = ['date']
    
    def __str__(self):
        return f"Economic Data - {self.date}"


class PredictionModel(models.Model):
    """Different prediction models used"""
    MODEL_TYPES = [
        ('ARIMA', 'ARIMA Time Series'),
        ('RF', 'Random Forest'),
        ('XGB', 'XGBoost'),
        ('LSTM', 'Long Short-Term Memory'),
        ('LR', 'Linear Regression'),
        ('ENSEMBLE', 'Ensemble Model'),
    ]
    
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    version = models.CharField(max_length=20, default='1.0')
    description = models.TextField(blank=True, null=True)
    
    # Model performance metrics
    mae_score = models.FloatField(blank=True, null=True, help_text="Mean Absolute Error for CRS Score")
    mse_score = models.FloatField(blank=True, null=True, help_text="Mean Squared Error for CRS Score")
    r2_score = models.FloatField(blank=True, null=True, help_text="R² Score for CRS Score")
    date_accuracy = models.FloatField(blank=True, null=True, help_text="Date prediction accuracy (%)")
    
    # Model parameters (stored as JSON)
    parameters = models.JSONField(default=dict, blank=True)
    feature_importance = models.JSONField(default=dict, blank=True)
    
    is_active = models.BooleanField(default=True)
    trained_on = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        unique_together = ['name', 'version']
    
    def __str__(self):
        return f"{self.name} v{self.version} ({self.model_type})"


class DrawPrediction(models.Model):
    """Predictions for future draws"""
    category = models.ForeignKey(DrawCategory, on_delete=models.CASCADE)
    model = models.ForeignKey(PredictionModel, on_delete=models.CASCADE)
    
    predicted_date = models.DateField()
    predicted_crs_score = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(1200)]
    )
    predicted_invitations = models.IntegerField(
        validators=[MinValueValidator(1)], blank=True, null=True
    )
    
    # Confidence intervals
    date_confidence = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Confidence percentage for date prediction"
    )
    score_confidence = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Confidence percentage for CRS score prediction"
    )
    
    # Ranges
    crs_score_lower = models.IntegerField(blank=True, null=True)
    crs_score_upper = models.IntegerField(blank=True, null=True)
    date_range_start = models.DateField(blank=True, null=True)
    date_range_end = models.DateField(blank=True, null=True)
    
    # Metadata
    prediction_date = models.DateTimeField(auto_now_add=True)
    is_published = models.BooleanField(default=False)
    notes = models.TextField(blank=True, null=True)
    
    class Meta:
        ordering = ['predicted_date']
        unique_together = ['category', 'model', 'predicted_date']
    
    def __str__(self):
        return f"{self.category.name} - {self.predicted_date} (CRS: {self.predicted_crs_score})"


class PredictionAccuracy(models.Model):
    """Track prediction accuracy against actual results"""
    prediction = models.ForeignKey(DrawPrediction, on_delete=models.CASCADE)
    actual_draw = models.ForeignKey(ExpressEntryDraw, on_delete=models.CASCADE)
    
    date_error_days = models.IntegerField(help_text="Days difference between predicted and actual date")
    score_error = models.IntegerField(help_text="CRS score difference between predicted and actual")
    
    date_accuracy_score = models.FloatField(help_text="Date prediction accuracy (0-100)")
    score_accuracy_score = models.FloatField(help_text="CRS score prediction accuracy (0-100)")
    
    evaluated_on = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['prediction', 'actual_draw']
    
    def __str__(self):
        return f"Accuracy for {self.prediction} vs {self.actual_draw}"
