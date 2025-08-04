from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from datetime import datetime, date, timedelta
import json
import pytz


class DrawCategory(models.Model):
    """Express Entry draw categories"""
    name = models.CharField(max_length=100, unique=True)
    code = models.CharField(max_length=20, unique=True, default='')
    description = models.TextField(blank=True, null=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        verbose_name = "Draw Category"
        verbose_name_plural = "Draw Categories"
        ordering = ['name']
    
    def __str__(self):
        return self.name
    
    @classmethod
    def get_ircc_category_mapping(cls):
        """
        Map IRCC official categories to our database category names.
        This allows pooling data across different versions of the same category.
        Updated to match actual database categories.
        """
        return {
            'French-language proficiency': [
                'French language proficiency (Version 1)',
                # Note: Version 2 doesn't exist in current data
            ],
            'Healthcare and social services occupations': [
                'Healthcare occupations (Version 1)',                    # 6 draws
                'Healthcare and social services occupations (Version 2)', # 3 draws
                # Total: 9 draws when pooled
            ],
            'STEM occupations': [
                'STEM occupations (Version 1)',
                # Note: Full name version doesn't exist in current data
            ],
            'Trade occupations': [
                'Trade occupations (Version 1)',
                # Note: Version 2 doesn't exist in current data
            ],
            'Agriculture and agri-food occupations': [
                'Agriculture and agri-food occupations (Version 1)',
                # Note: Version 2 doesn't exist in current data
            ],
            'Education occupations': [
                'Education occupations (Version 1)',
                # Note: Version 2 doesn't exist in current data
            ],
            'Transport occupations': [
                'Transport occupations (Version 1)',
                # Note: Version 2 doesn't exist in current data
            ],
            # Non-category specific draws (these have good data volumes)
            'Canadian Experience Class': ['Canadian Experience Class'],              # 45 draws
            'Provincial Nominee Program': ['Provincial Nominee Program'],            # 84 draws
            'Federal Skilled Worker': ['Federal Skilled Worker'],                    # 1 draw
            'Federal Skilled Trades': ['Federal Skilled Trades'],                    # 7 draws
            'General': [
                'General',           # 11 draws
                'No Program Specified'  # 167 draws - POOL THESE FOR BETTER PREDICTIONS
                # Total: 178 draws when pooled
            ],
        }
    
    @classmethod
    def get_pooled_categories(cls, category_name):
        """
        Get all category objects that should be pooled with the given category.
        Returns the main IRCC category name and list of related DrawCategory objects.
        """
        mapping = cls.get_ircc_category_mapping()
        
        # Find which IRCC category this belongs to
        ircc_category = None
        related_names = []
        
        for ircc_cat, name_list in mapping.items():
            if category_name in name_list:
                ircc_category = ircc_cat
                related_names = name_list
                break
        
        if not ircc_category:
            # If not found in mapping, treat as standalone
            ircc_category = category_name
            related_names = [category_name]
        
        # Get actual DrawCategory objects that exist in database
        related_categories = cls.objects.filter(name__in=related_names)
        
        return ircc_category, related_categories
    
    def get_pooled_data(self):
        """
        Get combined historical data from all related category versions.
        Returns queryset of ExpressEntryDraw objects from pooled categories.
        """
        ircc_category, related_categories = self.get_pooled_categories(self.name)
        
        # Get all draws from related categories
        pooled_draws = ExpressEntryDraw.objects.filter(
            category__in=related_categories
        ).order_by('date')
        
        return pooled_draws, ircc_category, len(related_categories)
    
    def has_recent_activity(self, months=24):
        """Check if category has draws within the specified number of months (default: 24 months)"""
        from django.utils import timezone
        
        # Get current date in Eastern Time (Ottawa)
        eastern = pytz.timezone('America/Toronto')
        now_eastern = timezone.now().astimezone(eastern)
        cutoff_date = now_eastern.date() - timedelta(days=months * 30)
        
        # Check if there are any draws after the cutoff date
        latest_draw = self.expressentrydraw_set.order_by('-date').first()
        return latest_draw and latest_draw.date >= cutoff_date
    
    @property
    def latest_draw_date(self):
        """Get the date of the most recent draw for this category"""
        latest_draw = self.expressentrydraw_set.order_by('-date').first()
        return latest_draw.date if latest_draw else None
    
    @property
    def days_since_last_draw(self):
        """Get the number of days since the last draw"""
        if not self.latest_draw_date:
            return None
        
        from django.utils import timezone
        eastern = pytz.timezone('America/Toronto')
        now_eastern = timezone.now().astimezone(eastern)
        return (now_eastern.date() - self.latest_draw_date).days


class ExpressEntryDraw(models.Model):
    """Historical Express Entry draw data"""
    round_number = models.IntegerField(unique=True)
    date = models.DateField()
    category = models.ForeignKey(DrawCategory, on_delete=models.CASCADE)
    invitations_issued = models.IntegerField(validators=[MinValueValidator(1)])
    lowest_crs_score = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(1200)]
    )
    url = models.URLField(max_length=500, blank=True, null=True)
    
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


class PolicyAnnouncement(models.Model):
    """Government policy announcements that may affect draw patterns"""
    ANNOUNCEMENT_TYPES = [
        ('IMMIGRATION_TARGET', 'Immigration Level Target'),
        ('MINISTER_MANDATE', 'Minister Mandate Letter'),
        ('BUDGET_ANNOUNCEMENT', 'Budget/Fiscal Announcement'), 
        ('PROGRAM_CHANGE', 'Program Rule Change'),
        ('SPECIAL_MEASURE', 'Special/Temporary Measure'),
        ('INTERNATIONAL_AGREEMENT', 'International Agreement'),
    ]
    
    IMPACT_LEVELS = [
        ('HIGH', 'High Impact'),
        ('MEDIUM', 'Medium Impact'),
        ('LOW', 'Low Impact'),
    ]
    
    date = models.DateField()
    announcement_type = models.CharField(max_length=30, choices=ANNOUNCEMENT_TYPES)
    title = models.CharField(max_length=200)
    description = models.TextField()
    expected_impact = models.CharField(max_length=10, choices=IMPACT_LEVELS, default='MEDIUM')
    
    # Quantitative impacts (when known)
    target_change = models.IntegerField(blank=True, null=True, help_text="Change in immigration targets")
    effective_date = models.DateField(blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-date']
    
    def __str__(self):
        return f"{self.get_announcement_type_display()} - {self.date}"


class GovernmentContext(models.Model):
    """Political and governmental context affecting immigration policy"""
    GOVERNMENT_TYPES = [
        ('LIBERAL_MAJORITY', 'Liberal Majority'),
        ('LIBERAL_MINORITY', 'Liberal Minority'),
        ('CONSERVATIVE_MAJORITY', 'Conservative Majority'),
        ('CONSERVATIVE_MINORITY', 'Conservative Minority'),
        ('COALITION', 'Coalition Government'),
    ]
    
    start_date = models.DateField()
    end_date = models.DateField(blank=True, null=True)
    government_type = models.CharField(max_length=30, choices=GOVERNMENT_TYPES)
    prime_minister = models.CharField(max_length=100)
    immigration_minister = models.CharField(max_length=100)
    
    # Policy priorities (scale 1-10)
    economic_immigration_priority = models.IntegerField(default=7, 
        validators=[MinValueValidator(1), MaxValueValidator(10)])
    humanitarian_priority = models.IntegerField(default=5,
        validators=[MinValueValidator(1), MaxValueValidator(10)])
    francophone_priority = models.IntegerField(default=6,
        validators=[MinValueValidator(1), MaxValueValidator(10)])
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-start_date']
    
    def __str__(self):
        return f"{self.get_government_type_display()} ({self.start_date})"
        
    def is_active(self, date):
        """Check if this government context is active for a given date"""
        if date < self.start_date:
            return False
        if self.end_date and date > self.end_date:
            return False
        return True


class PoolComposition(models.Model):
    """Express Entry pool composition data"""
    date = models.DateField()
    
    # Pool size by CRS score ranges
    total_candidates = models.IntegerField(validators=[MinValueValidator(0)])
    candidates_600_plus = models.IntegerField(default=0, validators=[MinValueValidator(0)])
    candidates_500_599 = models.IntegerField(default=0, validators=[MinValueValidator(0)])
    candidates_450_499 = models.IntegerField(default=0, validators=[MinValueValidator(0)])
    candidates_400_449 = models.IntegerField(default=0, validators=[MinValueValidator(0)])
    candidates_below_400 = models.IntegerField(default=0, validators=[MinValueValidator(0)])
    
    # Pool dynamics
    new_registrations = models.IntegerField(blank=True, null=True)
    expired_profiles = models.IntegerField(blank=True, null=True)
    average_crs = models.FloatField(blank=True, null=True)
    median_crs = models.FloatField(blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-date']
        unique_together = ['date']
    
    def __str__(self):
        return f"Pool Composition - {self.date} ({self.total_candidates} candidates)"


class PNPActivity(models.Model):
    """Provincial Nominee Program activity data"""
    PROVINCES = [
        ('ON', 'Ontario'),
        ('BC', 'British Columbia'),
        ('AB', 'Alberta'),
        ('SK', 'Saskatchewan'),
        ('MB', 'Manitoba'),
        ('NS', 'Nova Scotia'),
        ('NB', 'New Brunswick'),
        ('NL', 'Newfoundland and Labrador'),
        ('PE', 'Prince Edward Island'),
        ('YT', 'Yukon'),
        ('NT', 'Northwest Territories'),
        ('NU', 'Nunavut'),
    ]
    
    date = models.DateField()
    province = models.CharField(max_length=2, choices=PROVINCES)
    invitations_issued = models.IntegerField(validators=[MinValueValidator(0)])
    minimum_score = models.IntegerField(blank=True, null=True)
    program_stream = models.CharField(max_length=100, blank=True, null=True)
    
    # Economic context for the province
    provincial_unemployment = models.FloatField(blank=True, null=True)
    key_sectors = models.JSONField(default=list, blank=True, help_text="Key economic sectors targeted")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-date', 'province']
    
    def __str__(self):
        return f"{self.get_province_display()} PNP - {self.date} ({self.invitations_issued} invites)"


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
    """Track prediction accuracy over time"""
    model = models.ForeignKey(PredictionModel, on_delete=models.CASCADE)
    actual_draw = models.ForeignKey(ExpressEntryDraw, on_delete=models.CASCADE)
    predicted_score = models.IntegerField(default=0)
    actual_score = models.IntegerField(default=0)
    error = models.FloatField(default=0.0)  # abs(predicted - actual)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'predictor_prediction_accuracy'


class PreComputedPrediction(models.Model):
    """Store pre-computed predictions for fast website loading"""
    category = models.ForeignKey(DrawCategory, on_delete=models.CASCADE)
    predicted_date = models.DateField()
    predicted_crs_score = models.IntegerField()
    predicted_invitations = models.IntegerField(null=True, blank=True)
    confidence_score = models.FloatField(default=0.75)
    model_used = models.CharField(max_length=100, default='Ensemble')
    model_version = models.CharField(max_length=20, default='1.0')
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    # Additional prediction details
    prediction_rank = models.IntegerField(default=1)  # 1st, 2nd, 3rd prediction etc.
    uncertainty_range = models.JSONField(default=dict, blank=True)  # {min: 450, max: 480}
    
    # Date confidence intervals (95% CI)
    predicted_date_lower = models.DateField(null=True, blank=True)  # Lower bound of date CI
    predicted_date_upper = models.DateField(null=True, blank=True)  # Upper bound of date CI
    interval_type = models.CharField(max_length=10, default='CI', help_text='CI for frequentist, CrI for Bayesian')
    
    class Meta:
        db_table = 'predictor_precomputed_prediction'
        ordering = ['category', 'prediction_rank', 'predicted_date']
        unique_together = ['category', 'prediction_rank', 'model_used']
    
    def __str__(self):
        return f"{self.category.name} - {self.predicted_date} (CRS: {self.predicted_crs_score})"


class PredictionCache(models.Model):
    """Cache for expensive calculations and model training results"""
    cache_key = models.CharField(max_length=255, unique=True)
    cache_data = models.JSONField()
    expires_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'predictor_prediction_cache'
    
    @classmethod
    def get_cached(cls, key):
        """Get cached data if not expired"""
        from django.utils import timezone
        try:
            cache_obj = cls.objects.get(cache_key=key, expires_at__gt=timezone.now())
            return cache_obj.cache_data
        except cls.DoesNotExist:
            return None
    
    @classmethod
    def set_cache(cls, key, data, hours=24):
        """Set cached data with expiration"""
        from django.utils import timezone
        expires_at = timezone.now() + timezone.timedelta(hours=hours)
        cls.objects.update_or_create(
            cache_key=key,
            defaults={'cache_data': data, 'expires_at': expires_at}
        )
