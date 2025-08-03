from django.conf import settings

def google_analytics(request):
    """
    Context processor to make Google Analytics settings available in templates.
    """
    return {
        'settings': {
            'GOOGLE_ANALYTICS_ID': settings.GOOGLE_ANALYTICS_ID,
            'GOOGLE_ANALYTICS_ENABLED': settings.GOOGLE_ANALYTICS_ENABLED,
        }
    } 