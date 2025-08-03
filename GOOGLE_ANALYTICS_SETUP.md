# Google Analytics Setup Guide

## Overview

This guide explains how to set up Google Analytics (GA4) for the Express Entry Predictor website. The implementation includes privacy-compliant tracking, custom event monitoring, and comprehensive user behavior analytics.

## 🎯 Features Implemented

### Core Analytics
- **Page Views**: Automatic tracking of all page visits
- **User Sessions**: Session duration and bounce rate tracking
- **Geographic Data**: Visitor location analysis (anonymized)
- **Device Analytics**: Desktop vs mobile usage patterns

### Custom Event Tracking
- **Theme Changes**: Track light/dark mode preferences
- **Prediction Views**: Monitor which categories users view most
- **Calculator Usage**: Track draw calculator usage by category and CRS score
- **Chart Interactions**: Monitor user engagement with charts (hover, click, legend toggle)

### Privacy Features
- **IP Anonymization**: User IP addresses are anonymized
- **DNT Respect**: Honors "Do Not Track" browser settings
- **GDPR Compliant**: Privacy-focused implementation
- **Production Only**: Analytics only enabled in production environment

## 🚀 Setup Instructions

### 1. Create Google Analytics Account

1. Go to [Google Analytics](https://analytics.google.com/)
2. Click "Start measuring" or "Get started"
3. Create a new account for your website
4. Set up a new property for "Express Entry Predictor"
5. Choose "Web" as the platform
6. Enter your website URL and name

### 2. Get Your Tracking ID

1. In your GA4 property, go to **Admin** (gear icon)
2. Under **Property**, click **Data Streams**
3. Click on your web stream
4. Copy the **Measurement ID** (format: G-XXXXXXXXXX)

### 3. Configure Environment Variables

Add the Google Analytics ID to your environment variables:

#### For VPS/Production:
```bash
# Add to your environment file or set directly
export GOOGLE_ANALYTICS_ID="G-XXXXXXXXXX"

# Or add to /etc/environment
echo 'GOOGLE_ANALYTICS_ID="G-XXXXXXXXXX"' >> /etc/environment
```

#### For Local Development:
```bash
# Google Analytics is disabled in DEBUG mode for privacy
# No need to set for local development
```

### 4. Restart Your Application

```bash
# On VPS
sudo systemctl restart expressentry

# Or if using manual process management
# Kill and restart your Django process
```

### 5. Verify Implementation

1. **Real-time Reports**: Go to GA4 → Reports → Realtime
2. **Visit your website** in a new browser tab
3. **Check the realtime report** - you should see 1 active user
4. **Test custom events**:
   - Toggle dark/light theme (should log `theme_change` event)
   - View predictions (should log `prediction_view` event)
   - Use calculator (should log `calculator_use` event)
   - Hover over charts (should log `chart_interaction` event)

## 📊 Analytics Dashboard Setup

### Recommended Reports to Create

1. **User Engagement Report**
   - Page views by page
   - Session duration
   - Bounce rate by page

2. **Feature Usage Report**
   - Custom events by category
   - Theme preference distribution
   - Most viewed prediction categories
   - Calculator usage patterns

3. **Geographic Analysis**
   - Users by country
   - Popular pages by region
   - Session duration by location

### Custom Dimensions to Add

1. **Theme Preference** (light/dark)
2. **Prediction Category** (most viewed)
3. **Calculator Category** (most used)
4. **Chart Interaction Type** (hover/click/toggle)

## 🔍 Event Tracking Details

### Events Currently Tracked

| Event Name | Category | Description | Parameters |
|------------|----------|-------------|------------|
| `theme_change` | `user_preference` | User toggles theme | `event_label`: light/dark |
| `prediction_view` | `predictions` | User views prediction category | `event_label`: category name |
| `calculator_use` | `tools` | User calculates draw time | `event_label`: categories, `value`: CRS score |
| `chart_interaction` | `analytics` | User interacts with charts | `event_label`: chart_type_action |

### Chart Interaction Events

- **Timeline Chart**: `timeline_chart_hover`, `timeline_chart_click`, `timeline_chart_legend_toggle`
- **Moving Averages**: `moving_averages_hover`, `moving_averages_click`
- **Seasonal Patterns**: `seasonal_patterns_hover`, `seasonal_patterns_click`
- **Monthly Draws**: `monthly_draws_hover`, `monthly_draws_click`
- **CRS Trends**: `crs_trends_hover`, `crs_trends_click`, `crs_trends_legend_toggle`

## 🛡️ Privacy Compliance

### GDPR/Privacy Features
- **IP Anonymization**: `anonymize_ip: true`
- **DNT Respect**: `respect_dnt: true`
- **No Personal Data**: No collection of personal information
- **Production Only**: Analytics disabled in development mode

### Data Retention
- Configure data retention in GA4 settings
- Recommended: 14 months for user data
- Event data: 2 months for active analysis

## 🔧 Troubleshooting

### Analytics Not Working?

1. **Check Environment Variable**:
   ```bash
   echo $GOOGLE_ANALYTICS_ID
   ```

2. **Verify Production Mode**:
   ```bash
   # DEBUG should be False in production
   echo $DEBUG
   ```

3. **Check Browser Console**:
   - Open Developer Tools
   - Look for gtag errors in console
   - Verify gtag function exists: `typeof gtag`

4. **Test Custom Events**:
   ```javascript
   // In browser console
   window.trackEvent('test_event', 'test_category', 'test_label');
   ```

### Common Issues

1. **Events Not Showing**: Wait 24-48 hours for data to appear in reports
2. **Real-time Not Working**: Check for ad blockers or privacy extensions
3. **Tracking ID Invalid**: Ensure format is G-XXXXXXXXXX (not UA-XXXXXX)

## 📈 Analytics Insights You'll Get

### User Behavior
- Which prediction categories are most popular
- How users navigate through the site
- Time spent on different pages
- Device and browser preferences

### Feature Usage
- Dark vs light theme preference
- Calculator usage patterns
- Chart interaction frequency
- Geographic distribution of users

### Performance Metrics
- Page load times
- User engagement rates
- Session depth and duration
- Return visitor patterns

## 🔄 Maintenance

### Regular Tasks
1. **Monthly**: Review top events and pages
2. **Quarterly**: Analyze user flow and feature adoption
3. **Annually**: Update tracking for new features

### Adding New Events
```javascript
// Use the global tracking function
window.trackEvent('event_name', 'category', 'label', value);
```

## 📞 Support

For issues with Google Analytics setup:
1. Check the troubleshooting section above
2. Review Google Analytics documentation
3. Test in production environment (analytics disabled in development)

---

**Note**: Google Analytics is only enabled in production mode when a valid tracking ID is provided. This ensures privacy during development and testing. 