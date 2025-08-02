# 🔧 Admin Workflow - Express Entry Predictor

*Quick reference for updating predictions when new draws are published*

## 📊 **When New Express Entry Draws Are Published**

### 🚀 **Quick Update (2-3 commands)**

```bash
# 1. Update your CSV file with new draw data
# Edit: data/draw_data.csv

# 2. Load the new data
python3 manage.py load_draw_data

# 3. Regenerate all predictions
python3 manage.py compute_predictions --force

# 4. Auto-commit changes (optional)
./auto_commit.sh
```

**⏱️ Total time: ~2-3 minutes**

---

## 📈 **Available Commands**

### **Data Management**
```bash
# Load historical draws from CSV
python3 manage.py load_draw_data

# Setup prediction models (run once)
python3 manage.py setup_initial_data

# Check system health
python3 manage.py check
```

### **Prediction Generation**
```bash
# Generate predictions for all categories
python3 manage.py compute_predictions

# Force regenerate even if recent predictions exist
python3 manage.py compute_predictions --force

# Generate for specific category only
python3 manage.py compute_predictions --category "Canadian Experience Class"

# Generate more predictions (default is 10)
python3 manage.py compute_predictions --predictions 15
```

### **Model Testing**
```bash
# Test individual models
python3 manage.py run_predictions --model ensemble --steps 5
python3 manage.py run_predictions --model xgboost --steps 4
python3 manage.py run_predictions --model linear --steps 3

# Evaluate all model performance
python3 manage.py evaluate_models
```

---

## 🗂️ **Data File Format**

Your CSV file (`data/draw_data.csv`) should have these columns:

```csv
date,round_number,category,invitations_issued,lowest_crs_score,url
2025-08-15,91,Canadian Experience Class,1500,467,https://...
2025-08-15,91,Provincial Nominee Program,400,789,https://...
```

**Required columns:**
- `date` - Draw date (YYYY-MM-DD format)
- `round_number` - Draw round number
- `category` - Category name (exact match)
- `invitations_issued` - Number of invitations
- `lowest_crs_score` - Minimum CRS score
- `url` - Official announcement URL

---

## 🎯 **Categories with Sufficient Data**

These categories currently generate predictions (5+ draws):

✅ **Canadian Experience Class** (45 draws)  
✅ **Provincial Nominee Program** (84 draws)  
✅ **No Program Specified** (167 draws)  
✅ **French language proficiency** (20 draws)  
✅ **Federal Skilled Trades** (7 draws)  
✅ **Healthcare occupations** (6 draws)  
✅ **General** (11 draws)  

⚠️ **Categories with insufficient data** (< 5 draws):
- Agriculture and agri-food occupations (3 draws)
- Education occupations (1 draw)
- Federal Skilled Worker (1 draw)
- Healthcare and social services (3 draws)
- STEM occupations (3 draws)
- Trade occupations (4 draws)
- Transport occupations (3 draws)

---

## 🌐 **Website Pages**

After updating predictions, verify on:

- **🏠 Home**: http://127.0.0.1:8002/
- **🔮 Predictions**: http://127.0.0.1:8002/predictions/
- **📊 Analytics**: http://127.0.0.1:8002/analytics/
- **⚙️ Admin**: http://127.0.0.1:8002/admin/

---

## 🚨 **Troubleshooting**

### **Common Issues & Solutions**

**❌ No predictions generated:**
```bash
# Check if data loaded
python3 manage.py shell -c "from predictor.models import ExpressEntryDraw; print(f'Total draws: {ExpressEntryDraw.objects.count()}')"

# Force regenerate
python3 manage.py compute_predictions --force
```

**❌ "Insufficient data" error:**
- Category needs at least 5 draws to generate predictions
- Add more historical data for that category

**❌ Server not responding:**
```bash
# Restart development server
python3 manage.py runserver 8002
```

**❌ Database issues:**
```bash
# Apply any pending migrations
python3 manage.py migrate
```

---

## 📊 **Prediction Model Performance**

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| **Linear Regression** | ⚡⚡⚡ | ⭐⭐⭐ | Quick baseline |
| **Random Forest** | ⚡⚡ | ⭐⭐⭐⭐ | Robust predictions |
| **XGBoost** | ⚡⚡ | ⭐⭐⭐⭐⭐ | High accuracy |
| **LSTM** | ⚡ | ⭐⭐⭐⭐⭐ | Complex patterns |
| **Ensemble** | ⚡ | ⭐⭐⭐⭐⭐ | Maximum accuracy |

**System automatically selects the best model** for each category based on data quality and quantity.

---

## 🔄 **Automation Options**

### **Auto-Commit Script**
```bash
# Automatically commit and push changes
./auto_commit.sh
```

### **Scheduled Updates** (Optional)
For regular automated updates, you can set up:

```bash
# Daily check for new data (cron example)
0 9 * * * cd /path/to/ExpressEntryPredictor && python3 manage.py compute_predictions
```

---

## 📞 **Need Help?**

1. **Check this guide first** - covers 95% of common tasks
2. **Review logs**: Look at terminal output for error messages
3. **Test basic functionality**: `python3 manage.py check`
4. **Check deployment guide**: `DEPLOYMENT_GUIDE.md` for detailed info

---

## ✅ **Quick Checklist**

When new draws are published:

- [ ] Update `data/draw_data.csv` with new rows
- [ ] Run `python3 manage.py load_draw_data`
- [ ] Run `python3 manage.py compute_predictions --force`
- [ ] Verify website shows new predictions
- [ ] Commit changes: `./auto_commit.sh`

**🎉 You're done! New predictions are live.**

---

*Express Entry Predictor v2.0 - Admin Workflow Guide*  
*Updated: August 2025* 