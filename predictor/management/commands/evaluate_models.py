from django.core.management.base import BaseCommand
from django.utils import timezone
import pandas as pd
import numpy as np

from predictor.models import DrawCategory, ExpressEntryDraw
from predictor.ml_models import (
    ARIMAPredictor, RandomForestPredictor, XGBoostPredictor, 
    LinearRegressionPredictor, NeuralNetworkPredictor,
    ProphetPredictor, LSTMPredictor
)


class Command(BaseCommand):
    help = 'Evaluate and compare all available models using statistical criteria'

    def add_arguments(self, parser):
        parser.add_argument(
            '--category',
            type=str,
            help='Specific category name to evaluate models for',
        )
        parser.add_argument(
            '--min-data-points',
            type=int,
            default=10,
            help='Minimum data points required for evaluation',
        )

    def handle(self, *args, **options):
        category_name = options.get('category')
        min_data_points = options.get('min_data_points', 10)
        
        self.stdout.write('🧪 Model Evaluation & Comparison System')
        self.stdout.write('=' * 50)
        
        # Get categories to evaluate
        if category_name:
            categories = DrawCategory.objects.filter(name__icontains=category_name)
            if not categories.exists():
                self.stdout.write(self.style.ERROR(f'❌ Category "{category_name}" not found'))
                return
        else:
            categories = DrawCategory.objects.all()
        
        total_categories = len(categories)
        self.stdout.write(f'📊 Evaluating {total_categories} categories with min {min_data_points} data points\n')
        
        results_summary = []
        
        for i, category in enumerate(categories, 1):
            self.stdout.write(f'[{i}/{total_categories}] 🔍 Evaluating category: {category.name}')
            
            # Get data for this category
            draws = ExpressEntryDraw.objects.filter(category=category).order_by('date')
            
            if draws.count() < min_data_points:
                self.stdout.write(f'  ⏭️  Skipped - insufficient data ({draws.count()} < {min_data_points})')
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(draws.values('date', 'lowest_crs_score', 'invitations_issued'))
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Evaluate models for this category
            try:
                results = self._evaluate_models_for_category(df, category.name)
                results_summary.append({
                    'category': category.name,
                    'data_points': len(df),
                    'results': results
                })
                
                # Display results
                self._display_category_results(category.name, results)
                
            except Exception as e:
                self.stdout.write(f'  ❌ Error evaluating {category.name}: {e}')
        
        # Overall summary
        self._display_overall_summary(results_summary)
        
        self.stdout.write('\n✅ Model evaluation completed!')

    def _evaluate_models_for_category(self, df, category_name):
        """Evaluate all available models for a single category"""
        
        # Define models to test
        models_to_test = [
            ('Linear Regression', LinearRegressionPredictor()),
            ('Random Forest', RandomForestPredictor()),
        ]
        
        data_size = len(df)
        
        # Add models based on data size
        if data_size >= 8:
            try:
                models_to_test.append(('XGBoost', XGBoostPredictor()))
            except ImportError:
                pass
        
        if data_size >= 10:
            try:
                models_to_test.append(('ARIMA', ARIMAPredictor()))
            except ImportError:
                pass
                
        if data_size >= 15:
            try:
                models_to_test.extend([
                    ('LSTM', LSTMPredictor()),
                    ('Prophet', ProphetPredictor()),
                    ('Neural Network', NeuralNetworkPredictor()),
                ])
            except ImportError:
                pass
        
        results = {}
        target_col = 'lowest_crs_score'
        
        for name, model in models_to_test:
            try:
                result = self._evaluate_single_model(model, df, target_col, name)
                if result:
                    results[name] = result
                    
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return results

    def _evaluate_single_model(self, model, df, target_col, model_name):
        """Evaluate a single model using cross-validation and performance metrics"""
        
        try:
            # Prepare data
            X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
            y = df[target_col]
            
            if X.empty or len(X.columns) == 0:
                X = pd.DataFrame({'time_index': range(len(df))})
            
            # Cross-validation
            cv_scores = []
            n_folds = min(3, len(df) // 2)
            
            if n_folds >= 2:
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=n_folds, shuffle=False)
                
                for train_idx, val_idx in kf.split(X):
                    try:
                        train_df = df.iloc[train_idx].copy()
                        val_df = df.iloc[val_idx].copy()
                        
                        # Create model copy and train
                        model_copy = self._copy_model(model)
                        model_copy.train(train_df, target_col)
                        
                        # Predict
                        if hasattr(model_copy, 'predict'):
                            X_val = X.iloc[val_idx]
                            pred = model_copy.predict(X_val)
                            if isinstance(pred, (list, np.ndarray)):
                                pred = pred[0] if len(pred) > 0 else y.iloc[val_idx].mean()
                            
                            actual = y.iloc[val_idx].iloc[0] if len(y.iloc[val_idx]) > 0 else 0
                            mae = abs(pred - actual)
                            cv_scores.append(-mae)  # Negative MAE for scoring
                            
                    except Exception as e:
                        continue
            
            # Train on full dataset
            model.train(df, target_col)
            
            # Get metrics
            if hasattr(model, 'metrics') and model.metrics:
                mae = model.metrics.get('mae', np.inf)
                mse = model.metrics.get('mse', np.inf)
                r2 = model.metrics.get('r2', -np.inf)
            else:
                mae = mse = np.inf
                r2 = -np.inf
            
            # Calculate information criteria
            aic, bic = self._calculate_information_criteria(model, df, target_col)
            
            return {
                'cv_score': np.mean(cv_scores) if cv_scores else -mae,
                'cv_std': np.std(cv_scores) if cv_scores else 0,
                'n_cv_folds': len(cv_scores),
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'aic': aic,
                'bic': bic,
                'model': model
            }
            
        except Exception as e:
            return {'error': str(e)}

    def _copy_model(self, model):
        """Create a copy of the model"""
        model_class = type(model)
        try:
            return model_class()
        except:
            return model

    def _calculate_information_criteria(self, model, df, target_col):
        """Calculate AIC and BIC"""
        
        try:
            if hasattr(model, 'aic') and hasattr(model, 'bic'):
                return getattr(model, 'aic', np.inf), getattr(model, 'bic', np.inf)
            
            n = len(df)
            if hasattr(model, 'metrics') and model.metrics:
                mse = model.metrics.get('mse', 1.0)
            else:
                mse = 1.0
            
            # Estimate parameters
            if hasattr(model, 'n_estimators'):
                n_params = getattr(model, 'n_estimators', 100) // 20  # Simplified
            elif hasattr(model, 'coef_'):
                n_params = len(getattr(model, 'coef_', [1]))
            else:
                n_params = 3
            
            # Calculate AIC/BIC
            log_likelihood = -0.5 * n * (np.log(2 * np.pi * mse) + 1)
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + n_params * np.log(n)
            
            return aic, bic
            
        except:
            return np.inf, np.inf

    def _display_category_results(self, category_name, results):
        """Display results for a single category"""
        
        self.stdout.write(f'  📈 Results for {category_name}:')
        
        if not results:
            self.stdout.write('    ❌ No successful evaluations')
            return
        
        # Sort by composite score
        scored_results = []
        for name, result in results.items():
            if 'error' in result:
                self.stdout.write(f'    ❌ {name}: {result["error"]}')
                continue
                
            # Calculate composite score
            cv_score = result.get('cv_score', -100)
            r2 = result.get('r2', 0)
            mae = result.get('mae', 100)
            
            composite_score = cv_score * 0.5 + r2 * 0.3 - mae * 0.2
            scored_results.append((name, result, composite_score))
        
        # Sort by score (highest first)
        scored_results.sort(key=lambda x: x[2], reverse=True)
        
        for i, (name, result, score) in enumerate(scored_results, 1):
            symbol = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '📊'
            cv_score = result.get('cv_score', 0)
            r2 = result.get('r2', 0)
            mae = result.get('mae', 0)
            aic = result.get('aic', np.inf)
            n_folds = result.get('n_cv_folds', 0)
            
            self.stdout.write(
                f'    {symbol} #{i} {name}: '
                f'CV={cv_score:.3f} R²={r2:.3f} MAE={mae:.2f} '
                f'AIC={aic:.1f} Folds={n_folds}'
            )

    def _display_overall_summary(self, results_summary):
        """Display overall summary across all categories"""
        
        if not results_summary:
            return
        
        self.stdout.write('\n' + '=' * 50)
        self.stdout.write('📊 OVERALL SUMMARY')
        self.stdout.write('=' * 50)
        
        # Model win counts
        model_wins = {}
        model_performance = {}
        
        for category_data in results_summary:
            results = category_data['results']
            
            # Find best model for this category
            best_model = None
            best_score = -np.inf
            
            for name, result in results.items():
                if 'error' in result:
                    continue
                    
                cv_score = result.get('cv_score', -100)
                r2 = result.get('r2', 0)
                mae = result.get('mae', 100)
                composite_score = cv_score * 0.5 + r2 * 0.3 - mae * 0.2
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_model = name
                
                # Track performance
                if name not in model_performance:
                    model_performance[name] = []
                model_performance[name].append({
                    'cv_score': cv_score,
                    'r2': r2,
                    'mae': mae,
                    'composite': composite_score
                })
            
            if best_model:
                model_wins[best_model] = model_wins.get(best_model, 0) + 1
        
        # Display win counts
        self.stdout.write('\n🏆 Model Win Counts:')
        for model, wins in sorted(model_wins.items(), key=lambda x: x[1], reverse=True):
            percentage = (wins / len(results_summary)) * 100
            self.stdout.write(f'  {model}: {wins} wins ({percentage:.1f}%)')
        
        # Display average performance
        self.stdout.write('\n📈 Average Performance:')
        for model, performances in model_performance.items():
            if performances:
                avg_cv = np.mean([p['cv_score'] for p in performances])
                avg_r2 = np.mean([p['r2'] for p in performances])
                avg_mae = np.mean([p['mae'] for p in performances])
                avg_composite = np.mean([p['composite'] for p in performances])
                
                self.stdout.write(
                    f'  {model}: CV={avg_cv:.3f} R²={avg_r2:.3f} '
                    f'MAE={avg_mae:.2f} Composite={avg_composite:.3f}'
                ) 