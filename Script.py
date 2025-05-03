import pandas as pd
import statsmodels.api as sm

# Load data (df includes columns: date, category, stress_test, sensitivity)
df = df.sort_values('date')

# Assign exponential decay weights (e.g., λ=0.1)
df['weight'] = np.exp(-0.1 * (df['date'].max() - df['date']).dt.days)

# Group by category and fit model
results = {}
for category, group in df.groupby('category'):
    X = group[['sensitivity']]
    X = sm.add_constant(X)  # Add intercept
    y = group['stress_test']
    weights = group['weight']
    
    model = sm.WLS(y, X, weights=weights).fit()
    residuals = model.resid
    group['residual'] = residuals
    results[category] = {
        'model': model,
        'residuals': residuals,
        'latest_residual': residuals.iloc[-1]
    }

# Detect outliers using MAD
for category, data in results.items():
    residuals = data['residuals']
    mad = np.median(np.abs(residuals - np.median(residuals)))
    threshold = 3 * mad
    is_outlier = abs(data['latest_residual']) > threshold
    print(f"Category {category}: Outlier = {is_outlier}")
