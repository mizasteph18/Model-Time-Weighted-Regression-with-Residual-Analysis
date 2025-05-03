# Model-Time-Weighted-Regression-with-Residual-Analysis

To determine if the latest stress test figure is an outlier while accounting for portfolio changes and emphasizing recent data, the following approach is recommended:

### **Model: Time-Weighted Regression with Residual Analysis**
This method combines regression modeling (to control for portfolio sensitivity and client category) with time-based weighting (to prioritize recent data) and robust outlier detection.

---

#### **Steps to Implement**:

1. **Group Data by Client Category**  
   Segment the data by `client category` to account for category-specific behaviors.

2. **Build a Time-Weighted Regression Model**  
   Use a model like **Weighted Least Squares (WLS)** with exponential decay weights to prioritize recent observations.  
   - **Predictors**: Time (optional), portfolio sensitivity (`gross sensitivity`).  
   - **Response Variable**: Stress test figures.  
   - **Weights**: Assign weights using a decay factor (e.g., \( \text{weight}_t = e^{-\lambda (T-t)} \), where \( \lambda \) controls decay strength and \( T \) is the latest timestamp).

   Example formula:  
   \[
   \text{StressTest}_t = \beta_0 + \beta_1 \cdot \text{Sensitivity}_t + \epsilon_t
   \]  
   Recent data points (\( t \approx T \)) receive higher weights.

3. **Compute Residuals**  
   Calculate residuals (\( \epsilon_t = \text{Actual}_t - \text{Predicted}_t \)) for each observation. The residual of the latest data point is critical.

4. **Detect Outliers in Residuals**  
   Use robust outlier detection on residuals:  
   - **Median Absolute Deviation (MAD)**: Flag outliers where \( |\epsilon_T| > 3 \cdot \text{MAD} \).  
   - **Time-Weighted Z-Score**: Compute a Z-score for \( \epsilon_T \) using a weighted mean/standard deviation of residuals.

5. **Optional: Bayesian or Rolling-Window Refinement**  
   - **Bayesian Structural Time Series (BSTS)**: Incorporate sensitivity as a covariate and update posteriors dynamically.  
   - **Rolling Window Regression**: Re-estimate the model over a fixed window (e.g., last 12 months) to adapt to portfolio changes.

---

### **Advantages**:
- **Time Sensitivity**: Explicitly weights recent data more heavily.  
- **Controls for Portfolio Changes**: Uses `gross sensitivity` as a predictor to isolate stress test variations due to portfolio composition.  
- **Robust to Non-Stationarity**: Residual-based detection avoids assuming a fixed distribution.  
- **Interpretability**: Regression coefficients quantify how sensitivity impacts stress tests.

---

### **Example Code (Python)**:
```python
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
```

---

### **Alternatives**:
- **Isolation Forest with Time Features**: Encode time decay as a feature (e.g., `1/(1 + days_ago)`).  
- **Dynamic Linear Models (DLM)**: State-space models that adapt parameters over time (use `Pyro` or `pymc3`).  
- **Prophet with Regressors**: Facebook’s Prophet for time series with covariates.

Choose the method that balances interpretability, data size, and computational constraints. Start with the time-weighted regression for simplicity.
