import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Sample Data (replace with your actual dataset)
# Assume df is your dataset and the target variable is 'target'
# X = df.drop('target', axis=1)
# y = df['target']
X, y = np.random.rand(1000, 10), np.random.rand(1000)  # Replace with your actual dataset

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to evaluate
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
    'CatBoost': CatBoostRegressor(iterations=100, random_state=42, silent=True)  # Silent suppresses logging
}

# Dictionary to store the results
results = {'Model': [], 'MAE': [], 'R2': []}

# Train each model and evaluate its performance
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate MAE and R²
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store the results
    results['Model'].append(model_name)
    results['MAE'].append(mae)
    results['R2'].append(r2)

# Create a DataFrame to display the benchmark
results_df = pd.DataFrame(results)

# Print the benchmark
print(results_df)
