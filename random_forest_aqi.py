
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=UserWarning)

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('DATASET_NAME.csv')

# Create a new 'Date' column using 'Year', 'Month', and 'Day'
data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
data.set_index('Date', inplace=True)

# Handle missing values using time-based interpolation
data = data.interpolate(method='time')

# Add lag and rolling average features
for i in range(1, 4):
    data[f"AQI_lag_{i}"] = data['AQI'].shift(i)
data['AQI_rolling_avg_3'] = data['AQI'].rolling(window=3).mean()
data['AQI_rolling_avg_7'] = data['AQI'].rolling(window=7).mean()

# One-hot encode the 'Location' column and drop irrelevant columns
data.reset_index(inplace=True)
processed_data = pd.get_dummies(data, columns=['Location'], drop_first=True)
processed_data = processed_data.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Filename', 'AQI_Class'])
processed_data = processed_data.dropna()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
X = processed_data.drop(columns=['Date', 'AQI'])
y = processed_data['AQI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
param_dist = {
    'n_estimators': [50, 100, 200, 500, 1000],
    'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6, 8]
}
# Initialize the Random Forest regressor
rf = RandomForestRegressor(random_state=42)
# Use TimeSeriesSplit for time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
# Initialize RandomizedSearchCV with more iterations
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                   n_iter=50, scoring='neg_mean_squared_error',
                                   cv=tscv, verbose=2, random_state=42, n_jobs=-1)
# Fit RandomizedSearchCV to the training data
random_search.fit(X_train, y_train)
# Extract the best hyperparameters and train the Random Forest model with them
best_params = random_search.best_params_
rf_best = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42
)
rf_best.fit(X_train, y_train)

# Predictions using the best Random Forest model (before Human-in-the-Loop corrections)
y_pred_initial = rf_best.predict(X_test)

# Calculate RMSE for the initial model
rmse_initial = mean_squared_error(y_test, y_pred_initial, squared=False)

def compute_uncertainty(input_data, model):
    tree_predictions = np.array([tree.predict(input_data) for tree in model.estimators_])
    return np.var(tree_predictions, axis=0)

# Compute uncertainties and identify high uncertainty predictions
uncertainties = compute_uncertainty(X_test, rf_best)
threshold = np.percentile(uncertainties, 90)
high_uncertainty_indices = np.where(uncertainties > threshold)[0]

# Predictions using the best Random Forest model
y_pred_initial = rf_best.predict(X_test)

# Simulate human corrections
y_pred_initial[high_uncertainty_indices] = y_test.iloc[high_uncertainty_indices].values

# Retrain the model with corrections
X_combined = np.vstack((X_train, X_test))
y_combined = np.concatenate((y_train, y_pred_initial))
rf_best.fit(X_combined, y_combined)

# Predictions using the refined model (after Human-in-the-Loop corrections)
y_pred_refined = rf_best.predict(X_test)

# Calculate RMSE for the refined model
rmse_refined = mean_squared_error(y_test, y_pred_refined, squared=False)

rmse_initial

rmse_refined