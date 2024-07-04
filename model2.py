import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from scipy import stats

# Load the datasets
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

# Fill null values
train['Item_Weight'].fillna(train['Item_Weight'].mean(), inplace=True)
test['Item_Weight'].fillna(test['Item_Weight'].mean(), inplace=True)

train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0], inplace=True)
test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0], inplace=True)

# Simplify Item_Identifier
train['Item_Identifier'] = train['Item_Identifier'].str[:2]
test['Item_Identifier'] = test['Item_Identifier'].str[:2]

# Encode categorical variables
ordinal_cat_columns = ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type']
nominal_cat_columns = ['Item_Identifier', 'Item_Type', 'Outlet_Identifier', 'Outlet_Type']

# One-Hot Encoding for nominal categorical columns
train = pd.get_dummies(train, columns=nominal_cat_columns)
test = pd.get_dummies(test, columns=nominal_cat_columns)

# Initialize the OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

# Fit and transform the ordinal columns
train[ordinal_cat_columns] = ordinal_encoder.fit_transform(train[ordinal_cat_columns])
test[ordinal_cat_columns] = ordinal_encoder.transform(test[ordinal_cat_columns])

# Outlier detection and removal using Z-score
z_scores = np.abs(stats.zscore(train.select_dtypes(include=[np.number])))
filtered_entries = (z_scores < 3).all(axis=1)
train = train[filtered_entries]

# Splitting features and labels
y = train['Item_Outlet_Sales']
X = train.drop(columns=['Item_Outlet_Sales'])

# Ensure train and test have the same columns after encoding
test = test.reindex(columns=X.columns, fill_value=0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=786)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluating The Model
model_r2_score = r2_score(y_test, y_pred)
model_mae_score = mean_absolute_error(y_test, y_pred)
model_mse_score = mean_squared_error(y_test, y_pred)

print(f'Linear Regression R2 Score: {model_r2_score}')
print(f'Linear Regression Mean Absolute Error: {model_mae_score}')
print(f'Linear Regression Mean Squared Error: {model_mse_score}')

# Ridge Regression with Hyperparameter Tuning
ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5)
ridge_grid.fit(X_train, y_train)
ridge_best_model = ridge_grid.best_estimator_

y_pred_ridge = ridge_best_model.predict(X_test)
ridge_r2_score = r2_score(y_test, y_pred_ridge)
ridge_mae_score = mean_absolute_error(y_test, y_pred_ridge)
ridge_mse_score = mean_squared_error(y_test, y_pred_ridge)

print(f'Ridge Regression R2 Score: {ridge_r2_score}')
print(f'Ridge Regression Mean Absolute Error: {ridge_mae_score}')
print(f'Ridge Regression Mean Squared Error: {ridge_mse_score}')

# Lasso Regression with Hyperparameter Tuning
lasso_params = {'alpha': [0.01, 0.1, 1.0, 10.0]}
lasso_grid = GridSearchCV(Lasso(), lasso_params, cv=5)
lasso_grid.fit(X_train, y_train)
lasso_best_model = lasso_grid.best_estimator_

y_pred_lasso = lasso_best_model.predict(X_test)
lasso_r2_score = r2_score(y_test, y_pred_lasso)
lasso_mae_score = mean_absolute_error(y_test, y_pred_lasso)
lasso_mse_score = mean_squared_error(y_test, y_pred_lasso)

print(f'Lasso Regression R2 Score: {lasso_r2_score}')
print(f'Lasso Regression Mean Absolute Error: {lasso_mae_score}')
print(f'Lasso Regression Mean Squared Error: {lasso_mse_score}')

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rf_r2_score = r2_score(y_test, y_pred_rf)
rf_mae_score = mean_absolute_error(y_test, y_pred_rf)
rf_mse_score = mean_squared_error(y_test, y_pred_rf)

print(f'Random Forest R2 Score: {rf_r2_score}')
print(f'Random Forest Mean Absolute Error: {rf_mae_score}')
print(f'Random Forest Mean Squared Error: {rf_mse_score}')

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

gb_r2_score = r2_score(y_test, y_pred_gb)
gb_mae_score = mean_absolute_error(y_test, y_pred_gb)
gb_mse_score = mean_squared_error(y_test, y_pred_gb)

print(f'Gradient Boosting R2 Score: {gb_r2_score}')
print(f'Gradient Boosting Mean Absolute Error: {gb_mae_score}')
print(f'Gradient Boosting Mean Squared Error: {gb_mse_score}')

# Cross-Validation for Better Performance Estimation
cv_r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
cv_mae_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
cv_mse_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

print(f'Cross-Validation R2 Scores: {cv_r2_scores}')
print(f'Cross-Validation MAE Scores: {cv_mae_scores}')
print(f'Cross-Validation MSE Scores: {cv_mse_scores}')

# Plotting Actual vs Predicted Sales for the best model
plt.scatter(y_test, y_pred_rf)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales (Random Forest)')
plt.show()
