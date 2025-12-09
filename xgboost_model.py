import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

# 1. Load Data (Using the links from the script)
print("Downloading and loading data...")
train_url = 'https://docs.google.com/uc?export=download&id=1-MP9hUKl7g0-zc0og1etVY1d4hmzDp5o'
df = pd.read_csv(train_url)

# 2. Preprocessing
print("Preprocessing...")
# Convert booleans to int
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

# One-hot encode (handling categorical)
categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Handle Missing values (Median/Mode as per the script)
df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True))

# 3. Split Data
X = df_encoded.drop(['accident_risk', 'id'], axis=1) # Drop ID as it's not a feature
y = df_encoded['accident_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2025, stratify=pd.qcut(y, q=5) # Stratified by risk buckets
)

# 4. Train Optimized XGBoost (Using best params from the randomized search)
print("Training Model...")
best_params = {
    'colsample_bytree': 0.94, 
    'gamma': 0.025, 
    'learning_rate': 0.068, 
    'max_depth': 8, 
    'n_estimators': 195, 
    'subsample': 0.735,
    'random_state': 2025
}
model = xgb.XGBRegressor(**best_params)
model.fit(X_train, y_train)

# 5. Calculate Operational Metrics (Simulating Business Logic)
# Scenario: "High Risk" is defined as a predicted probability > 0.6
threshold = 0.6
y_pred = model.predict(X_test)

# Binarize for classification metrics
y_test_binary = (y_test > threshold).astype(int)
y_pred_binary = (y_pred > threshold).astype(int)

# High-Risk Area Precision (Precision for the positive class)
high_risk_precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)

# Severe Accident Recall (Recall for the positive class)
severe_accident_recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)

print(f"Operational Metrics Calculated: Precision={high_risk_precision:.2f}, Recall={severe_accident_recall:.2f}")

# 6. Save Everything
artifacts = {
    'model': model,
    'feature_names': X_train.columns.tolist(),
    'categorical_cols': categorical_cols.tolist(),
    'bool_cols': bool_cols.tolist(),
    'metrics': {
        'precision': high_risk_precision,
        'recall': severe_accident_recall,
        'test_rmse': np.sqrt(((y_test - y_pred) ** 2).mean())
    }
}

joblib.dump(artifacts, 'accident_model_artifacts.joblib')
print("✅ Artifacts saved to 'accident_model_artifacts.joblib'")
