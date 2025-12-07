import os
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Try to locate dataset in a few likely places
candidates = [
    Path("./data/red_wine_quality_Final.csv"),
    Path("../data/red_wine_quality_Final.csv"),
    Path("C:/Users/user/Desktop/VS_studio_projects/data/red_wine_quality_Final.csv")
]

csv_path = None
for p in candidates:
    if p.exists():
        csv_path = p
        break

if csv_path is None:
    raise FileNotFoundError("Could not find red_wine_quality_Final.csv. Tried: " + ", ".join(str(p) for p in candidates))

print(f"Loading data from: {csv_path}")

df = pd.read_csv(csv_path, encoding='utf-8', decimal='.', delimiter=',')

# Load feature list if available
feature_file = Path('feature_names.pkl')
if feature_file.exists():
    feature_names = joblib.load(feature_file)
    print("Loaded feature list from feature_names.pkl")
else:
    # fallback: use all numeric except quality
    feature_names = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'quality']
    print("feature_names.pkl not found — using numeric columns:", feature_names)

X = df[feature_names].copy()
y = df['quality'].astype(float).copy()

# Simple imputation: fill numeric columns with medians
for c in X.columns:
    if X[c].isna().any():
        X[c] = X[c].fillna(X[c].median())

# Save medians for possible future imputations
medians = X.median()
joblib.dump(medians, 'training_medians.pkl')

# Compute training target range
t_min = float(y.min())
t_max = float(y.max())

# Rescale target to desired range 3-8 (linear min-max)
out_min, out_max = 3.0, 8.0
if t_max == t_min:
    raise ValueError('Target has zero range')
y_scaled = (y - t_min) / (t_max - t_min) * (out_max - out_min) + out_min

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train MLP on scaled target
mlp = MLPRegressor(hidden_layer_sizes=(64,32), activation='relu', solver='adam',
                   alpha=0.0001, learning_rate='adaptive', max_iter=1000, random_state=42)
print('Training MLP on rescaled target (3-8)...')
mlp.fit(X_train_scaled, y_train)

# Evaluate
y_pred_train = mlp.predict(X_train_scaled)
y_pred_test = mlp.predict(X_test_scaled)

print('Train R2:', r2_score(y_train, y_pred_train))
print('Test R2: ', r2_score(y_test, y_pred_test))
print('Test MAE:', mean_absolute_error(y_test, y_pred_test))

# Save model, scaler and target_range
joblib.dump(mlp, 'wine_model.pkl')
joblib.dump(scaler, 'wine_scaler.pkl')
with open('target_range.json', 'w') as f:
    json.dump({'min': t_min, 'max': t_max}, f)

print('Saved wine_model.pkl, wine_scaler.pkl, and target_range.json')
