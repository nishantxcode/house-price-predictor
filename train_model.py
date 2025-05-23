# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load data
df = pd.read_csv('data/Bengaluru_House_Data.csv')

# Handle total_sqft
def convert_sqft_to_num(x):
    try:
        if '-' in x:
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None



# Basic cleaning
df = df.dropna()
df = df[df['total_sqft'].apply(lambda x: str(x).replace('.', '').replace(' ', '').replace('-', '').isdigit())]
df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
df = df.dropna(subset=['total_sqft'])


# BHK extraction from 'size'
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else 0)

# Encode 'location'
df['location'] = df['location'].apply(lambda x: x.strip())
label_enc = LabelEncoder()
df['location'] = label_enc.fit_transform(df['location'])

# Filter columns
X = df[['location', 'total_sqft', 'bath', 'bhk']]
y = df['price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!\nRMSE: {rmse:.2f}\nR²: {r2:.2f}")

# Save model and encoder
os.makedirs('predictor/ml', exist_ok=True)
joblib.dump(model, 'predictor/ml/model.pkl')
joblib.dump(label_enc, 'predictor/ml/location_encoder.pkl')

