import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def generate_mock_historical_data():
    """Generates synthetic historical data for training the model."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    data = []
    
    for dt in dates:
        hour = dt.hour
        day_of_week = dt.dayofweek
        
        # Base fare pattern
        base_fare = 15.0
        
        # Add rush hour effect
        if (8 <= hour <= 10) or (17 <= hour <= 19):
            base_fare += 10.0
            
        # Add weekend effect
        if day_of_week >= 5:
            base_fare += 5.0
            
        # Add noise
        fare = base_fare + np.random.normal(0, 2.0)
        
        data.append({
            'hour': hour,
            'day_of_week': day_of_week,
            'fare': max(5.0, fare) # Minimum fare
        })
        
    return pd.DataFrame(data)

def train_model():
    print("Loading historical data...")
    df = generate_mock_historical_data()
    
    X = df[['hour', 'day_of_week']]
    y = df['fare']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Evaluation:\nMean Squared Error: {mse:.4f}\nR2 Score: {r2:.4f}")
    
    joblib.dump(model, 'fare_prediction_model.pkl')
    print("Model saved as 'fare_prediction_model.pkl'")

if __name__ == "__main__":
    train_model()
