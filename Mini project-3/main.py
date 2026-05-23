import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib

def generate_mock_data(n_samples=10000):
    """Generate synthetic household power consumption data for demonstration."""
    print("Generating synthetic data...")
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    
    # Simulate base power consumption with daily and seasonal patterns
    hour = dates.hour
    month = dates.month
    
    # Global active power simulation
    global_active_power = 1.0 + np.sin(hour * np.pi / 12) + np.cos(month * np.pi / 6) * 0.5
    global_active_power += np.random.normal(0, 0.2, n_samples)
    global_active_power = np.clip(global_active_power, 0.2, 5.0)
    
    data = {
        'Date': dates.strftime('%d/%m/%Y'),
        'Time': dates.strftime('%H:%M:%S'),
        'Global_active_power': global_active_power,
        'Global_reactive_power': global_active_power * 0.1 + np.random.normal(0, 0.05, n_samples),
        'Voltage': np.random.normal(240, 1.5, n_samples),
        'Global_intensity': global_active_power * 4.2 + np.random.normal(0, 0.5, n_samples),
        'Sub_metering_1': np.random.exponential(0.5, n_samples),
        'Sub_metering_2': np.random.exponential(0.6, n_samples),
        'Sub_metering_3': global_active_power * 5 + np.random.normal(0, 1, n_samples)
    }
    return pd.DataFrame(data)

def preprocess_data(df):
    """Clean, parse dates, and engineer features."""
    print("Preprocessing data...")
    # Convert '?' to NaN and drop
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Parse DateTime
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    
    # Feature Engineering
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Month'] = df['Datetime'].dt.month
    df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Drop non-numeric for modeling
    df_numeric = df.drop(columns=['Date', 'Time', 'Datetime'])
    return df_numeric

def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"--- {name} ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}\n")
    return rmse, mae, r2

def main():
    # 1. Data Understanding & Exploration
    df = generate_mock_data()
    
    # 2. Data Preprocessing & Feature Engineering
    df_processed = preprocess_data(df)
    
    # Define Target and Features
    # Predict Global_active_power based on time and other environmental/metering factors
    target = 'Global_active_power'
    X = df_processed.drop(columns=[target])
    y = df_processed[target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Model Selection and Training
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42),
        "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    }
    
    results = {}
    best_r2 = -float('inf')
    best_model = None
    best_model_name = ""
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        rmse, mae, r2 = evaluate_model(name, y_test, predictions)
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name
            
    print(f"Best Model: {best_model_name} with R2 = {best_r2:.4f}")
    
    # Save the best model
    joblib.dump(best_model, 'power_pulse_model.pkl')
    
    # 4. Visualizations
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    # Feature Importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances ({best_model_name})")
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png')
        print("Saved feature_importance.png")
        
    # Actual vs Predicted Plot (Sample 100 points)
    best_predictions = best_model.predict(X_test_scaled)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values[:100], label='Actual Power', marker='o')
    plt.plot(best_predictions[:100], label='Predicted Power', marker='x')
    plt.title("Actual vs Predicted Household Energy Usage (First 100 Test Samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Global Active Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/actual_vs_predicted.png')
    print("Saved actual_vs_predicted.png")
    
if __name__ == "__main__":
    main()
