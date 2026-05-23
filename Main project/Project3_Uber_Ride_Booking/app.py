import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import os

# Try loading the model
try:
    model = joblib.load('fare_prediction_model.pkl')
except FileNotFoundError:
    model = None

def predict_best_time(pickup_date):
    """Predict fares for the next 24 hours based on the selected date."""
    if model is None:
        return None
        
    predictions = []
    day_of_week = pickup_date.weekday()
    
    for hour in range(24):
        # Create input array matching the features the model was trained on
        X_input = pd.DataFrame({'hour': [hour], 'day_of_week': [day_of_week]})
        predicted_fare = model.predict(X_input)[0]
        
        time_str = f"{hour:02d}:00"
        predictions.append({'Time': time_str, 'Predicted Fare ($)': round(predicted_fare, 2)})
        
    return pd.DataFrame(predictions)

def main():
    st.set_page_config(page_title="Uber Booking Optimizer", layout="wide")
    
    st.title("🚖 Optimized Uber Ride Booking System")
    st.write("Find the best time to book your Uber ride for the lowest fare based on historical patterns.")
    
    if model is None:
        st.warning("⚠️ Prediction model not found. Please run `python model.py` to train and generate the model.")
        
    st.sidebar.header("Ride Details")
    pickup = st.sidebar.text_input("Pickup Location", "Central Park, NY")
    destination = st.sidebar.text_input("Destination", "Times Square, NY")
    pickup_date = st.sidebar.date_input("Planned Pickup Date", datetime.today())
    
    if st.sidebar.button("Find Best Time to Book"):
        st.subheader(f"Fare Predictions for {pickup_date.strftime('%Y-%m-%d')}")
        st.write(f"Route: **{pickup}** to **{destination}**")
        
        with st.spinner("Analyzing historical data and extracting current trends..."):
            df_predictions = predict_best_time(pickup_date)
            
        if df_predictions is not None:
            # Find the best time
            min_fare_idx = df_predictions['Predicted Fare ($)'].idxmin()
            best_time = df_predictions.loc[min_fare_idx, 'Time']
            best_fare = df_predictions.loc[min_fare_idx, 'Predicted Fare ($)']
            
            st.success(f"**Recommended Booking Time:** {best_time} with an estimated fare of **${best_fare}**")
            
            # Line chart for visualization
            st.line_chart(df_predictions.set_index('Time'))
            
            with st.expander("View Detailed Hourly Predictions"):
                st.dataframe(df_predictions, use_container_width=True)
        else:
            st.error("Cannot generate predictions. Model is missing.")

if __name__ == "__main__":
    main()
