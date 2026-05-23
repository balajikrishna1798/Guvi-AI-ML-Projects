import time
import random
from datetime import datetime
from database import create_connection, insert_ride_data
# import selenium and beautifulsoup for actual scraping
# from selenium import webdriver
# from bs4 import BeautifulSoup

def extract_uber_data(pickup, destination):
    """
    Simulates extracting real-time Uber data.
    In a real-world scenario, this would use Selenium to automate a browser,
    input the pickup and destination into the Uber price estimator, and parse the HTML with BeautifulSoup.
    Since scraping Uber is heavily rate-limited and often blocked, this provides mock data for demonstration.
    """
    print(f"Extracting data for ride from {pickup} to {destination}...")
    
    # Simulate network delay
    time.sleep(2)
    
    now = datetime.now()
    
    # Mock data generation based on time of day (simulate surge pricing)
    hour = now.hour
    base_fare_go = random.uniform(10.0, 20.0)
    base_fare_sedan = base_fare_go * 1.5
    
    # Surge between 8-10 AM and 5-7 PM
    surge = 1.5 if (8 <= hour <= 10) or (17 <= hour <= 19) else 1.0
    
    data = [
        (pickup, destination, now.strftime('%Y-%m-%d %H:%M:%S'), 'Uber Go', round(base_fare_go * surge, 2), random.randint(2, 10), random.randint(15, 45)),
        (pickup, destination, now.strftime('%Y-%m-%d %H:%M:%S'), 'Go Sedan', round(base_fare_sedan * surge, 2), random.randint(3, 15), random.randint(15, 45))
    ]
    
    return data

def run_extraction_job():
    pickup = "Central Park, NY"
    destination = "Times Square, NY"
    
    conn = create_connection()
    if conn:
        data = extract_uber_data(pickup, destination)
        insert_ride_data(conn, data)
        print("Data extracted and stored in database.")
        conn.close()
    else:
        print("Failed to connect to DB.")

if __name__ == "__main__":
    run_extraction_job()
