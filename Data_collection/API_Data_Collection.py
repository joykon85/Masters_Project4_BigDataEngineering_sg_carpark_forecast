
import requests
import schedule
import time
import os
from pymongo import MongoClient
from datetime import datetime

# Environment variables for API keys (Ensure you set them in your environment)
LTA_API_KEY = os.getenv("LTA_API_KEY", "7GQ4fcMqRTuEm4Tb681Y6A==")  # Replace if not using environment variable

# API URLs
API_ENDPOINTS = {
    "carpark_availability": "https://datamall2.mytransport.sg/ltaodataservice/CarParkAvailabilityv2",
    "traffic_incidents": "https://datamall2.mytransport.sg/ltaodataservice/TrafficIncidents",
    "weather_forecast": "https://api-open.data.gov.sg/v2/real-time/api/twenty-four-hr-forecast",
    "weather_forecast_2hrs": "https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast"
}

# Headers for LTA APIs
HEADERS_LTA = {
    "AccountKey": LTA_API_KEY,
    "accept": "application/json"
}

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "lta_data"

# Connect to MongoDB (Use a single connection for efficiency)
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

# Function to fetch data from an API
def fetch_data(api_name, url, headers=None):
    try:
        response = requests.get(url, headers=headers, timeout=20)
        #response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get("value", data)  # LTA APIs have "value", weather API returns full JSON
        else:
            print(f"[ERROR] Failed to fetch data from {api_name}: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"[ERROR] Exception while fetching {api_name}: {e}")
        return None

# Function to store data in MongoDB and print total count
def store_data(collection_name, data):
    if not data:
        print(f"[WARNING] No data received for {collection_name}, skipping storage.")
        return
    
    collection = db[collection_name]
    timestamp = datetime.now()
    
    # Add timestamp to each record and insert into MongoDB
    if isinstance(data, list):  # LTA APIs return lists
        for item in data:
            item["timestamp"] = timestamp
        collection.insert_many(data)
    else:  # Weather API returns a dictionary
        data["timestamp"] = timestamp
        collection.insert_one(data)

    # Get and print total count after insertion
    total_records = collection.count_documents({})
    print(f"[INFO] Stored {len(data) if isinstance(data, list) else 1} records in {collection_name} at {timestamp}")
    print(f"[INFO] Total records in {collection_name}: {total_records}")

# Function to fetch and store data
def fetch_and_store_data():
    print("\n[INFO] Fetching latest data...")

    # Fetch and store data for each API
    carpark_data = fetch_data("Carpark Availability", API_ENDPOINTS["carpark_availability"], HEADERS_LTA)
    store_data("carpark_availability", carpark_data)

    traffic_data = fetch_data("Traffic Incidents", API_ENDPOINTS["traffic_incidents"], HEADERS_LTA)
    store_data("traffic_incidents", traffic_data)

    weather_data = fetch_data("Weather Forecast", API_ENDPOINTS["weather_forecast"])
    store_data("weather_forecast", weather_data)

    weather_data_2hr = fetch_data("Weather Forecast 2hrs", API_ENDPOINTS["weather_forecast_2hrs"])
    store_data("weather_forecast_2hrs", weather_data_2hr)

    print("[INFO] Data fetch complete.")

# Schedule the job every 5 minutes throughout the entire day (00:00 to 23:55)
for hour in range(24):  # Loop from 00 to 23 hours
    for minute in range(0, 60, 5):  # Loop from 0 to 55 minutes with a step of 5
        time_str = f"{hour:02d}:{minute:02d}"  # Format the time as HH:MM
        schedule.every().day.at(time_str).do(fetch_and_store_data)

# Main function to run the scheduled tasks
if __name__ == "__main__":
    while True:
        schedule.run_pending()  # Run any pending scheduled tasks
        time.sleep(1)  # Wait a second to avoid high CPU usage


