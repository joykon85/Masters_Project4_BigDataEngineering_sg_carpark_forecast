# Masters_Project4_BigDataEngineering_sg_carpark_forecast

# NUS ISS Data Science Masters Group Project 4 - Big Data Engineering.

# Real-Time Carpark Forecasting Pipeline on GCP

## Project Overview

This project implements an end-to-end **real-time data and machine learning pipeline on Google Cloud Platform (GCP)** using Singapore **LTA carpark availability data**.

The system ingests real-time data, stores it in a cloud data warehouse, runs forecasting models using multiple ML frameworks, and serves predictions through an interactive dashboard.

---

## Architecture & Workflow

### 1. Real-time Data Ingestion
- Pulls live carpark availability data from the **LTA API**
- Streams data using **Pub/Sub**
- Processes and transforms data with **Dataflow (Apache Beam)**

### 2. Data Storage
- Stores raw and processed data in **BigQuery** as the central data warehouse

### 3. ML & Inference Pipelines
- Builds and evaluates forecasting models using:
  - **LSTM (Long Short-Term Memory)** for time-series forecasting
  - **Gradient-Boosted Trees (GBT) Regressor** as a classical ML baseline
- Models are implemented and orchestrated using:
  - **PySpark on Dataproc** (distributed batch processing)
  - **Kubeflow on Vertex AI** (pipeline orchestration and experimentation)
- Generates carpark availability forecasts

### 4. Serving & Analytics
- Writes forecast outputs back to **BigQuery**
- Visualizes real-time and forecasted data using **Looker Studio**

---

## Tech Stack

- **Cloud Platform:** Google Cloud Platform (GCP)
- **Streaming & Processing:** Pub/Sub, Dataflow (Apache Beam)
- **Data Warehouse:** BigQuery
- **ML & Pipelines:**  
  - LSTM (Deep Learning)  
  - Gradient-Boosted Trees Regressor  
  - PySpark (Dataproc)  
  - Kubeflow (Vertex AI)
- **Visualization:** Looker Studio
- **Language:** Python

---

## Key Learning Outcomes

- Designed a production-style **real-time data pipeline**
- Implemented and compared **deep learning (LSTM)** and **tree-based (GBT) forecasting models**
- Integrated **streaming, batch, and ML workflows** on GCP
- Deployed inference outputs to a **cloud data warehouse and analytics dashboard**

---

## Notes

Note: The real-time data ingestion component and Looker Studio dashboard were implemented by a collaborator and are not included in this repository.
