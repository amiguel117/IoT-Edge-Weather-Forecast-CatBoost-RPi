# 🌦️ Intelligent IoT-Based 48-Hour Weather Forecast System

This project implements a real-time edge-AI system for forecasting weather using IoT and machine learning on a Raspberry Pi. It combines OpenWeather API data with on-device inference using a CatBoost regression model, visualized via Node-RED, and secured using MQTT over TLS.

---

## 🔧 Features

- 🌐 Real-time data from OpenWeather API
- 🤖 ML-powered 48-hour temperature forecast using CatBoost
- 🧠 Edge-side prediction hosted on Raspberry Pi
- 📊 Dashboard visualization with Node-RED
- 📡 Secure MQTT message publishing (TLS)
- ⚠️ Alarm generation for barometric pressure drops

---

## 📁 Project Structure

- `data/`: Sample output weather data in CSV
- `models/`: Pre-trained CatBoost model (`.cbm`)
- `notebooks/`: Jupyter training notebook
- `node-red/`: Screenshot & Node-RED flow file
- `app/`: FastAPI app (`predict.py`) for temperature forecast
- `report/`: Academic PDF report (Unitec)
- `README.md`: This document

---

## 🛠️ Tools & Libraries

- **Raspberry Pi 4B** – Edge device used for deployment
- **Python 3.11** – Backend programming language
- **CatBoost** – ML model for temperature regression forecasting
- **FastAPI** – Lightweight REST API for model inference
- **Node-RED v3.1.1** – Flow-based development tool for dataflow and dashboard
- **InfluxDB 2.7** – Time-series database for storing environmental readings
- **MQTT (EMQX Broker)** – Lightweight messaging protocol with TLS encryption
- **OpenWeather API** – Source for public weather data

---

## 🚀 Run the Prediction API (FastAPI)

```bash
uvicorn app.predict:app --reload --port 8000

GET http://localhost:8000/predict?hours_to_forecast=48

