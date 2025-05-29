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

## 🚀 Run the Prediction API (FastAPI)

```bash
uvicorn app.predict:app --reload --port 8000

GET http://localhost:8000/predict?hours_to_forecast=48

