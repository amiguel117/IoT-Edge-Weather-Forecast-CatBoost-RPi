# 📁 App Folder

This folder contains the FastAPI application used to run inference on the trained CatBoost model and provide a RESTful API endpoint.

### File(s):
- `predict.py`: Main inference API script.

### Endpoint:
```http
GET /predict?hours_to_forecast=48
