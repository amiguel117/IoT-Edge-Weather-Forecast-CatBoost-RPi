# 📁 Models Folder

This folder contains the trained CatBoost regression model used for 48-hour temperature forecasting.

### File(s):
- `catboost_trained_model.cbm`: Serialized CatBoost model trained on 7-day rolling window weather data.

### Details:
- Forecasts hourly temperature for the next 48 hours.
- Trained using Optuna for hyperparameter optimization.
- Compatible with the Python FastAPI app in `/app/predict.py`.

> For inference, the model expects specific preprocessed features in the same order used during training.
