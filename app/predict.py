import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from catboost import CatBoostRegressor
from influxdb_client import InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta # Added timedelta

# --- Configuration ---
INFLUX_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
INFLUX_TOKEN = "93376XfLX3ar22dS4ofolYBR5KawK3za2D0UoIxEFCj_RShoHyH94IwtKoYj7a2tyPR9ycpwBnSuJ86SHOs7QQ==" # Your actual token
INFLUX_ORG = "Paul"
INFLUX_BUCKET = "weather_live"
MEASUREMENT_NAME = "auckland"

MODEL_FILENAME = "catboost_trained_model.cbm" # Your Colab-trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

# Define the features your Colab model expects, IN THE SAME ORDER AND WITH EXACT NAMES AS TRAINING X
# Based on IoT_Weather_Predict_CatBoost(1).ipynb (CELL 5 X.dtypes output)
# 'temp_C' was the target (y) and is NOT an input feature here.
EXPECTED_FEATURES_ORDER = [
    'timestamp',         # Original epoch seconds from 'dt'
    'pressure_hPa',
    'humidity',          # Note: not humidity_percent
    'temp_min_C',
    'temp_max_C',
    'wind_m_s',          # Note: not wind_speed_m_s
    'wind_deg',
    'clouds_percent',    # Note: not clouds_perc
    'year',
    'month',
    'day',
    'hour',
    'dayofweek',
    'dayofyear'
]

CATEGORICAL_FEATURE_NAMES = [] # Based on Colab notebook, X had no remaining categoricals after processing

# Raw field names AS THEY EXIST IN INFLUXDB needed to create EXPECTED_FEATURES_ORDER
RAW_INFLUXDB_FIELDS_FOR_QUERY = list(set([
    'timestamp', 'pressure_hPa', 'humidity', 'temp_min_C', 'temp_max_C',
    'wind_m_s', 'wind_deg', 'clouds_percent', 'grnd_level', 
    'sea_level', 'wind_gust', 'temp_C', 
    'weather_main', 'weather_description', 'weather_icon' 
]))

model = None
influx_client = None
query_api = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, influx_client, query_api
    print("Lifespan: Application startup...")
    try:
        print(f"Lifespan: Loading model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print(f"Lifespan: ERROR - Model file not found at {MODEL_PATH}")
            model = None
        else:
            model = CatBoostRegressor() 
            model.load_model(MODEL_PATH)
            print("Lifespan: Model loaded successfully.")
    except Exception as e:
        print(f"Lifespan: Error loading model: {e}")
        model = None
    try:
        print(f"Lifespan: Initializing InfluxDB client for URL: {INFLUX_URL}, Org: {INFLUX_ORG}, Bucket: {INFLUX_BUCKET}")
        influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = influx_client.query_api()
        print("Lifespan: InfluxDB client initialized successfully.")
    except Exception as e:
        print(f"Lifespan: Error initializing InfluxDB client: {e}")
        influx_client = None
        query_api = None
    yield
    print("Lifespan: Application shutting down...")
    if influx_client:
        influx_client.close()
        print("Lifespan: InfluxDB client closed.")
    model = None
    print("Lifespan: Model unloaded.")

app = FastAPI(lifespan=lifespan)

def get_initial_features_from_db() -> pd.DataFrame:
    if not query_api:
        print("get_initial_features_from_db: InfluxDB client not available.")
        raise HTTPException(status_code=503, detail="InfluxDB client not available.")

    unique_raw_fields = list(set(RAW_INFLUXDB_FIELDS_FOR_QUERY))
    if not unique_raw_fields:
        raise HTTPException(status_code=500, detail="Internal server error: Raw fields for query not defined.")
    flux_field_set_str = "[" + ", ".join([f'"{field}"' for field in unique_raw_fields]) + "]"

    flux_query = f'''
        from(bucket: "{INFLUX_BUCKET}")
          |> range(start: -1d) 
          |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENT_NAME}")
          |> filter(fn: (r) => contains(value: r["_field"], set: {flux_field_set_str}))
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"], desc: true) 
          |> limit(n: 1) 
          |> rename(columns: {{_time: "record_influx_time"}}) 
    '''
    print(f"get_initial_features_from_db: Querying InfluxDB for latest data...")
    try:
        tables = query_api.query(query=flux_query)
        if not tables or not tables[0].records:
            raise HTTPException(status_code=404, detail="No recent comprehensive weather data found in InfluxDB for initial features.")
        latest_data_dict = tables[0].records[0].values
        print(f"get_initial_features_from_db: Latest raw data fetched: {latest_data_dict}")
    except InfluxDBError as e:
        print(f"get_initial_features_from_db: InfluxDB query error: {e}")
        raise HTTPException(status_code=500, detail=f"InfluxDB query error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query InfluxDB for latest data: {e}")

    # Prepare a dictionary that will be used to create the first row of features
    initial_feature_values = {}
    
    # Get initial epoch_seconds from 'timestamp' field (which should be epoch ms from 'dt')
    epoch_ms_from_field = latest_data_dict.get('timestamp')
    if epoch_ms_from_field is None:
        influx_time_obj = latest_data_dict.get('record_influx_time')
        if isinstance(influx_time_obj, str):
            influx_time_obj = datetime.fromisoformat(influx_time_obj.replace("Z", "+00:00"))
        if not isinstance(influx_time_obj, datetime):
            raise HTTPException(status_code=500, detail="Cannot determine initial timestamp.")
        current_dt_obj_utc = influx_time_obj.astimezone(timezone.utc) if influx_time_obj.tzinfo else influx_time_obj.replace(tzinfo=timezone.utc)
        initial_epoch_seconds = int(current_dt_obj_utc.timestamp())
        print(f"Warning: Using InfluxDB record time for initial date features. Epoch_seconds: {initial_epoch_seconds}")
    else:
        try:
            initial_epoch_seconds = int(float(epoch_ms_from_field)) // 1000
            print(f"Using 'timestamp' field for initial date features. Original ms: {int(float(epoch_ms_from_field))}, converted_seconds: {initial_epoch_seconds}")
        except ValueError:
            raise HTTPException(status_code=500, detail="Invalid format for 'timestamp' field from InfluxDB.")

    # Populate initial_feature_values based on EXPECTED_FEATURES_ORDER
    # These are the non-time-varying features (pressure, humidity etc.) that will persist
    # and the first value for the 'timestamp' feature field.
    for feature_name in EXPECTED_FEATURES_ORDER:
        if feature_name in ['year', 'month', 'day', 'hour', 'dayofweek', 'dayofyear']:
            continue # These will be set iteratively

        # Map model feature names to actual InfluxDB field names if they differ
        influx_field_name = feature_name 
        # No specific mappings needed here if EXPECTED_FEATURES_ORDER uses names as in InfluxDB
        # (e.g., 'humidity' not 'humidity_percent')

        value = latest_data_dict.get(influx_field_name)

        if feature_name == 'timestamp':
            initial_feature_values[feature_name] = initial_epoch_seconds
        elif feature_name in CATEGORICAL_FEATURE_NAMES: # Should be empty for this model
            initial_feature_values[feature_name] = str(value) if value is not None else "NA_placeholder"
        elif value is not None:
            try:
                initial_feature_values[feature_name] = float(value)
            except (ValueError, TypeError):
                initial_feature_values[feature_name] = np.nan
        else:
            initial_feature_values[feature_name] = np.nan # Use NaN for missing numeric features

    # Create a DataFrame from these initial values (will be updated in the loop)
    # Ensure all EXPECTED_FEATURES_ORDER columns are present, even if initially NaN for date parts
    df_data = {col: [initial_feature_values.get(col)] for col in EXPECTED_FEATURES_ORDER}
    features_df = pd.DataFrame(df_data, columns=EXPECTED_FEATURES_ORDER)
    
    print(f"get_initial_features_from_db: Initial base features prepared.")
    return features_df


@app.get("/predict")
async def predict_weather(hours_to_forecast: int = Query(48, description="Number of hours to forecast.", ge=1)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")

    try:
        print("/predict: Fetching initial weather data and engineering base features...")
        current_features_df = get_initial_features_from_db()
        
        if current_features_df.empty:
            raise HTTPException(status_code=500, detail="Failed to generate initial features for prediction.")

        all_forecast_values = []
        
        current_epoch_seconds = int(current_features_df['timestamp'].iloc[0])

        persisted_features = {}
        # List of features that are assumed to persist from the last observation
        # These should match names in EXPECTED_FEATURES_ORDER and be present in latest_data_dict
        non_date_features_to_persist = [
            'pressure_hPa', 'humidity', 'temp_min_C', 'temp_max_C', 
            'wind_m_s', 'wind_deg', 'clouds_percent'
            # Add 'grnd_level', 'sea_level', 'wind_gust' if they are in EXPECTED_FEATURES_ORDER
            # and you want to persist them.
        ]
        for f_name in non_date_features_to_persist:
            if f_name in current_features_df.columns:
                persisted_features[f_name] = current_features_df[f_name].iloc[0]
            else: 
                persisted_features[f_name] = np.nan # Default to NaN if not found initially
                print(f"Warning: Persisted feature '{f_name}' not found in initial DataFrame, will use NaN if expected by model.")


        print(f"/predict: Starting iterative forecast for {hours_to_forecast} hours...")
        # Make a copy for iteration to avoid SettingWithCopyWarning
        iterative_features_df = current_features_df.copy()

        for i in range(hours_to_forecast):
            # Update time-based features for the current forecast step
            current_dt_obj_utc = datetime.fromtimestamp(current_epoch_seconds, timezone.utc)
            iterative_features_df.loc[0, 'year'] = current_dt_obj_utc.year
            iterative_features_df.loc[0, 'month'] = current_dt_obj_utc.month
            iterative_features_df.loc[0, 'day'] = current_dt_obj_utc.day
            iterative_features_df.loc[0, 'hour'] = current_dt_obj_utc.hour
            iterative_features_df.loc[0, 'dayofweek'] = current_dt_obj_utc.weekday()
            iterative_features_df.loc[0, 'dayofyear'] = current_dt_obj_utc.timetuple().tm_yday
            iterative_features_df.loc[0, 'timestamp'] = current_epoch_seconds

            # Ensure persisted features are set for the current prediction step
            for pf_name, pf_value in persisted_features.items():
                 if pf_name in iterative_features_df.columns:
                    iterative_features_df.loc[0, pf_name] = pf_value
            
            # Ensure all columns are in the expected order for the model
            X_predict_step_df = iterative_features_df[EXPECTED_FEATURES_ORDER]
            
            # Predict one step ahead (e.g., temp_C for this current_epoch_seconds)
            prediction_result = model.predict(X_predict_step_df)
            
            if isinstance(prediction_result, np.ndarray):
                predicted_value_for_step = float(prediction_result.flatten()[0])
            else:
                predicted_value_for_step = float(prediction_result) 

            all_forecast_values.append(predicted_value_for_step)
            
            # Prepare for the next iteration: Advance time by 1 hour (3600 seconds)
            current_epoch_seconds += 3600
            
            # Naive: Use the predicted temp_C as the 'temp_C' for the next step's persisted features
            # This is a very simple way to make the forecast iterative on temperature.
            # The model itself was NOT trained with 'temp_C' as an input feature,
            # so this line is only relevant if 'temp_C' was part of persisted_features
            # (which it isn't by default in non_date_features_to_persist).
            # If 'temp_min_C' and 'temp_max_C' are to be updated, they'd need their own logic.
            # For now, we are only updating time features and keeping others static.

        print(f"/predict: Iterative prediction complete. Generated {len(all_forecast_values)} values.")
        return {"forecast": all_forecast_values}

    except HTTPException as e:
        print(f"/predict: HTTPException occurred: {e.detail}")
        raise e
    except Exception as e:
        print(f"/predict: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server for predict:app (Colab model compatible - Iterative Forecast)...")
    print(f"Model expected at: {MODEL_PATH}")
    print(f"InfluxDB Target: Bucket='{INFLUX_BUCKET}', Measurement='{MEASUREMENT_NAME}'")
    print(f"Model expects features in this order: {EXPECTED_FEATURES_ORDER}")
    print(f"Raw fields queried from InfluxDB: {RAW_INFLUXDB_FIELDS_FOR_QUERY}")
    
    uvicorn.run("predict:app", host="0.0.0.0", port=8000, reload=True)
