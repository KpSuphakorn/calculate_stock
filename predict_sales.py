import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import gc
from datetime import date, timedelta, datetime
import warnings
import sys # เพิ่ม sys สำหรับ exit

warnings.filterwarnings('ignore')

# --- Configuration (แบบเดิม) ---
TRAIN_FILE = 'train.csv'
MODEL_FILE = 'lgbm_model.joblib' # <<< ตรวจสอบชื่อไฟล์โมเดล
OUTPUT_PREDICTIONS_FILE = 'sales_predictions.csv' # <<< ชื่อไฟล์ที่จะบันทึกผล Predict

# --- Simulation Dates (แบบเดิม) ---
sim_today_dt = date(2014, 12, 31)
prediction_start_dt = sim_today_dt + timedelta(days=1)
prediction_end_dt = date(2015, 1, 31) # <<< กำหนดวันสิ้นสุดให้ชัดเจน

sim_today_ts = pd.Timestamp(sim_today_dt)
prediction_start_ts = pd.Timestamp(prediction_start_dt)
prediction_end_ts = pd.Timestamp(prediction_end_dt)

# --- Feature Engineering Params (แบบเดิม) ---
MAX_LAG = 364
SHIFT_BASE_ROLLING = 7
MAX_ROLL_WINDOW = 90 # ไม่ได้ใช้ในโค้ดแก้ไขนี้ แต่คงไว้เผื่ออ้างอิง

# --- ฟังก์ชันช่วยเหลือ ---
def create_features(dataframe):
    """สร้าง Features"""
    print("Applying feature engineering...")
    df_feat = dataframe.copy()
    df_feat['year'] = df_feat['date'].dt.year
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['day'] = df_feat['date'].dt.day
    df_feat['dayofweek'] = df_feat['date'].dt.dayofweek
    df_feat['dayofyear'] = df_feat['date'].dt.dayofyear
    df_feat['weekofyear'] = df_feat['date'].dt.isocalendar().week.astype(int)
    df_feat['quarter'] = df_feat['date'].dt.quarter
    df_feat['is_weekend'] = (df_feat['dayofweek'] >= 5).astype(int)
    print("  - Date features created.")

    df_feat.sort_values(['store', 'item', 'date'], inplace=True)

    # --- Lag Features (เหมือนเดิม) ---
    lags = [7, 14, 21, 28, 35, 60, 91, 182, 364]
    group_cols = ['store', 'item'] # กำหนดตัวแปรก่อนใช้ง่ายกว่า
    sales_col = 'sales' # กำหนดตัวแปรก่อนใช้ง่ายกว่า
    for lag in lags:
        df_feat[f'{sales_col}_lag_{lag}'] = df_feat.groupby(group_cols, observed=True)[sales_col].shift(lag)
    print(f"  - Lag features created for lags: {lags}")

    # --- Rolling Window Features (แก้ไข) ---
    shift_base = SHIFT_BASE_ROLLING
    windows = [7, 14, 28, 60, 90]
    # Groupby ครั้งเดียว แล้วใช้ transform กับ lambda function ที่มีการ shift และ rolling ภายใน
    for window in windows:
        min_p = max(1, window // 4)
        df_feat[f'{sales_col}_roll_mean_{shift_base}_{window}'] = df_feat.groupby(group_cols, observed=True)[sales_col].transform(
             lambda x: x.shift(shift_base).rolling(window, min_periods=min_p).mean()
         )
        df_feat[f'{sales_col}_roll_std_{shift_base}_{window}'] = df_feat.groupby(group_cols, observed=True)[sales_col].transform(
             lambda x: x.shift(shift_base).rolling(window, min_periods=min_p).std()
         )
    print(f"  - Rolling window features created for windows: {windows} (based on lag {shift_base})")
    # --- สิ้นสุดการแก้ไข Rolling Window ---

    print("Feature engineering complete.")
    return df_feat

# --- โค้ดหลัก (ทำงานต่อลงมาเลย) ---
print("--- Starting Sales Prediction Process ---")

# 1. Load Model
print(f"Loading trained model from {MODEL_FILE}...")
model = None
try:
    model = joblib.load(MODEL_FILE)
    print("Model loaded successfully.")
    print(f"DEBUG: Type of loaded object: {type(model)}")
    if not hasattr(model, 'predict'):
        print("Error: Loaded object does not have a 'predict' method.")
        sys.exit(1)
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_FILE}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# 2. Load Historical Data
history_needed_start_dt = prediction_start_dt - timedelta(days=MAX_LAG)
history_start_ts = pd.Timestamp(history_needed_start_dt)
history_end_ts = sim_today_ts

print(f"Loading historical data from {history_needed_start_dt} to {sim_today_dt}...")
try:
    df_history_full = pd.read_csv(TRAIN_FILE, parse_dates=['date'])
    df_history_full.sort_values('date', inplace=True)
    history_df = df_history_full[
        (df_history_full['date'] >= history_start_ts) &
        (df_history_full['date'] <= history_end_ts)
    ].copy()
    if history_df.empty:
         print(f"Error: No historical data found for the period {history_start_ts.date()} to {history_end_ts.date()}.")
         sys.exit(1)
    print(f"Loaded {len(history_df)} rows of historical data.")
    del df_history_full
    gc.collect()
except FileNotFoundError:
    print(f"Error: Training data file '{TRAIN_FILE}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred loading historical data: {e}")
    sys.exit(1)

# 3. Define Scope & Create Future Shell
active_stores = history_df['store'].unique()
active_items = history_df['item'].unique()
print(f"\nPredicting for {len(active_stores)} stores and {len(active_items)} items.")

future_date_range = pd.date_range(start=prediction_start_ts, end=prediction_end_ts, freq='D')
future_shell_list = []
for dt in future_date_range:
    for store_id in active_stores:
        for item_id in active_items:
            future_shell_list.append({'date': dt, 'store': store_id, 'item': item_id})
future_shell_df = pd.DataFrame(future_shell_list)
print(f"Created future shell DataFrame with shape: {future_shell_df.shape}")

# 4. Combine Data & Feature Engineering
future_shell_df['sales'] = np.nan
combined_df = pd.concat([history_df, future_shell_df], ignore_index=True)
combined_df.sort_values(['store', 'item', 'date'], inplace=True)
print(f"Combined history and future, shape: {combined_df.shape}")
del history_df, future_shell_df
gc.collect()

# เรียกใช้ create_features ที่แก้ไขแล้ว
combined_features_df = create_features(combined_df)
del combined_df
gc.collect()

# 5. Isolate & Prepare Future Features
print(f"\nIsolating features from date {prediction_start_dt} onwards...")
X_future = combined_features_df[combined_features_df['date'] >= prediction_start_ts].copy()
print(f"Isolated future features, shape: {X_future.shape}")
del combined_features_df
gc.collect()

try:
    X_future['store'] = X_future['store'].astype('category')
    X_future['item'] = X_future['item'].astype('category')
    X_future['dayofweek'] = X_future['dayofweek'].astype('category')
    X_future['month'] = X_future['month'].astype('category')
except KeyError as e:
    print(f"Error converting column to category: {e}.")
    sys.exit(1)

known_non_numeric_or_target = ['date', 'store', 'item', 'dayofweek', 'month', 'sales']
cols_to_fill = [col for col in X_future.columns if col not in known_non_numeric_or_target]
print(f"Filling NaNs with 0 in potential numerical feature columns: {len(cols_to_fill)} columns")
X_future.loc[:, cols_to_fill] = X_future.loc[:, cols_to_fill].fillna(0)
print("Handled categoricals and selectively filled NaNs.")

target = 'sales'
features = [col for col in X_future.columns if col not in ['date', target, 'id']]
X_future_final = X_future[features]
print(f"Number of features prepared for prediction: {len(features)}")

# 6. Predict
print("\nMaking predictions...")
try:
    if model is None: raise ValueError("Model object is None.")
    predictions = model.predict(X_future_final)
    predictions[predictions < 0] = 0
    print("Predictions made.")
except Exception as e:
    print(f"Error during prediction: {e}")
    print(f"DEBUG: Type of model object: {type(model)}")
    sys.exit(1)

# 7. Format Output
results_df = X_future[['date', 'store', 'item']].copy()
results_df['predicted_sales'] = predictions
results_df['predicted_sales'] = results_df['predicted_sales'].round(2)
print("\nCreated final results DataFrame.")
print("\nPrediction Results (Sample):")
print(results_df.head().to_string())

# 8. Save Predictions to CSV
print(f"\nSaving prediction results to '{OUTPUT_PREDICTIONS_FILE}'...")
try:
    results_df.to_csv(OUTPUT_PREDICTIONS_FILE, index=False)
    print("Prediction results saved successfully.")
except Exception as e:
    print(f"Error saving prediction results: {e}")

print("\nPrediction script finished.")