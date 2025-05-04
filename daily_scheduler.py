import schedule
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import gc
from datetime import date, timedelta, datetime, timezone
from pymongo import MongoClient, errors as pymongo_errors
import warnings
import sys
import os
import logging
from typing import List, Dict, Tuple, Any, Optional # เพิ่ม Type Hinting เพื่อความชัดเจน

warnings.filterwarnings('ignore')

# --- 1. Configuration ---
# ตั้งค่าต่างๆ ของสคริปต์
HISTORY_DATA_FILE = 'train.csv'                     # ไฟล์ข้อมูล Sales ในอดีต
MODEL_FILE = 'lgbm_model.joblib'                      # ไฟล์โมเดลที่ Train แล้ว
SALES_PREDICTIONS_OUTPUT_FILE = 'sales_predictions.csv' # ไฟล์ CSV ชั่วคราวสำหรับเก็บผล Predict
STOCKOUT_REPORT_OUTPUT_FILE = 'stockout_predictions.csv' # ไฟล์ CSV สำหรับเก็บผล Stockout ทั้งหมด
LAST_RUN_DATE_FILE = 'last_simulation_date.txt'    # ไฟล์เก็บวันที่สคริปต์รันสำเร็จล่าสุด
DEFAULT_SIM_START_DATE = date(2017, 1, 1)           # วันที่เริ่มต้นถ้าไม่พบไฟล์ last_run_date.txt
PREDICTION_DAYS = 31                               # จำนวนวันที่จะพยากรณ์ล่วงหน้า
# Feature Engineering Params (ต้องตรงกับตอน Train)
MAX_LAG = 364
ROLLING_WINDOW_SHIFT_DAYS = 7
LAGS = [7, 14, 21, 28, 35, 60, 91, 182, 364]
WINDOWS = [7, 14, 28, 60, 90]
# MongoDB Config (แก้ไขตามระบบของคุณ)
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://forecasting:CoSOXguW0hMipaSV@forecastingdb.ygwe41n.mongodb.net/forecastingDB?retryWrites=true&w=majority")
MONGO_DB_NAME = "Store"
MONGO_STOCK_COLLECTION = "stock"
MONGO_LEADTIME_COLLECTION = "leadtime"
MONGO_REPORT_COLLECTION = "report"                 # Collection สำหรับเก็บ Report รายวัน
# MongoDB Field Names (แก้ไขตาม Schema ของคุณ)
MONGO_FIELD_STORE_ID = "store_id"
MONGO_FIELD_ITEM_ID = "item_id"
MONGO_FIELD_QUANTITY = "quantity"                  # ชื่อ Field สต็อกใน Collection 'stock'
MONGO_FIELD_LEAD_TIME_DAYS = "lead_time_days"      # ชื่อ Field Lead Time ใน Collection 'leadtime'
# Logging Config
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)-8s - [Scheduler] - %(message)s'
LOG_DATEFMT = '%Y-%m-%d %H:%M:%S'

# --- 2. Logging Setup ---
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.info("Logging configured for daily scheduler script.")

# --- 3. Helper Functions ---

def create_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    สร้าง Features ที่ใช้ในการพยากรณ์จากข้อมูล Date และ Sales.
    (Logic การสร้าง Feature ควรจะเหมือนกับตอน Train โมเดลทุกประการ)
    """
    logging.info("Applying feature engineering...")
    df_feat = dataframe.copy()

    # === Date Features ===
    dt_col = df_feat['date'] # อ้างอิงถึงคอลัมน์ date
    df_feat['year'] = dt_col.dt.year
    df_feat['month'] = dt_col.dt.month
    df_feat['day'] = dt_col.dt.day
    df_feat['dayofweek'] = dt_col.dt.dayofweek
    df_feat['dayofyear'] = dt_col.dt.dayofyear
    df_feat['weekofyear'] = dt_col.dt.isocalendar().week.astype(int)
    df_feat['quarter'] = dt_col.dt.quarter
    df_feat['is_weekend'] = (df_feat['dayofweek'] >= 5).astype(int)
    logging.debug("  - Date features created.")

    # Sort ก่อนทำ Lag/Rolling (สำคัญ)
    df_feat.sort_values(['store', 'item', 'date'], inplace=True)

    group_cols = ['store', 'item']
    sales_col = 'sales'

    # === Lag Features ===
    for lag in LAGS:
        df_feat[f'{sales_col}_lag_{lag}'] = df_feat.groupby(group_cols, observed=True)[sales_col].shift(lag)
    logging.debug(f"  - Lag features created for lags: {LAGS}")

    # === Rolling Window Features ===
    shift_base = ROLLING_WINDOW_SHIFT_DAYS
    for window in WINDOWS:
        min_p = max(1, window // 4)
        # คำนวณ Rolling Mean/Std ของ Sales ที่ Shift แล้ว
        # Groupby -> เลือกคอลัมน์ Sales -> Transform(lambda ที่ shift().rolling().agg_func())
        df_feat[f'{sales_col}_roll_mean_{shift_base}_{window}'] = df_feat.groupby(group_cols, observed=True)[sales_col].transform(
             lambda x: x.shift(shift_base).rolling(window, min_periods=min_p).mean()
         )
        df_feat[f'{sales_col}_roll_std_{shift_base}_{window}'] = df_feat.groupby(group_cols, observed=True)[sales_col].transform(
             lambda x: x.shift(shift_base).rolling(window, min_periods=min_p).std()
         )
    logging.debug(f"  - Rolling window features created for windows: {WINDOWS} (based on lag {shift_base})")

    logging.info("Feature engineering complete.")
    return df_feat

def run_prediction_logic(current_sim_today_dt: date, prediction_days_arg: int) -> bool:
    """
    ทำหน้าที่ทั้งหมดเกี่ยวกับการพยากรณ์: โหลดข้อมูล, สร้าง Feature, Predict, Save ผล.
    Return True ถ้าสำเร็จ, False ถ้าล้มเหลว.
    """
    logging.info(f"--- Running Prediction Logic for Sim Date: {current_sim_today_dt} ---")
    try:
        # --- คำนวณช่วงวันที่ ---
        pred_start_dt = current_sim_today_dt + timedelta(days=1)
        pred_end_dt = pred_start_dt + timedelta(days=prediction_days_arg - 1)
        current_sim_today_ts = pd.Timestamp(current_sim_today_dt)
        pred_start_ts = pd.Timestamp(pred_start_dt)
        pred_end_ts = pd.Timestamp(pred_end_dt)
        logging.info(f"Predicting period: {pred_start_dt} to {pred_end_dt}")

        # --- โหลด Model ---
        logging.info(f"Loading model from '{MODEL_FILE}'...")
        model = joblib.load(MODEL_FILE)
        logging.info("Model loaded successfully.")
        if not hasattr(model, 'predict'): raise ValueError("Invalid model object loaded.")

        # --- โหลดข้อมูล History ที่จำเป็น ---
        hist_needed_start_dt = pred_start_dt - timedelta(days=MAX_LAG)
        hist_start_ts = pd.Timestamp(hist_needed_start_dt)
        hist_end_ts = current_sim_today_ts
        logging.info(f"Loading historical data from '{HISTORY_DATA_FILE}'...")
        df_history_full = pd.read_csv(HISTORY_DATA_FILE, parse_dates=['date'])
        df_history_full.sort_values('date', inplace=True)
        history_df = df_history_full[
            (df_history_full['date'] >= hist_start_ts) &
            (df_history_full['date'] <= hist_end_ts)
        ].copy()
        if history_df.empty:
             logging.error(f"No historical data found for period ending {hist_end_ts.date()}.")
             return False
        logging.info(f"Loaded {len(history_df)} rows of historical data.")
        del df_history_full; gc.collect()

        # --- เตรียมข้อมูลสำหรับพยากรณ์ ---
        active_stores = history_df['store'].unique()
        active_items = history_df['item'].unique()
        logging.info(f"Scope: {len(active_stores)} stores, {len(active_items)} items.")

        # สร้าง DataFrame โครงร่างสำหรับอนาคต
        future_date_range = pd.date_range(start=pred_start_ts, end=pred_end_ts, freq='D')
        future_index = pd.MultiIndex.from_product(
            [future_date_range, active_stores, active_items],
            names=['date', 'store', 'item']
        )
        future_shell_df = pd.DataFrame(index=future_index).reset_index()
        future_shell_df['sales'] = np.nan # เพิ่มคอลัมน์ sales สำหรับใช้ใน Feature Engineering
        logging.info(f"Future shell shape: {future_shell_df.shape}")

        # รวม History และ Future
        combined_df = pd.concat([history_df, future_shell_df], ignore_index=True)
        combined_df.sort_values(['store', 'item', 'date'], inplace=True) # Sort ก่อนสร้าง Feature
        logging.debug(f"Combined df shape: {combined_df.shape}")
        del history_df, future_shell_df; gc.collect()

        # สร้าง Features
        combined_features_df = create_features(combined_df)
        del combined_df; gc.collect()

        # แยกข้อมูลส่วนอนาคตที่จะใช้ Predict
        X_future = combined_features_df[combined_features_df['date'] >= pred_start_ts].copy()
        logging.debug(f"Isolated future features shape: {X_future.shape}")
        del combined_features_df; gc.collect()

        # --- เตรียม Features ขั้นสุดท้าย ---
        # แปลง Type เป็น Category
        categorical_cols = ['store', 'item', 'dayofweek', 'month'] # เพิ่มเติมถ้ามี
        for col in categorical_cols:
            if col in X_future.columns:
                 X_future[col] = X_future[col].astype('category')
        # เติม NaN เฉพาะคอลัมน์ตัวเลข Feature
        known_non_numeric_or_target = ['date', 'store', 'item', 'dayofweek', 'month', 'sales']
        cols_to_fill = [col for col in X_future.columns if col not in known_non_numeric_or_target]
        logging.debug(f"Filling NaNs with 0 in {len(cols_to_fill)} columns.")
        X_future.loc[:, cols_to_fill] = X_future.loc[:, cols_to_fill].fillna(0)
        # เลือก Feature ที่จะใช้ Predict (ต้องตรงกับตอน Train)
        target = 'sales'
        features = [col for col in X_future.columns if col not in ['date', target, 'id']]
        X_future_final = X_future[features]
        logging.info(f"Prepared {len(features)} features for prediction.")

        # --- ทำการ Predict ---
        logging.info("Making predictions...")
        predictions = model.predict(X_future_final)
        predictions[predictions < 0] = 0 # ไม่ให้ยอดขายติดลบ
        logging.info("Predictions made.")

        # --- จัดรูปแบบและ Save ผลลัพธ์ ---
        results_df = X_future[['date', 'store', 'item']].copy()
        results_df['predicted_sales'] = predictions
        results_df['predicted_sales'] = results_df['predicted_sales'].round(2)
        logging.info("Formatted prediction results.")
        logging.info(f"Saving prediction results to '{SALES_PREDICTIONS_OUTPUT_FILE}'...")
        results_df.to_csv(SALES_PREDICTIONS_OUTPUT_FILE, index=False)
        logging.info("Prediction results saved successfully.")
        return True # คืนค่าว่าทำงานสำเร็จ

    # --- จัดการ Error ที่อาจเกิดขึ้นระหว่าง Process ---
    except FileNotFoundError as e:
        logging.error(f"File not found during prediction ({HISTORY_DATA_FILE} or {MODEL_FILE}): {e}")
        return False
    except KeyError as e:
         logging.error(f"Column not found - check CSV headers/config: {e}", exc_info=True)
         return False
    except Exception as e:
        logging.error(f"Error during prediction logic: {e}", exc_info=True)
        return False # คืนค่าว่าทำงานล้มเหลว

def run_stockout_calculation(prediction_start_dt: date, prediction_end_dt: date):
    """
    คำนวณวันหมดสต็อกโดยอ่านผล Predict, ดึงข้อมูลจาก MongoDB, และ Insert Report Alert.
    """
    logging.info("--- Running Stockout Calculation Logic ---")
    global SALES_PREDICTIONS_OUTPUT_FILE, MONGO_URI, MONGO_DB_NAME, MONGO_STOCK_COLLECTION, MONGO_LEADTIME_COLLECTION, MONGO_REPORT_COLLECTION
    global MONGO_FIELD_STORE_ID, MONGO_FIELD_ITEM_ID, MONGO_FIELD_QUANTITY
    global MONGO_FIELD_LEAD_TIME_DAYS, STOCKOUT_REPORT_OUTPUT_FILE
    mongo_client = None
    db = None
    try:
        # --- โหลด Predictions ---
        logging.info(f"Loading predictions from {SALES_PREDICTIONS_OUTPUT_FILE}...")
        predictions_df = pd.read_csv(SALES_PREDICTIONS_OUTPUT_FILE, parse_dates=['date'])
        required_cols = ['date', 'store', 'item', 'predicted_sales']
        if not all(col in predictions_df.columns for col in predictions_df.columns):
            missing_cols = [col for col in required_cols if col not in predictions_df.columns]
            logging.error(f"Prediction file '{SALES_PREDICTIONS_OUTPUT_FILE}' missing columns: {missing_cols}")
            return
        if predictions_df.empty: logging.error("Prediction file is empty."); return
        logging.info(f"Loaded {len(predictions_df)} prediction records.")

        # --- เชื่อมต่อ MongoDB ---
        logging.info(f"Connecting to MongoDB (DB: {MONGO_DB_NAME})...")
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tls=True, tlsAllowInvalidCertificates=False)
        db = mongo_client[MONGO_DB_NAME]
        db.command('ping')
        stock_collection = db[MONGO_STOCK_COLLECTION]
        leadtime_collection = db[MONGO_LEADTIME_COLLECTION]
        report_collection = db[MONGO_REPORT_COLLECTION]
        logging.info("MongoDB connection successful.")

        # --- ดึง Inventory & Lead Time ---
        inventory_map = {}
        leadtime_map = {}
        logging.info(f"Fetching current stock from collection '{MONGO_STOCK_COLLECTION}'...")
        stock_projection = {MONGO_FIELD_STORE_ID: 1, MONGO_FIELD_ITEM_ID: 1, MONGO_FIELD_QUANTITY: 1, "_id": 0}
        stock_cursor = stock_collection.find({}, stock_projection)
        stock_fetched_count = 0; stock_missing_fields = 0
        for doc in stock_cursor:
            store_id = doc.get(MONGO_FIELD_STORE_ID); item_id = doc.get(MONGO_FIELD_ITEM_ID); stock = doc.get(MONGO_FIELD_QUANTITY)
            if store_id is not None and item_id is not None and stock is not None:
                try: inventory_map[(int(store_id), int(item_id))] = float(stock); stock_fetched_count += 1
                except (ValueError, TypeError): stock_missing_fields += 1
            else: stock_missing_fields += 1
        logging.info(f"Fetched stock data for {stock_fetched_count} combinations.")
        if stock_missing_fields > 0: logging.warning(f"Skipped {stock_missing_fields} stock documents.")

        logging.info(f"Fetching lead time data from collection '{MONGO_LEADTIME_COLLECTION}'...")
        leadtime_projection = {MONGO_FIELD_ITEM_ID: 1, MONGO_FIELD_LEAD_TIME_DAYS: 1, "_id": 0}
        leadtime_cursor = leadtime_collection.find({}, leadtime_projection)
        leadtime_fetched_count = 0; leadtime_missing_fields = 0
        for doc in leadtime_cursor:
             item_id = doc.get(MONGO_FIELD_ITEM_ID); lead_time = doc.get(MONGO_FIELD_LEAD_TIME_DAYS)
             if item_id is not None and lead_time is not None:
                  try: leadtime_map[int(item_id)] = int(lead_time); leadtime_fetched_count +=1
                  except(ValueError, TypeError): leadtime_missing_fields += 1
             else: leadtime_missing_fields += 1
        logging.info(f"Fetched lead time data for {leadtime_fetched_count} items.")
        if leadtime_missing_fields > 0: logging.warning(f"Skipped {leadtime_missing_fields} lead time documents.")

        # --- คำนวณ Stockout และ Alert ---
        logging.info("Calculating predicted days until stockout and reorder alert...")
        stockout_results = []
        items_without_stock_info = 0
        items_without_leadtime = 0
        prediction_period_days = (prediction_end_dt - prediction_start_dt).days + 1
        logging.info(f"Using prediction period: {prediction_period_days} days")

        grouped_predictions = predictions_df.groupby(['store', 'item'], observed=True)
        for name, group in grouped_predictions:
            store_id, item_id = name
            store_item_key = (int(store_id), int(item_id))
            current_stock = inventory_map.get(store_item_key)
            lead_time_val = leadtime_map.get(int(item_id)) # Get lead time (might be None)

            days_until_stockout_str = ""
            stock_value_for_result: Any = "N/A"
            lead_time_for_result: Any = "N/A"
            reorder_alert = "N/A" # Default

            if current_stock is None:
                days_until_stockout_str = "Unknown Stock"; items_without_stock_info += 1; reorder_alert = "N/A (Unknown Stock)"
            elif current_stock <= 0:
                days_until_stockout_str = "0 (Already Stockout)"; stock_value_for_result = current_stock; reorder_alert = "Yes (Already Stockout)"
            else:
                # Simulate stock depletion
                remaining_stock = float(current_stock); days_counted = 0; stockout_day_found = False
                sorted_group = group.sort_values('date')
                for _, row in sorted_group.iterrows():
                    if remaining_stock <= 0: stockout_day_found = True; break
                    predicted_sale_today = row['predicted_sales']; remaining_stock -= predicted_sale_today; days_counted += 1
                    if remaining_stock <= 0: stockout_day_found = True; break
                stock_value_for_result = current_stock
                if stockout_day_found: days_until_stockout_str = str(days_counted)
                else: days_until_stockout_str = f"> {prediction_period_days}"

                # Determine reorder alert status
                try:
                    if days_until_stockout_str.startswith(">"): days_numeric = float('inf')
                    else: days_numeric = int(days_until_stockout_str)

                    if lead_time_val is not None:
                        lead_time_numeric = int(lead_time_val); lead_time_for_result = lead_time_numeric
                        reorder_threshold = lead_time_numeric + 2
                        if days_numeric <= reorder_threshold: reorder_alert = "Yes"
                        else: reorder_alert = "No"
                    else: reorder_alert = "N/A (No Lead Time)"; items_without_leadtime +=1
                except (ValueError, TypeError): reorder_alert = "N/A (Calc Error)"

            # Add record for this item/store
            stockout_results.append({
                    'Store': store_id, 'Item': item_id,
                    'Current Stock': stock_value_for_result, 'Lead Time (Days)': lead_time_for_result,
                    'Days Until Stockout': days_until_stockout_str,
                    'Reorder Alert (<= LT+2)': reorder_alert
                })
        logging.info("Stockout calculation complete.")
        if items_without_stock_info > 0: logging.warning(f"Could not calc stockout for {items_without_stock_info} items.")
        if items_without_leadtime > 0: logging.warning(f"Could not determine reorder alert for {items_without_leadtime} items.")


        # --- จัดการผลลัพธ์ Stockout ---
        if not stockout_results: logging.warning("Stockout calculation did not produce results."); return
        stockout_df = pd.DataFrame(stockout_results)

        # Sort function (เหมือนเดิม)
        def sort_key_stockout(day_val):
            if isinstance(day_val, str):
                if day_val.startswith(">"):
                    try: return float(day_val[1:].strip()) + 0.5
                    except ValueError: return float('inf')
                if "Unknown" in day_val: return float('inf') - 1
                if "Already" in day_val: return 0
                try: return float(day_val)
                except ValueError: return float('inf') - 2
            elif isinstance(day_val, (int, float)): return float(day_val)
            return float('inf')
        stockout_df['sort_val'] = stockout_df['Days Until Stockout'].apply(sort_key_stockout)
        stockout_df.sort_values(by=['Reorder Alert (<= LT+2)', 'sort_val'], ascending=[True, True], inplace=True)
        stockout_df.drop(columns=['sort_val'], inplace=True)

        # แสดงผล Log
        logging.info("--- Predicted Stockout and Reorder Alert (Sorted) ---\n" + stockout_df.to_string())

        # Save Stockout CSV
        logging.info(f"Saving stockout results to '{STOCKOUT_REPORT_OUTPUT_FILE}'...")
        stockout_df.to_csv(STOCKOUT_REPORT_OUTPUT_FILE, index=False)
        logging.info("Stockout results saved successfully.")

        # --- Insert Report to MongoDB (เอา alert_reason ออก) ---
        alert_df = stockout_df[stockout_df['Reorder Alert (<= LT+2)'].astype(str).str.startswith("Yes", na=False)].copy()
        if not alert_df.empty:
            logging.info(f"Found {len(alert_df)} items needing reorder alert. Preparing insert for '{MONGO_REPORT_COLLECTION}'...")
            # Rename columns for the sub-document array
            alert_df.rename(columns={
                'Store': MONGO_FIELD_STORE_ID,
                'Item': MONGO_FIELD_ITEM_ID,
                'Current Stock': 'current_stock', # ใช้ชื่อ Field ที่สื่อความหมายใน Report
                'Lead Time (Days)': 'item_lead_time',
                'Days Until Stockout': 'predicted_stockout_in_days'
                # ไม่ต้องเอา 'Reorder Alert (<= LT+2)' มาแล้ว
            }, inplace=True)

            # Sort alert items before creating the list
            logging.debug(f"Sorting alert items by store_id and item_id for the report...")
            alert_df.sort_values(by=[MONGO_FIELD_STORE_ID, MONGO_FIELD_ITEM_ID], inplace=True)

            # Select only the columns needed for the array
            report_item_columns = [
                MONGO_FIELD_STORE_ID, MONGO_FIELD_ITEM_ID,
                'current_stock', 'item_lead_time',
                'predicted_stockout_in_days'
                # เอา alert_reason ออก
            ]
            items_to_reorder_list = alert_df[report_item_columns].to_dict('records')

            # Create the main report document
            report_date_ts = datetime.now(timezone.utc) # วันเวลาที่สร้าง Report
            single_report_doc = {
                "report_date": report_date_ts,
                "items_to_reorder": items_to_reorder_list # Array ที่เรียงแล้ว และไม่มี alert_reason
            }
            try:
                # Insert ลง MongoDB
                insert_result = report_collection.insert_one(single_report_doc)
                logging.info(f"Successfully inserted daily report document with ID: {insert_result.inserted_id} containing {len(items_to_reorder_list)} items.")
            except Exception as insert_e:
                logging.error(f"Failed to insert alert report document into '{MONGO_REPORT_COLLECTION}': {insert_e}")
        else:
            logging.info("No items triggered the reorder alert in this run for report generation.")
        # --- End Insert Report ---

    except FileNotFoundError:
        logging.error(f"Prediction file '{SALES_PREDICTIONS_OUTPUT_FILE}' not found.")
    except Exception as e:
        logging.error(f"Error during stockout calculation logic: {e}", exc_info=True)
    finally:
        if mongo_client:
            mongo_client.close()
            logging.debug("MongoDB connection closed after stockout calculation.")

# --- 5. Scheduling Logic ---

def daily_job():
    """ฟังก์ชันงานหลักที่จะรันทุกวันตามเวลา"""
    global PREDICTION_DAYS, LAST_RUN_DATE_FILE, DEFAULT_SIM_START_DATE
    logging.info("=== Starting scheduled daily job ===")

    # กำหนดวันที่ทำงานปัจจุบัน
    current_sim_today = None
    try:
        if os.path.exists(LAST_RUN_DATE_FILE):
            with open(LAST_RUN_DATE_FILE, 'r') as f: last_run_date_str = f.read().strip()
            if last_run_date_str:
                logging.info(f"Read last simulation date: {last_run_date_str}")
                current_sim_today = date.fromisoformat(last_run_date_str) + timedelta(days=1)
            else:
                 logging.warning(f"State file empty. Starting from initial date.")
                 current_sim_today = DEFAULT_SIM_START_DATE
        else:
            logging.warning(f"State file not found. Starting from initial date: {DEFAULT_SIM_START_DATE}")
            current_sim_today = DEFAULT_SIM_START_DATE
    except Exception as e:
        logging.error(f"Error reading state file: {e}. Starting from initial date.")
        current_sim_today = DEFAULT_SIM_START_DATE

    logging.info(f"Determined Current Simulation Date: {current_sim_today}")
    prediction_days_config = PREDICTION_DAYS
    pred_start_dt = current_sim_today + timedelta(days=1)
    pred_end_dt = pred_start_dt + timedelta(days=prediction_days_config - 1)

    prediction_success = False
    job_successful = False
    try:
        # รัน Prediction
        prediction_success = run_prediction_logic(current_sim_today, prediction_days_config)

        # รัน Stockout Calculation (ถ้า Prediction สำเร็จ)
        if prediction_success:
            run_stockout_calculation(pred_start_dt, pred_end_dt)
            job_successful = True # Assume success if it runs without fatal error
        else:
            logging.warning("Skipping stockout calculation because prediction failed.")
            job_successful = False

        # บันทึก State ถ้าสำเร็จ
        if job_successful:
            try:
                with open(LAST_RUN_DATE_FILE, 'w') as f: f.write(current_sim_today.isoformat())
                logging.info(f"Successfully updated state file '{LAST_RUN_DATE_FILE}' with date: {current_sim_today.isoformat()}")
            except OSError as e:
                logging.error(f"Failed to write state file '{LAST_RUN_DATE_FILE}': {e}")

    except Exception as e:
        logging.error(f"Unhandled exception in daily_job for date {current_sim_today}: {e}", exc_info=True)
    finally:
         logging.info(f"=== Scheduled daily job finished for date {current_sim_today} ===")


# --- ตั้งเวลาให้รัน daily_job ทุกวัน เวลา 08:00 น. ---
schedule.every().day.at("08:00").do(daily_job)

logging.info("Scheduler started. Waiting for the scheduled time (08:00)...")
print(f"Current time: {datetime.now()}. Waiting for 08:00 to run the first job.")
print(f"(To test immediately, uncomment 'daily_job()' below and comment out the 'while True' loop)")

# --- ใช้สำหรับทดสอบรันทันที 1 ครั้ง ---
# logging.warning("Running job immediately for testing purposes instead of scheduling.")
# daily_job()
# ---------------------------------------


# --- 6. ทำให้สคริปต์ทำงานค้างไว้ และรันงานตามตาราง ---
while True:
    try:
        schedule.run_pending()
        time.sleep(30)
    except KeyboardInterrupt:
        logging.info("Scheduler stopped manually.")
        break
    except Exception as e:
        logging.error(f"Error in scheduler loop: {e}", exc_info=True)
        logging.info("Waiting 5 minutes before retrying scheduler loop...")
        time.sleep(300)
# ----------------------------------------------------

logging.info("Scheduler script finished.")