import pandas as pd
import numpy as np
from pymongo import MongoClient, errors as pymongo_errors
from datetime import timedelta, datetime
import sys
import os

# --- Configuration ---
PREDICTIONS_FILE = 'sales_predictions.csv' # ไฟล์ Input ที่ได้จาก predict_sales.py
STOCKOUT_OUTPUT_FILE = 'stockout_predictions.csv' # ไฟล์ Output ที่จะ Save

# --- MongoDB Configuration (ใช้ชื่อ DB ที่ถูกต้อง) ---
MONGO_URI = "mongodb+srv://forecasting:CoSOXguW0hMipaSV@forecastingdb.ygwe41n.mongodb.net/forecastingDB?retryWrites=true&w=majority"
MONGO_DB_NAME = "Store"                 # <<< แก้ไขเป็น "Store" (S ตัวใหญ่)
MONGO_STOCK_COLLECTION = "stock"
MONGO_LEADTIME_COLLECTION = "leadtime"
# Field Names
MONGO_STOCK_FIELD_STORE = "store_id"
MONGO_STOCK_FIELD_ITEM = "item_id"
MONGO_STOCK_FIELD_STOCK = "quantity"
MONGO_LEADTIME_FIELD_ITEM = "item_id"
MONGO_LEADTIME_FIELD_DAYS = "lead_time_days"

# --- โค้ดหลัก ---
print("--- Starting Stockout Calculation Process ---")

# 1. Load Predictions
print(f"Loading predictions from {PREDICTIONS_FILE}...")
try:
    predictions_df = pd.read_csv(PREDICTIONS_FILE, parse_dates=['date'])
    required_cols = ['date', 'store', 'item', 'predicted_sales']
    if not all(col in predictions_df.columns for col in required_cols):
         missing_cols = [col for col in required_cols if col not in predictions_df.columns]
         print(f"Error: Prediction file '{PREDICTIONS_FILE}' is missing required columns: {missing_cols}")
         sys.exit(1)
    if predictions_df.empty:
        print("Error: Prediction file is empty.")
        sys.exit(1)
    print(f"Loaded {len(predictions_df)} prediction records.")
except FileNotFoundError:
    print(f"Error: Prediction file '{PREDICTIONS_FILE}' not found. Please run predict_sales.py first.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading predictions: {e}")
    sys.exit(1)

# 2. Connect to MongoDB & Fetch Data
mongo_client = None
inventory_map = {}
leadtime_map = {}
print(f"\nConnecting to MongoDB (DB: {MONGO_DB_NAME})...")
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tls=True, tlsAllowInvalidCertificates=False)
    db = mongo_client[MONGO_DB_NAME] # ใช้ DB 'Store'
    db.command('ping')
    stock_collection = db[MONGO_STOCK_COLLECTION]
    leadtime_collection = db[MONGO_LEADTIME_COLLECTION]
    print("MongoDB connection successful.")

    # 3. Fetch Inventory (Stock) Data
    print(f"Fetching current stock from collection '{MONGO_STOCK_COLLECTION}'...")
    stock_projection = {MONGO_STOCK_FIELD_STORE: 1, MONGO_STOCK_FIELD_ITEM: 1, MONGO_STOCK_FIELD_STOCK: 1, "_id": 0}
    stock_cursor = stock_collection.find({}, stock_projection)
    stock_fetched_count = 0
    stock_missing_fields = 0
    for doc in stock_cursor:
        store_id_val = doc.get(MONGO_STOCK_FIELD_STORE)
        item_id_val = doc.get(MONGO_STOCK_FIELD_ITEM)
        stock_val = doc.get(MONGO_STOCK_FIELD_STOCK)
        if store_id_val is not None and item_id_val is not None and stock_val is not None:
            try:
                store_item_key = (int(store_id_val), int(item_id_val))
                inventory_map[store_item_key] = float(stock_val)
                stock_fetched_count += 1
            except (ValueError, TypeError):
                stock_missing_fields += 1
        else:
             stock_missing_fields += 1
    print(f"Fetched stock data for {stock_fetched_count} store-item combinations.") # ควรจะได้ค่า > 0 แล้ว
    if stock_missing_fields > 0: print(f"Warning: Skipped {stock_missing_fields} stock documents due to missing/invalid fields.")

    # 4. Fetch Lead Time Data
    print(f"Fetching lead time data from collection '{MONGO_LEADTIME_COLLECTION}'...")
    leadtime_projection = {MONGO_LEADTIME_FIELD_ITEM: 1, MONGO_LEADTIME_FIELD_DAYS: 1, "_id": 0}
    leadtime_cursor = leadtime_collection.find({}, leadtime_projection)
    leadtime_fetched_count = 0
    leadtime_missing_fields = 0
    for doc in leadtime_cursor:
        item_id_val = doc.get(MONGO_LEADTIME_FIELD_ITEM)
        lead_time_val = doc.get(MONGO_LEADTIME_FIELD_DAYS)
        if item_id_val is not None and lead_time_val is not None:
            try:
                leadtime_map[int(item_id_val)] = int(lead_time_val)
                leadtime_fetched_count += 1
            except(ValueError, TypeError):
                 leadtime_missing_fields += 1
        else: leadtime_missing_fields += 1
    print(f"Fetched lead time data for {leadtime_fetched_count} items.") # ควรจะได้ค่า > 0 แล้ว
    if leadtime_missing_fields > 0: print(f"Warning: Skipped {leadtime_missing_fields} lead time documents due to missing/invalid fields.")

except pymongo_errors.ConnectionFailure as e:
    print(f"Error: MongoDB connection failed: {e}")
    print("Cannot calculate stockout without MongoDB connection. Exiting.")
    sys.exit(1)
except pymongo_errors.ConfigurationError as e:
     print(f"Error: MongoDB configuration error (check URI): {e}")
     sys.exit(1)
except pymongo_errors.OperationFailure as e:
    print(f"Error: MongoDB operation failure: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error during MongoDB operation: {e}")
    print("Cannot calculate stockout without inventory data. Exiting.")
    sys.exit(1)
finally:
    if mongo_client:
        mongo_client.close()
        print("MongoDB connection closed.")

# 5. Calculate Stockout Days
print("\nCalculating predicted days until stockout...")
stockout_results = []
items_without_stock_info = 0
prediction_period_days = 0
try:
    min_pred_date = predictions_df['date'].min()
    max_pred_date = predictions_df['date'].max()
    prediction_period_days = (max_pred_date - min_pred_date).days + 1
    print(f"Inferred prediction period: {prediction_period_days} days ({min_pred_date.date()} to {max_pred_date.date()})")
except Exception as e:
    print(f"Warning: Could not determine prediction period: {e}")
    prediction_period_days = 30 # Fallback
    print(f"Using default prediction period: {prediction_period_days} days")

grouped_predictions = predictions_df.groupby(['store', 'item'], observed=True)
for name, group in grouped_predictions:
    store_id, item_id = name
    store_item_key = (int(store_id), int(item_id))
    current_stock = inventory_map.get(store_item_key)
    lead_time = leadtime_map.get(int(item_id), 'N/A')
    days_until_stockout_str = ""
    stock_value_for_result = "N/A"
    if current_stock is None:
        days_until_stockout_str = "Unknown Stock"
        items_without_stock_info += 1
    elif current_stock <= 0:
        days_until_stockout_str = "0 (Already Stockout)"
        stock_value_for_result = current_stock
    else:
        remaining_stock = float(current_stock)
        days_counted = 0
        stockout_day_found = False
        sorted_group = group.sort_values('date')
        for _, row in sorted_group.iterrows():
            if remaining_stock <= 0: stockout_day_found = True; break
            predicted_sale_today = row['predicted_sales']
            remaining_stock -= predicted_sale_today
            days_counted += 1
            if remaining_stock <= 0: stockout_day_found = True; break
        stock_value_for_result = current_stock
        if stockout_day_found: days_until_stockout_str = str(days_counted)
        else: days_until_stockout_str = f"> {prediction_period_days}"
    stockout_results.append({
            'Store': store_id, 'Item': item_id,
            'Current Stock': stock_value_for_result,
            'Lead Time (Days)': lead_time,
            'Days Until Stockout': days_until_stockout_str
        })
print("Stockout calculation complete.")
if items_without_stock_info > 0:
    print(f"Warning: Could not calculate stockout for {items_without_stock_info} items due to missing inventory data in MongoDB.")

# 6. Format and Output Results
if not stockout_results:
    print("Stockout calculation did not produce any results.")
else:
    stockout_df = pd.DataFrame(stockout_results)
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
    stockout_df.sort_values(by='sort_val', ascending=True, inplace=True)
    stockout_df.drop(columns=['sort_val'], inplace=True)
    print("\n--- Predicted Days Until Stockout (Sorted) ---")
    print(stockout_df.to_string())

    # 7. Save Stockout Results to CSV
    print(f"\nSaving stockout results to {STOCKOUT_OUTPUT_FILE}...")
    try:
        stockout_df.to_csv(STOCKOUT_OUTPUT_FILE, index=False)
        print("Stockout results saved successfully.")
    except Exception as e:
        print(f"Error saving stockout results: {e}")

print("\nStockout calculation script finished.")