from database import fetch_sales_data
from processing import preprocess_data
from forecasting import Forecaster
from config import SPLIT_DATE
import pandas as pd
import numpy as np
import sys
import os

def run_pipeline():
    print("Starting Forecasting Pipeline...")
    
    # 1. Fetch Data
    print("Fetching data from database...")
    try:
        raw_df = fetch_sales_data()
        print(f"Data fetched: {raw_df.shape[0]} rows")
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)
        
    # 2. Process Data
    print("Processing data...")
    try:
        processed_df = preprocess_data(raw_df)
        print(f"Data processed: {processed_df.shape[0]} monthly records")
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)
        
    # 3. Train or Load Model
    forecaster = Forecaster()
    model_loaded = False
    
    if os.path.exists("best_model.pkl"):
        print(" Saved model found. Loading...")
        if forecaster.load():
            model_loaded = True
            print(" Model loaded successfully.")
            
            # --- HEALTH CHECK ON LOAD ---
            metrics = forecaster.last_metrics
            r2_val = metrics.get("R2", -1)
            if r2_val != -1:
                print(f" Quality Check (History): R2 = {r2_val:.2f}")
                if r2_val < 0.40:
                    print(" ALERT: This model has low precision. Consider re-training.")
            else:
                print(" Warning: No historical metrics found for this model.")
    
    if not model_loaded:
        print(f"Training model (Split Date: {SPLIT_DATE})...")
        try:
            metrics, _, test_df = forecaster.train(processed_df, split_date=SPLIT_DATE)
            print(f"Training complete. Metrics: {metrics}")
            
            # Save model with its metrics
            forecaster.save(metrics=metrics)
            print("Model and metrics saved to 'best_model.pkl'.")
            
            # --- VALIDATION / TEST LOGIC (Only runs when training) ---
            print("Generating predictions for validation/review...")
            
            if test_df is not None and not test_df.empty:
                X_test = test_df[forecaster.features]
                preds_log = forecaster.model.predict(X_test)
                preds_real = np.expm1(preds_log)
                
                # Add predictions to dataframe
                test_df["Predicted_Quantity"] = preds_real.round(2)
                test_df["Actual_Quantity"] = np.expm1(test_df["Quantity_log"]).round(2)
                test_df["Error"] = test_df["Actual_Quantity"] - test_df["Predicted_Quantity"]
                
                # Save Results
                output_file = "forecast_results.csv"
                columns_to_save = ["date", "IdProduct", "IdWarehouse", "Actual_Quantity", "Predicted_Quantity", "Error"]
                test_df[columns_to_save].to_csv(output_file, index=False)
                print(f"Validation Results saved to '{output_file}'")

            # --- QUALITY GATE (Immediate Feedack) ---
            r2_val = metrics.get("R2", -1)
            if r2_val < 0.40:
                print("\n" + "!"*50)
                print(f"WARNING: Model Quality is Low (R2 = {r2_val:.2f})")
                print("   Data might be highly volatile. Use suggestions with caution.")
                print("!"*50 + "\n")
            else:
                print(f"\nQuality Check Passed (R2 = {r2_val:.2f})\n")

        except Exception as e:
            print(f"Error during training: {e}")
            sys.exit(1)
    else:
        print("Pipeline optimized: Skipping test phase (Using existing model).")

 

    # 5. GENERATE NEXT MONTH FORECAST (THE REAL GOAL)
    print("PREDICTING NEXT MONTH (Future)...")
    
    try:
        # We need raw_df for this. It is already available in this scope.
        future_forecast = forecaster.predict_next_month(raw_df)
        
        if not future_forecast.empty:
            # --- MEJORA: Recuperar nombres de productos ---
            # Creamos un mapeo rápido ID -> Nombre usando los datos originales
            product_names = raw_df[['IdProduct', 'NameProduct']].drop_duplicates('IdProduct')
            
            # Unimos los nombres al pronóstico
            future_forecast = future_forecast.merge(product_names, on='IdProduct', how='left')
            
            # Reordenamos columnas para que sea bonito
            cols = ['date', 'IdProduct', 'NameProduct', 'IdWarehouse', 'Predicted_Quantity']
            future_forecast = future_forecast[cols]
            
            # Guardar
            future_file = "next_month_forecast.csv"
            future_forecast.to_csv(future_file, index=False)
            print(f"Future Forecast saved to '{future_file}'")
            
        else:
            print("Warning: No predictions generated for next month.")
            
    except Exception as e:
        print(f"Error predicting next month: {e}")
    
    print("Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()
