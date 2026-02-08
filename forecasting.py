import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import joblib
import os
from processing import smoothed_target_encode
from config import MODEL_PATH

class Forecaster:
    def __init__(self):
        self.model = None
        self.features = [
            "Month", "Year",
            "IdCompany_te", "IdProduct_te", "IdBranchCompany_te", "IdWarehouse_te",
            "lag_1", "lag_3", "lag_6",
            "roll_3", "roll_6", "roll_12",
            "UnitPrice_log"
        ]
        self.target = "Quantity_log"
        self.te_groups = ["IdCompany", "IdProduct", "IdBranchCompany", "IdWarehouse"]
        
    def train(self, df, split_date="2025-01-01"):
        """Trains the LightGBM model and returns metrics."""
        train_df = df[df["date"] < split_date].copy()
        test_df = df[df["date"] >= split_date].copy()
        
        self.global_mean = train_df[self.target].mean()
        
        # Apply Target Encoding
        self.encodings = {} # Store for future inference
        for col in self.te_groups:
            # We calculate 'smooth' map from training data
            train_df, test_df, smooth_map = smoothed_target_encode(train_df, test_df, col, self.target, self.global_mean)
            self.encodings[col] = smooth_map
            
        X_train = train_df[self.features]
        y_train = train_df[self.target]
        X_test = test_df[self.features]
        y_test = test_df[self.target]
        
        # Hyperparameter tuning with GridSearchCV
        tscv = TimeSeriesSplit(n_splits=5)
        
        # ... (GridSearch config remains same) ...
        # Simplified for brevity in this specific replacement block
        param_grid = {
            'num_leaves': [31, 64, 127],
            'reg_alpha': [0.1, 0.5],
            'min_data_in_leaf': [30, 50, 100, 300, 400],
            'lambda_l1': [0, 1, 1.5],
            'lambda_l2': [0, 1]
        }
        
        lgbm = lgb.LGBMRegressor(random_state=42, importance_type='gain', n_jobs=-1, verbose=-1)
        
        grid_search = GridSearchCV(
            estimator=lgbm,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")
        
        # Metrics
        preds_log = self.model.predict(X_test)
        y_test_real = np.expm1(y_test)
        preds_real = np.expm1(preds_log)
        
        metrics = {
            "MAE": mean_absolute_error(y_test_real, preds_real),
            "RMSE": mean_squared_error(y_test_real, preds_real, squared=False),
            "R2": r2_score(y_test_real, preds_real)
        }
        
        return metrics, self.model, test_df

    def predict_test(self, df, split_date="2025-01-01"):
        """Generates predictions on the test set using the trained model."""
        if self.model is None:
            if not self.load():
                raise Exception("Model not found. Please train first.")
                
        test_df = df[df["date"] >= split_date].copy()
        
        # Apply Target Encoding using SAVED encodings
        if not hasattr(self, 'encodings') or not hasattr(self, 'global_mean'):
             # Fallback if loading old model without encodings (should re-train ideally)
             print("Warning: Encodings not found in model file. Re-calculating (Suboptimal).")
             # ... fallback logic ...
             train_df_subset = df[df["date"] < split_date].copy()
             self.global_mean = train_df_subset[self.target].mean()
             self.encodings = {}
             for col in self.te_groups:
                  _, _, smooth = smoothed_target_encode(train_df_subset, None, col, self.target, self.global_mean)
                  self.encodings[col] = smooth
        
        for col in self.te_groups:
             # Use the saved smooth map
             # Note: smoothed_target_encode expects 'train' as first arg, but we only have 'test' to transform.
             # We hack it by passing 'test' as 'train' but with 'smooth_map' provided?
             # No, function signature: train, test, ...
             # Transform Train is mandatory in current func.
             # Let's just do the map manually here for clarity or refactor func.
             # Re-using func: pass test as "train" argument? No, logic updates "train" inplace.
             test_df, _, _ = smoothed_target_encode(test_df, None, col, self.target, self.global_mean, smooth_map=self.encodings[col])
        
        if test_df.empty:
            return None, pd.DataFrame()

        X_test = test_df[self.features]
        preds_log = self.model.predict(X_test)
        preds_real = np.expm1(preds_log)
        
        # Metrics
        y_test = test_df[self.target]
        y_test_real = np.expm1(y_test)
        
        metrics = {
            
            "MAE": mean_absolute_error(y_test_real, preds_real),
            "RMSE": np.sqrt(mean_squared_error(y_test_real, preds_real)),
            "R2": r2_score(y_test_real, preds_real)
        }
        
        return metrics, test_df

    def save(self, metrics=None):
        if self.model:
            # We save a dictionary containing model + metadata
            payload = {
                "model": self.model,
                "encodings": self.encodings,
                "global_mean": self.global_mean,
                "last_metrics": metrics # Save stats for performance monitoring
            }
            joblib.dump(payload, MODEL_PATH)
            self.last_metrics = metrics
            
    def load(self):
        if os.path.exists(MODEL_PATH):
            payload = joblib.load(MODEL_PATH)
            # Check if it's the new dictionary format or old raw model
            if isinstance(payload, dict) and "model" in payload:
                self.model = payload["model"]
                self.encodings = payload.get("encodings", {})
                self.global_mean = payload.get("global_mean", 0)
                self.last_metrics = payload.get("last_metrics", {})
            else:
                self.model = payload # Backward compatibility
                self.last_metrics = {}
            return True
        return False

    def predict_next_month(self, raw_df):
        """
        Generates purchase suggestions (Quantity) for the strictly next month 
        relative to the last date in raw_df.
        """
        if self.model is None:
            if not self.load():
                 raise Exception("Model not loaded. Please train first.")
                 
        # 1. Determine Next Month
        raw_df = raw_df.copy()
        raw_df["CreateDate"] = pd.to_datetime(raw_df["CreateDate"])
        last_date = raw_df["CreateDate"].max()
        next_month_date = (pd.to_datetime(last_date).to_period('M') + 1).to_timestamp()
        
        # 2. Construct Placeholder Data
        # We need the last state of every product to generate features (Lags)
        # Identify active products (appeared in history) and get their LAST KNOWN PRICE
        
        # --- BUSINESS LOGIC: RECENCY FILTER ---
        # Only predict for products sold in the last 4 months to avoid discontinued items
        cutoff_date = last_date - pd.DateOffset(months=4)
        active_mask = raw_df["CreateDate"] >= cutoff_date
        raw_df_active = raw_df[active_mask].copy()
        
        # Sort by date to ensure 'last' is truly recent
        raw_df_sorted = raw_df_active.sort_values("CreateDate")
        unique_groups = raw_df_sorted[['IdCompany', 'IdProduct', 'IdBranchCompany', 'IdWarehouse', 'UnitPrice']].drop_duplicates(
            subset=['IdCompany', 'IdProduct', 'IdBranchCompany', 'IdWarehouse'], 
            keep='last'
        )
        
        future_df = unique_groups.copy()
        future_df["CreateDate"] = next_month_date
        future_df["Quantity"] = 0 # Dummy
        # UnitPrice is now inherited from the last transaction
        # If price is 0 or NaN, we might want to default to 1 or global mean to pass filters.
        # But assuming data quality is okay (processed > 1 filter checks later).
        # Just in case, fillna(0) and let preprocessing filter handle if price < 1
        future_df["UnitPrice"] = future_df["UnitPrice"].fillna(0) 
        
        # 3. Concatenate and Process
        # We append future rows to history so that rolling/lags can be computed
        combined_raw = pd.concat([raw_df, future_df], ignore_index=True)
        
        # Process (is_inference=True keeps our dummy rows)
        from processing import preprocess_data
        combined_processed = preprocess_data(combined_raw, is_inference=True)
        
        # 4. Filter for Target Month
        future_rows = combined_processed[combined_processed["date"] == next_month_date].copy()
        
        if future_rows.empty:
            return pd.DataFrame()
            
        # 5. Apply Target Encoding (using SAVED map)
        for col in self.te_groups:
             # Transform using saved encodings
             future_rows, _, _ = smoothed_target_encode(future_rows, None, col, self.target, self.global_mean, smooth_map=self.encodings[col])
             
        # 6. Predict
        X_future = future_rows[self.features]
        preds_log = self.model.predict(X_future)

        preds_real = np.expm1(preds_log).round(2)
        
        future_rows["Predicted_Quantity"] = np.ceil(preds_real).astype(int)
        
        return future_rows[["date", "IdProduct", "IdWarehouse", "Predicted_Quantity"]]
