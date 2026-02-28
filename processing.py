import pandas as pd
import numpy as np

def preprocess_data(df, is_inference=False):

    df_monthly = df.copy()
    
    # Aseguramos que CreateDate sea datetime (la que viene del nuevo SQL
    df_monthly["CreateDate"] = pd.to_datetime(df_monthly["CreateDate"])
    
    # REPARACIÓN DEL ERROR: Creamos 'year_month' explícitamente
    # Esto es lo que el entrenamiento busca después
    df_monthly["year_month"] = df_monthly["CreateDate"].dt.to_period('M')
    df_monthly["date"] = df_monthly["CreateDate"] 

    # 2. ANTI-HIJACKING: Composite ID
    df_monthly["IdProduct_Unique"] = df_monthly["IdProduct"].astype(str) + "_" + df_monthly["NameProduct"]
    
    # 3. Time Features
    df_monthly["Month"] = df_monthly["CreateDate"].dt.month
    df_monthly["Year"] = df_monthly["CreateDate"].dt.year
    
    # 5. Transformations
    df_monthly["Quantity_log"] = np.log1p(df_monthly["Quantity"])
    df_monthly["UnitPrice_log"] = np.log1p(df_monthly["UnitPrice"])
    
    # 6. Lags & Rolling Features
    # Ensure sorted by date for correct shifting
    df_monthly = df_monthly.sort_values("date")
    
    # Define features group
    feature_group = ["IdCompany", "IdProduct_Unique", "IdBranchCompany", "IdWarehouse"]
    
    # Lags
    for lag in [1, 3, 6]:
        df_monthly[f"lag_{lag}"] = df_monthly.groupby(feature_group)["Quantity_log"].shift(lag)
        
    # Rolling Means
    # Note: Shift(1) ensures we don't use current month data for prediction (leakage prevention)
    for window in [3, 6, 12]:
        df_monthly[f"roll_{window}"] = df_monthly.groupby(feature_group)["Quantity_log"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    
    if not is_inference:
        df_monthly_clean = df_monthly.dropna().reset_index(drop=True)
    else:
        # We only care about rows that valid timestamps? 
        # Actually, let's just return it all and let the caller filter the future row.
        df_monthly_clean = df_monthly.reset_index(drop=True)
    
    return df_monthly_clean

def smoothed_target_encode(train, test, col, target, global_mean, m=15, smooth_map=None):

    if smooth_map is None:
        # Fit phase
        agg = train.groupby(col)[target].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']
        
        smooth = (counts * means + m * global_mean) / (counts + m)
    else:
        # Use provided map
        smooth = smooth_map
    
    # Transform Train
    train[col + "_te"] = train[col].map(smooth).fillna(global_mean)
    
    # Transform Test (if provided)
    if test is not None:
        test[col + "_te"] = test[col].map(smooth).fillna(global_mean)
        
    return train, test, smooth
