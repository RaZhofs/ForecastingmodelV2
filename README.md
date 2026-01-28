# 📈 Automated Sales Forecasting System

## Overview
This project is a production-grade forecasting pipeline designed to automate monthly sales predictions. It connects to a SQL database, processes historical sales data, and generates a "Shopping List" (purchase suggestions) for the upcoming month using a LightGBM machine learning model.

## Key Features
-   **🔮 Future Inference:** Automatically detects the last available date and predicts demand for the *next* month.
-   **⚡ Instant Predictions:** Persists model state and Target Encodings, allowing for instant inference without re-training.
-   **🛡️ Quality Gate:** Automatically checks model reliability ($R^2$ score) and warns if data volatility is high ($R^2 < 0.40$).
-   **🔌 Database Integration:** Connects directly to SQL Server to fetch live data.

## Setup

### 1. Prerequisites
-   Python 3.10+
-   ODBC Driver 17 for SQL Server

### 2. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory (or rename `.env.example`) with your database credentials:
```ini
DB_SERVER=your_server
DB_NAME=MyCoin
DB_USER=sa
DB_PASSWORD=your_password
USE_MOCK_DATA=False  # Set to True to test with CSV
```

## Usage

Run the main pipeline:
```bash
python main.py
```

### What Happens When You Run It?
1.  **Fetch:** Data is pulled from the DB (or CSV in mock mode).
2.  **Process:** Data is aggregated to monthly levels; features (lags, rolling means, encodings) are engineered.
3.  **Train/Load:**
    -   If `best_model.pkl` exists, it loads the model and encodings instantly.
    -   If not, it trains a new model using `GridSearchCV` for hyperparameter optimization and saves it.
4.  **Validate:** It tests the model on the most recent known data to verify accuracy ($R^2$, MAE).
5.  **Predict:** It generates the **Next Month Forecast**.

## Outputs

| File | Description |
| :--- | :--- |
| **`next_month_forecast.csv`** | 🎯 **The Goal.** Predicted demand for the upcoming month. Use this for inventory planning. |
| `forecast_results.csv` | Validation report comparing predictions vs. actuals for the test period. |
| `best_model.pkl` | The trained model artifact (includes Model + Target Encodings). |

## Project Structure
-   `main.py`: The pipeline orchestrator.
-   `forecasting.py`: Forecasting engine (LightGBM wrapper, GridSearch, Inference logic).
-   `processing.py`: Data cleaning and feature engineering (Lags, Rolling, Target Encoding).
-   `database.py`: SQL connection and data fetching logic.
-   `config.py`: Configuration constants.
