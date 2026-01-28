import os
from dotenv import load_dotenv

load_dotenv()

# Database Configuration
DB_SERVER = os.getenv("DB_SERVER", "localhost")
DB_NAME = os.getenv("DB_NAME", "MyCoin")
DB_USER = os.getenv("DB_USER", "sa")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_DRIVER = os.getenv("DB_DRIVER", "{ODBC Driver 17 for SQL Server}")

# Feature Flag for Testing without DB
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "False").lower() == "true"

# Model Configuration
MODEL_PATH = "best_model.pkl"
COMPANY_ID = 150
SPLIT_DATE = "2025-01-01"
