import pyodbc
import pandas as pd
from config import DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD, DB_DRIVER, COMPANY_ID, USE_MOCK_DATA

def get_db_connection():
    """Establishes a connection to the SQL Server database."""
    conn_str = (
        f"DRIVER={DB_DRIVER};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_NAME};"
        f"UID={DB_USER};"
        f"PWD={DB_PASSWORD}"
    )
    return pyodbc.connect(conn_str)

def fetch_sales_data():
    """Fetches sales data using SQL or from a local CSV if using mock mode."""
    
    if USE_MOCK_DATA:
        print("MOCK MODE: Loading data from 'DbSalesModified.csv'")
        # Loading local CSV to simulate DB fetch
        # We ensure columns align with what the pipeline expects
        try:
            df = pd.read_csv("DbSalesModified.csv")
            # The CSV might already be somewhat processed or raw. 
            # Based on previous view, it has columns like 'CreateDate', 'Quantity', 'UnitPrice', 'IdCompany' etc.
            # We just return it. The processing logic will handle formatting.
            return df
        except FileNotFoundError:
            raise Exception("Mock data file 'DbSalesModified.csv' not found.")

    """Queries the database."""
    query = f"""
    DECLARE @CompanyID INT = 150;
DECLARE @CreateDate DATE = '2023-01-01'

SELECT 
    -- 1. Agrupamos por el primer día del mes (Equivale a year_month en Python)
    DATEFROMPARTS(YEAR(S.CreateDate), MONTH(S.CreateDate), 1) AS CreateDate,
    S.IdCompany,
    S.IdWarehouse,      
    S.IdBranchCompany,
    SD.IdProduct,
    -- Traemos el nombre limpio para que Python cree el ID Único sin ruido
    LTRIM(RTRIM(LOWER(REPLACE(REPLACE(REPLACE(SD.NameProduct, '  ', ' '), '  ', ' '), '  ', ' ')))) AS NameProduct,
    
    -- 2. Métricas Agregadas: SQL lo hace mucho más rápido que Pandas en grandes volúmenes
    SUM(SD.Quantity) AS Quantity,
    AVG(SD.UnitPrice) AS UnitPrice
    
FROM MyCoin.dbo.Sale S
INNER JOIN MyCoin.dbo.SaleDetails SD ON S.IdSale = SD.IdSale
WHERE S.IdCompany = @CompanyID 
  AND SD.UnitPrice > 1 
  AND S.IdStatus = 19 
  AND S.CreateDate >= @CreateDate
  AND SD.IdProduct IN (
      SELECT SD_Sub.IdProduct
      FROM MyCoin.dbo.Sale S_Sub
      INNER JOIN MyCoin.dbo.SaleDetails SD_Sub ON S_Sub.IdSale = SD_Sub.IdSale
      WHERE S_Sub.IdCompany = @CompanyID 
        AND S_Sub.IdStatus = 19
        AND S_Sub.CreateDate >= @CreateDate
        AND SD_Sub.UnitPrice > 1
      GROUP BY SD_Sub.IdProduct
      HAVING COUNT(DISTINCT DATEFROMPARTS(YEAR(S_Sub.CreateDate), MONTH(S_Sub.CreateDate), 1)) >= 6
  )
GROUP BY 
    DATEFROMPARTS(YEAR(S.CreateDate), MONTH(S.CreateDate), 1),
    S.IdCompany,
    S.IdWarehouse,
    S.IdBranchCompany,
    SD.IdProduct,
    SD.NameProduct -- Agrupamos por nombre para mantener la integridad
ORDER BY CreateDate DESC;
    """
    
    conn = get_db_connection()
    try:
        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()
