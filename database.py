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
    DECLARE @CompanyID INT = {COMPANY_ID}; 
    
    SELECT 
        S.IdSale,
        S.SaleNumber,
        S.IdCompany,
        CONVERT(date, S.CreateDate) AS CreateDate,
        S.TotalSale,
        S.IdWarehouse,      
        S.IdBranchCompany,
        S.IdDocument,
        SD.IdSaleDetails,
        SD.IdProduct,
        SD.Quantity,
        SD.UnitPrice,
        SD.NameProduct
    FROM MyCoin.dbo.Sale S
    INNER JOIN MyCoin.dbo.SaleDetails SD ON S.IdSale = SD.IdSale
    WHERE S.IdCompany = @CompanyID 
      AND S.IdStatus = 19 
      -- FILTRO 1: Solo datos de los últimos 30 meses (desde 2023)
      AND S.CreateDate >= '2023-01-01'
      AND SD.IdProduct IN (
          SELECT SD_Sub.IdProduct
          FROM MyCoin.dbo.Sale S_Sub
          INNER JOIN MyCoin.dbo.SaleDetails SD_Sub ON S_Sub.IdSale = SD_Sub.IdSale
          WHERE S_Sub.IdCompany = @CompanyID 
            AND S_Sub.IdStatus = 19
            -- FILTRO 2: Los mismos 30 meses para la subconsulta
            AND S_Sub.CreateDate >= '2023-01-01'
          GROUP BY SD_Sub.IdProduct
          -- FILTRO 3: Mantenemos tu regla de los 6 meses de historia real
          HAVING COUNT(DISTINCT DATEFROMPARTS(YEAR(S_Sub.CreateDate), MONTH(S_Sub.CreateDate), 1)) >= 6
      )
    ORDER BY S.IdSale, SD.IdSaleDetails;
    """
    
    conn = get_db_connection()
    try:
        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()
