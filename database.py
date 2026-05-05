import sqlite3
import pandas as pd
from datetime import datetime, date, time
from typing import List, Dict

DB_PATH = "portfolio.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            nome TEXT,
            ticker TEXT,
            country TEXT,
            data_compra TEXT,
            hora_compra TEXT,
            preco_compra REAL,
            quantidade REAL
        )
    ''')
    conn.commit()
    conn.close()

def add_asset(asset_info: dict) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    
    # Format date and time
    dt_compra = asset_info.get('Data Compra')
    if isinstance(dt_compra, date):
        dt_compra_str = dt_compra.isoformat()
    else:
        dt_compra_str = str(dt_compra)
        
    hr_compra = asset_info.get('Hora Compra')
    if isinstance(hr_compra, time):
        hr_compra_str = hr_compra.isoformat()
    else:
        hr_compra_str = str(hr_compra)

    cursor.execute('''
        INSERT INTO portfolio (type, nome, ticker, country, data_compra, hora_compra, preco_compra, quantidade)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        asset_info.get('Type'),
        asset_info.get('Nome'),
        asset_info.get('Ticker'),
        asset_info.get('Country'),
        dt_compra_str,
        hr_compra_str,
        asset_info.get('Preço Compra'),
        asset_info.get('Quantidade')
    ))
    conn.commit()
    last_row_id = cursor.lastrowid
    conn.close()
    return last_row_id

def remove_asset(ticker: str):
    conn = get_connection()
    cursor = conn.cursor()
    # Removes all entries for this ticker, since it acts like a wallet summary. 
    # Can be adjusted if specific lots need to be removed.
    cursor.execute('DELETE FROM portfolio WHERE upper(ticker) = ?', (ticker.upper(),))
    conn.commit()
    conn.close()

def load_portfolio() -> List[dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM portfolio')
    rows = cursor.fetchall()
    conn.close()
    
    portfolio = []
    for row in rows:
        portfolio.append({
            'id': row['id'],
            'Type': row['type'],
            'Nome': row['nome'],
            'Ticker': row['ticker'],
            'Country': row['country'],
            'Data Compra': row['data_compra'],
            'Hora Compra': row['hora_compra'],
            'Preço Compra': row['preco_compra'],
            'Quantidade': row['quantidade']
        })
    return portfolio
