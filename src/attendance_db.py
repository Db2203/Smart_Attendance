# attendance_db.py

import sqlite3
from config import DATABASE

def init_db():
    """Initialize the attendance database and create the table if it doesn't exist."""
    conn = sqlite3.connect(DATABASE['db_file'])
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reg_no TEXT,
            name TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def insert_attendance(reg_no, name):
    """Insert an attendance record for a recognized student."""
    conn = sqlite3.connect(DATABASE['db_file'])
    c = conn.cursor()
    c.execute('''
        INSERT INTO attendance (reg_no, name) VALUES (?, ?)
    ''', (reg_no, name))
    conn.commit()
    conn.close()

def fetch_attendance():
    """Fetch all attendance records."""
    conn = sqlite3.connect(DATABASE['db_file'])
    c = conn.cursor()
    c.execute('SELECT * FROM attendance')
    rows = c.fetchall()
    conn.close()
    return rows
