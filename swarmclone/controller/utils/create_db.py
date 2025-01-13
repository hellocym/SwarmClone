import sqlite3

def create_database():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            disabled BOOLEAN DEFAULT FALSE
        )
    ''')
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()