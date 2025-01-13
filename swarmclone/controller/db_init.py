import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# 创建用户表
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    password TEXT NOT NULL,
    public_key BLOB
)
''')

# 插入admin用户
cursor.execute("INSERT INTO users (username, password, public_key) VALUES (?, ?, ?)", ('admin', '123456', b''))
conn.commit()
conn.close()