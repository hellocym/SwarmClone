import asyncio
import sqlite3
import random
import string
from datetime import datetime


async def generate_unique_api_key(db_connection):
    """
    生成唯一的API密钥。

    此函数通过随机选择字母和数字来生成一个64字符长度的API密钥
    并检查其在数据库中的唯一性。如果生成的API密钥已存在于数据库中
    则重复生成过程 直到生成唯一的API密钥。

    参数:
    db_connection: 数据库连接对象，用于执行数据库操作。

    返回:
    唯一的API密钥字符串。
    """
    while True:
        # 生成一个64字符长度的随机API密钥，包含字母和数字
        api_key = ''.join(random.choices(string.ascii_letters + string.digits, k=64))
        
        # 创建数据库游标
        cursor = db_connection.cursor()
        
        # 检查生成的API密钥是否已存在于数据库中
        cursor.execute("SELECT 1 FROM api_keys WHERE api_key = ?", (api_key,))
        
        # 如果数据库中不存在该API密钥，则返回它
        if cursor.fetchone() is None:
            return api_key


async def initialize_api_key():
    """
    异步初始化面板函数。
    
    该函数负责创建与数据库的连接，并设置必要的表格。
    它还会生成一个唯一的API密钥 并将其插入到数据库中。
    
    Returns:
        bool: 表示初始化是否成功的布尔值。成功时返回True 否则返回False。
    """
    # 创建数据库连接
    db_connection = sqlite3.connect('panel_data.db')
    
    try:
        # 创建游标对象
        cursor = db_connection.cursor()
        
        # 执行SQL语句以创建api_keys表（如果尚不存在）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_key TEXT UNIQUE NOT NULL,
                expiration_date TEXT NOT NULL
            )
        ''')
        
        # 提交数据库事务
        db_connection.commit()

        # 生成唯一的API密钥
        api_key = await generate_unique_api_key(db_connection)
        
        # 获取当前日期作为到期日
        expiration_date = datetime.now().strftime('%Y-%m-%d')
        
        # 将API密钥和到期日插入到api_keys表中
        cursor.execute("INSERT INTO api_keys (api_key, expiration_date) VALUES (?, ?)", (api_key, expiration_date))
        
        # 再次提交事务
        db_connection.commit()

        # 初始化成功
        api_key_initialized_success = True
        
    except sqlite3.Error as e:
        # 捕获sqlite3错误并打印错误信息
        print(f"An error occurred: {e}")
        # 初始化失败
        api_key_initialized_success = False
    finally:
        # 关闭数据库连接
        db_connection.close()

    # 返回初始化是否成功的标志
    return api_key_initialized_success, api_key, db_connection


async def is_api_key_expired(db_connection, api_key):
    """
    异步检查给定的API密钥是否过期。

    该函数通过查询数据库获取API密钥的过期日期 并将其与当前日期进行比较 以确定API密钥是否过期。

    参数:
    - db_connection: 数据库连接对象，用于执行数据库查询。
    - api_key: 要检查的API密钥字符串。

    返回:
    - 如果API密钥过期或未找到 返回True 否则返回False。
    """
    # 创建一个游标对象，用于执行SQL语句
    cursor = db_connection.cursor()
    # 执行SQL查询，获取API密钥的过期日期
    cursor.execute("SELECT expiration_date FROM api_keys WHERE api_key = ?", (api_key,))
    # 获取查询结果的第一行
    result = cursor.fetchone()
    # 如果查询结果为None，表示API密钥不存在，认为已过期
    if result is None:
        return True  # API密钥未找到，认为已过期
    # 将获取到的过期日期字符串转换为datetime对象
    expiration_date = datetime.strptime(result[0], '%Y-%m-%d')
    # 比较过期日期与当前日期，如果过期日期早于当前日期则返回True，否则返回False
    return expiration_date < datetime.now()
