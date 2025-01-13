import sys
import os
from pathlib import Path
from getpass import getpass
import tomlkit
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Boolean, Integer
from passlib.context import CryptContext

# 动态获取项目根目录
project_root = Path(__file__).resolve().parent.parent.parent

# 定义用户模型
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)

# 密码加密工具
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def main():
    # 加载配置文件
    config_path = project_root / "dist" / "server_config.toml"
    with open(config_path, 'r') as f:
        config = tomlkit.load(f)
    
    db_url = config['app']['database_url']
    
    # 创建数据库引擎和会话
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)  # 确保表存在
    session = Session(engine)
    
    try:
        # 获取用户输入
        username = input('Enter username: ')
        password = getpass('Enter password: ')
        confirm_password = getpass('Confirm password: ')
        
        # 检查密码是否匹配
        if password != confirm_password:
            print('Passwords do not match.')
            sys.exit(1)
        
        # 检查用户名是否已存在
        existing_user = session.query(User).filter_by(username=username).first()
        if existing_user:
            print('Username already exists.')
            sys.exit(1)
        
        # 加密密码（使用与后端相同的 bcrypt 加密算法）
        hashed_password = pwd_context.hash(password)
        
        # 创建新用户
        new_user = User(username=username, hashed_password=hashed_password, disabled=False)
        session.add(new_user)
        
        # 提交会话
        session.commit()
        
        print(f'User {username} created successfully.')
    except Exception as e:
        print(f'An error occurred: {e}')
        session.rollback()
    finally:
        session.close()

if __name__ == '__main__':
    main()