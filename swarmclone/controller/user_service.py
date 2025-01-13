from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Boolean
from pydantic_settings import BaseSettings
import tomlkit as toml
from pathlib import Path
from .settings import db_settings

class UserSettings(BaseSettings):
    database_url: str

    @classmethod
    def from_toml(cls, toml_path: str):
        # 使用 open() 打开文件并读取配置
        print(f"[user_service.py]Loading configuration from: {toml_path}")
        with open(toml_path, "r", encoding="utf-8") as f:
            config = toml.load(f)
        # 只提取 database_url
        database_url = config["app"]["database_url"]
        print(f"[user_service.py]Loaded database URL: {database_url}")
        return cls(database_url=database_url)

# 加载配置文件
config_path = Path(__file__).parent.parent / "dist" / "server_config.toml"
print(f"[user_service.py]Config path: {config_path}")
user_settings = UserSettings.from_toml(config_path)

# 创建数据库引擎
print("[user_service.py]Creating database engine...")
engine = create_engine(user_settings.database_url)
# 定义基类
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, disabled={self.disabled})>"

# 创建数据库表
print("[user_service.py]Creating database tables...")
Base.metadata.create_all(bind=engine)

# 定义 SessionLocal
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    print("[user_service.py]Creating a new database session...")
    db = SessionLocal()
    try:
        yield db
    finally:
        print("[user_service.py]Closing database session...")
        db.close()

class UserService:
    def __init__(self, db: Session):
        print("[user_service.py]Initializing UserService...")
        self.db = db

    def get_user(self, username: str):
        print(f"[user_service.py]Fetching user with username: {username}")
        user = self.db.query(User).filter(User.username == username).first()
        if user:
            print(f"[user_service.py]User found: {user}")
        else:
            print(f"[user_service.py]No user found with username: {username}")
        return user