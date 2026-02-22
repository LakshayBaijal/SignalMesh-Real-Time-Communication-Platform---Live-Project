# database.py
import os
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DB_PATH = os.environ.get("DATABASE_URL")  # recommended to set in Render
if not DB_PATH:
    DB_PATH = "sqlite:///./world_chat.db"

# if SQLite, ensure check_same_thread=False
if DB_PATH.startswith("sqlite"):
    engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DB_PATH)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), unique=True, index=True, nullable=False)
    password = Column(String(512), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), index=True, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# create tables if module imported standalone
Base.metadata.create_all(bind=engine)
