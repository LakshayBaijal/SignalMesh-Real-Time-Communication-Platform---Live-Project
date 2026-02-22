# database.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

# Use a SQLite file in repo root (Render ephemeral filesystem — ok for testing)
DB_FILENAME = os.environ.get("CHAT_DB", "chat.db")
DATABASE_URL = f"sqlite:///{DB_FILENAME}"

# create engine (sqlite needs check_same_thread=False for multiple threads)
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    password = Column(String(256), nullable=False)  # store hashed password


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), index=True, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# IMPORTANT: ensure tables exist when module is imported
Base.metadata.create_all(bind=engine)
