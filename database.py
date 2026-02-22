# database.py  (replace your current file)
import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# Use environment DATABASE_URL if present (Render will provide this for managed Postgres)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./signalmesh.db")

# If using sqlite local file, keep check_same_thread
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )
else:
    # PostgreSQL / other drivers: do not pass sqlite-only args.
    # pool_pre_ping helps keep connections healthy on cloud DBs.
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        echo=False,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Users table
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)

# Messages table
class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create database and tables automatically (safe for both sqlite and postgres)
Base.metadata.create_all(bind=engine)
