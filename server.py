# server.py
import os
import logging
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from typing import Generator

# Try to import DB pieces from your database.py (this file should exist in repo)
# It should provide SessionLocal, engine, Base, User, Message
try:
    from database import SessionLocal, engine, Base, User, Message  # file: database.py
except Exception as e:
    raise RuntimeError(
        "Failed to import from database.py. Make sure database.py exists and exports "
        "SessionLocal, engine, Base, User, Message. Import error: " + str(e)
    )

# Create tables on startup (safe both locally & on Render)
Base.metadata.create_all(bind=engine)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signalmesh")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# FastAPI app
app = FastAPI(title="Signal Mesh")

# Serve static files (make sure your chat.html, chat.js, style.css are in ./static/)
if not os.path.isdir("static"):
    # In case user kept static files at repo root, create a hint log.
    logger.info("No 'static' directory found — ensure your static files are in ./static/")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Dependency to get DB session
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Health check
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# Serve the chat UI (expects static/chat.html in repo)
@app.get("/", response_class=FileResponse)
def index():
    index_path = os.path.join("static", "chat.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    # fallback: minimal response
    return JSONResponse({"msg": "Place chat.html in ./static/ and redeploy."}, status_code=200)

# Signup route (expects JSON: { "username": "...", "password": "..." })
@app.post("/signup")
def signup(payload: dict, db: Session = Depends(get_db)):
    """
    Signup endpoint:
      - Expects JSON body with 'username' and 'password'
      - Returns 201 on success
      - Returns 400 if user exists or input invalid
    """
    username = payload.get("username")
    password = payload.get("password")

    if not username or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="username & password required")

    # check existing
    existing = db.query(User).filter(User.username == username).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="user already exists")

    hashed = hash_password(password)
    user = User(username=username, password=hashed)
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError as e:
        db.rollback()
        # This covers race conditions / unique constraint
        logger.warning("IntegrityError on signup: %s", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="user already exists")
    except Exception as e:
        db.rollback()
        logger.exception("Unexpected error during signup")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="internal server error")

    return {"id": user.id, "username": user.username}

# Simple login endpoint (returns basic JSON; adapt to tokens if you use JWT)
@app.post("/login")
def login(payload: dict, db: Session = Depends(get_db)):
    username = payload.get("username")
    password = payload.get("password")

    if not username or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="username & password required")

    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials")

    # Return minimal user info; replace with JWT if needed
    return {"id": user.id, "username": user.username, "msg": "login successful"}

# Message endpoints (simple example that fits a chat UI)
@app.get("/messages")
def get_messages(limit: int = 100, db: Session = Depends(get_db)):
    q = db.query(Message).order_by(Message.timestamp.desc()).limit(limit).all()
    # return messages newest-first reversed for client convenience
    return {"messages": [ {"id": m.id, "username": m.username, "content": m.content, "timestamp": m.timestamp.isoformat()} for m in reversed(q) ]}

@app.post("/messages")
def post_message(payload: dict, db: Session = Depends(get_db)):
    username = payload.get("username", "anonymous")
    content = payload.get("content", "")
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="empty message")

    msg = Message(username=username, content=content)
    db.add(msg)
    try:
        db.commit()
        db.refresh(msg)
    except Exception:
        db.rollback()
        logger.exception("Failed to write message")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="failed to save message")

    return {"id": msg.id, "username": msg.username, "content": msg.content, "timestamp": msg.timestamp.isoformat()}


# Generic exception handler to make logs clearer in Render
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception for request %s: %s", request.url, exc)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})

# Uvicorn startup: Render exposes $PORT; bind to it.
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    # Important: host 0.0.0.0 so Render can route in
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
