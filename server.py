# server.py
import os
import json
import shutil
import uuid
import logging
from datetime import datetime, timedelta
from typing import Generator, Optional

from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt, JWTError

# Import database artifacts from your database.py (must exist in repo)
# database.py should define: SessionLocal, engine, Base, User, Message
try:
    from database import SessionLocal, engine, Base, User, Message
except Exception as e:
    raise RuntimeError(
        "Failed to import from database.py. Ensure database.py provides SessionLocal, engine, Base, User, Message. Import error: "
        + str(e)
    )

# create tables (idempotent)
Base.metadata.create_all(bind=engine)

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signalmesh")

# password hashing (bcrypt backend). We'll handle 72-byte limitation.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT / auth settings
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week token by default

# static / uploads
STATIC_DIR = "static"
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# FastAPI app
app = FastAPI(title="Signal Mesh")

# mount static (chat.html, chat.js, style.css should be at ./static/)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- DB dependency ---
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- helpers ---
def _truncate_for_bcrypt(password: str) -> str:
    """
    Return a version safe for bcrypt (max 72 bytes).
    Two options:
      - truncate silently (used here), OR
      - raise an exception to ask frontend to send smaller password.
    We'll *reject* very long raw input to avoid accidental huge strings.
    """
    b = password.encode("utf-8")
    if len(b) > 72:
        # Option: raise HTTPException instead to force client-side change
        # Here we choose to reject so user sees clear message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password too long. Maximum 72 bytes are allowed.",
        )
    return password


def hash_password(password: str) -> str:
    pw = _truncate_for_bcrypt(password)
    return pwd_context.hash(pw)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(plain, hashed)
    except Exception:
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# --- REST endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/signup")
def signup(payload: dict, db: Session = Depends(get_db)):
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""

    if not username or not password:
        raise HTTPException(status_code=400, detail="username & password required")

    # if password too long, _truncate_for_bcrypt will raise a 400 HTTPException
    hashed = hash_password(password)

    new_user = User(username=username, password=hashed)
    db.add(new_user)
    try:
        db.commit()
        db.refresh(new_user)
    except IntegrityError as e:
        db.rollback()
        logger.info("Signup IntegrityError: %s", e)
        raise HTTPException(status_code=400, detail="user already exists")
    except Exception as e:
        db.rollback()
        logger.exception("Unexpected signup error")
        raise HTTPException(status_code=500, detail="internal server error")

    return {"id": new_user.id, "username": new_user.username}


@app.post("/login")
def login(payload: dict, db: Session = Depends(get_db)):
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""

    if not username or not password:
        raise HTTPException(status_code=400, detail="username & password required")

    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password):
        raise HTTPException(status_code=401, detail="invalid credentials")

    token = create_access_token({"sub": user.username})
    return {"token": token, "username": user.username}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accept a file upload and return JSON with url and filename for chat to share.
    Files are saved into ./static/uploads/<uuid>_<safe_filename>
    """
    filename = file.filename or "upload"
    # sanitize filename very simply
    safe_name = "".join(c for c in filename if c.isalnum() or c in (" ", ".", "_", "-")).strip()
    unique = f"{uuid.uuid4().hex}_{safe_name}"
    dest = os.path.join(UPLOADS_DIR, unique)

    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.exception("upload failed")
        raise HTTPException(status_code=500, detail="upload failed")

    url = f"/static/uploads/{unique}"
    return {"url": url, "filename": safe_name}


@app.get("/messages")
def get_messages(limit: int = 100, db: Session = Depends(get_db)):
    q = db.query(Message).order_by(Message.timestamp.desc()).limit(limit).all()
    return {
        "messages": [
            {
                "id": m.id,
                "username": m.username,
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
            }
            for m in reversed(q)
        ]
    }


# --- WebSocket chat infra ---
# Store connections
auth_clients: dict[str, WebSocket] = {}   # username -> websocket
anon_clients: list[WebSocket] = []        # list of anonymous websockets


async def broadcast_json(obj: dict):
    text = json.dumps(obj)
    to_remove_auth = []
    for uname, ws in list(auth_clients.items()):
        try:
            await ws.send_text(text)
        except Exception:
            to_remove_auth.append(uname)
    for uname in to_remove_auth:
        auth_clients.pop(uname, None)

    to_remove_anon = []
    for ws in list(anon_clients):
        try:
            await ws.send_text(text)
        except Exception:
            to_remove_anon.append(ws)
    for ws in to_remove_anon:
        try:
            anon_clients.remove(ws)
        except Exception:
            pass


async def broadcast_user_count():
    obj = {"type": "user_count", "auth": len(auth_clients), "total": len(auth_clients) + len(anon_clients)}
    await broadcast_json(obj)


# Anonymous WS path: /ws
@app.websocket("/ws")
async def websocket_anon(ws: WebSocket):
    await ws.accept()
    anon_clients.append(ws)
    try:
        # send basic welcome (optional)
        await broadcast_user_count()
        while True:
            data = await ws.receive_text()
            # anonymous messages are not allowed to post — show server message to guide
            # but we will forward raw strings to viewers as server: "Guest message"
            # If frontend expects raw colon-separated messages, keep simple behavior:
            try:
                # if frontend sends JSON, try forward as-is to auth clients
                json.loads(data)  # just check it's JSON
                await broadcast_json({"type": "raw", "data": data})
            except Exception:
                # plain text -> broadcast as server message
                await broadcast_json({"type": "message", "name": "guest", "text": data})
    except WebSocketDisconnect:
        try:
            anon_clients.remove(ws)
        except Exception:
            pass
        await broadcast_user_count()
    except Exception:
        logger.exception("anon ws error")
        try:
            anon_clients.remove(ws)
        except Exception:
            pass
        await broadcast_user_count()


# Auth WS path: /ws/{token}
@app.websocket("/ws/{token}")
async def websocket_auth(ws: WebSocket, token: str):
    # validate token
    try:
        payload = decode_access_token(token)
        username = payload.get("sub")
        if not username:
            await ws.close(code=1008)
            return
    except HTTPException:
        await ws.close(code=1008)
        return

    # accept and register
    await ws.accept()
    # if user already connected, replace old connection
    prev = auth_clients.get(username)
    if prev:
        try:
            await prev.close()
        except Exception:
            pass
    auth_clients[username] = ws

    # announce join
    await broadcast_json({"type": "join", "name": username})
    await broadcast_user_count()

    try:
        # send recent messages to this user? (client may call /messages instead)
        while True:
            raw = await ws.receive_text()
            # try parse JSON payload
            try:
                obj = json.loads(raw)
            except Exception:
                obj = {"type": "message", "text": raw, "name": username}

            # handle typing indicator
            t = obj.get("type")
            if t == "typing":
                await broadcast_json({"type": "typing", "name": username})
                continue
            elif t == "message":
                text = obj.get("text", "")[:2000]
                # persist into DB
                try:
                    db = SessionLocal()
                    msg = Message(username=username, content=text)
                    db.add(msg)
                    db.commit()
                    db.refresh(msg)
                except Exception:
                    db.rollback()
                    logger.exception("failed to save message")
                finally:
                    db.close()
                await broadcast_json({"type": "message", "name": username, "text": text})
                continue
            elif t == "file":
                # expect {type:"file", name:username, url:..., filename:...}
                await broadcast_json({"type": "file", "name": username, "url": obj.get("url"), "filename": obj.get("filename")})
                continue
            elif t == "raw":
                # forward raw server content
                await broadcast_json({"type": "raw", "data": obj.get("data")})
                continue
            else:
                # unknown -> echo as message
                await broadcast_json({"type": "message", "name": username, "text": obj.get("text", str(obj))})
    except WebSocketDisconnect:
        # remove this client
        try:
            if auth_clients.get(username) is ws:
                auth_clients.pop(username, None)
        except Exception:
            pass
        await broadcast_json({"type": "leave", "name": username})
        await broadcast_user_count()
    except Exception:
        logger.exception("auth ws error")
        try:
            if auth_clients.get(username) is ws:
                auth_clients.pop(username, None)
        except Exception:
            pass
        await broadcast_user_count()


# catch-all exception handler for clearer logs
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled exception for %s: %s", request.url, exc)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


# run with HOST/PORT env. Render provides $PORT
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
