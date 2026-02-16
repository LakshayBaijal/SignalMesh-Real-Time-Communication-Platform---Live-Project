# server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from passlib.context import CryptContext
from jose import jwt
import json, os, shutil

# SQLite DB (from database.py)
from database import SessionLocal, User, Message
import os
from dotenv import load_dotenv

load_dotenv()
# SECRET - change to a secure env var in production
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ensure uploads folder and serve it
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# store connections
auth_clients = {}        # username -> websocket
anon_clients = set()     # set of anonymous websockets

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------- helper functions ----------------
def hash_password(p: str) -> str:
    return pwd_context.hash(p)

def verify_password(p: str, h: str) -> bool:
    return pwd_context.verify(p, h)

def create_token(username: str) -> str:
    return jwt.encode({"username": username}, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

async def broadcast_raw(text: str):
    """
    Broadcast a raw string to all connected clients (anon + auth).
    """
    to_remove = []
    # auth clients
    for user, ws in list(auth_clients.items()):
        try:
            await ws.send_text(text)
        except:
            to_remove.append(("auth", user))
    # anon clients
    for ws in list(anon_clients):
        try:
            await ws.send_text(text)
        except:
            to_remove.append(("anon", ws))
    # cleanup
    for t, v in to_remove:
        if t == "auth" and v in auth_clients:
            del auth_clients[v]
        if t == "anon" and v in anon_clients:
            anon_clients.remove(v)

def wrap_message(username, message_text):
    """
    Standard JSON message when server needs to wrap plain text messages.
    """
    return json.dumps({"type":"message","name": username, "text": message_text})

async def broadcast_user_count():
    total_auth = len(auth_clients)
    total_anon = len(anon_clients)
    data = json.dumps({
        "type": "user_count",
        "auth": total_auth,
        "anon": total_anon,
        "total": total_auth + total_anon
    })
    await broadcast_raw(data)

async def send_existing_users_to_socket(websocket: WebSocket, exclude_username=None):
    """
    Send 'join' messages (one per already-online user) to the provided websocket.
    exclude_username: skip sending that username (typically the new user)
    """
    for user in list(auth_clients.keys()):
        if exclude_username and user == exclude_username:
            continue
        data = json.dumps({
            "type": "join",
            "name": user,
            "color": None
        })
        try:
            await websocket.send_text(data)
        except:
            pass

# ---------------- database persistence helper ----------------
def save_message_to_db(username: str, content: str):
    """
    Synchronous helper — opens a short-lived SQLAlchemy session,
    saves message, commits, and closes.
    """
    db = SessionLocal()
    try:
        msg = Message(username=username, content=content)
        db.add(msg)
        db.commit()
    except Exception as e:
        # don't let DB errors crash websocket loop; just log
        print("save_message_to_db error:", e)
        db.rollback()
    finally:
        db.close()

# ---------------- routes ----------------
@app.get("/")
def get_chat():
    return FileResponse("static/chat.html")

# ---------------- signup (SQLite) ----------------
@app.post("/signup")
def signup(data=Body(...)):
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")

    db = SessionLocal()
    try:
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")

        new_user = User(
            username=username,
            password=hash_password(password)
        )
        db.add(new_user)
        db.commit()
        return {"message": "Signup successful"}
    finally:
        db.close()

# ---------------- login (SQLite) ----------------
@app.post("/login")
def login(data=Body(...)):
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            raise HTTPException(status_code=400, detail="User not found")
        if not verify_password(password, user.password):
            raise HTTPException(status_code=400, detail="Wrong password")

        token = create_token(username)
        return {"token": token}
    finally:
        db.close()

# ---------------- file upload endpoint ----------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # basic filename handling - in production sanitize or generate unique names
    filename = file.filename
    base, ext = os.path.splitext(filename)
    safe_path = os.path.join(UPLOAD_DIR, filename)
    counter = 1
    while os.path.exists(safe_path):
        safe_path = os.path.join(UPLOAD_DIR, f"{base}_{counter}{ext}")
        counter += 1

    with open(safe_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "url": f"/uploads/{os.path.basename(safe_path)}",
        "filename": os.path.basename(safe_path)
    }

# ---------------- anonymous readonly websocket ----------------
@app.websocket("/ws")
async def websocket_anonymous(websocket: WebSocket):
    await websocket.accept()
    anon_clients.add(websocket)
    print("Anonymous client connected. total anon:", len(anon_clients))
    # notify clients about counts
    await broadcast_user_count()
    try:
        # keep connection alive; if anon sends data, we ignore it (read-only)
        while True:
            try:
                _ = await websocket.receive_text()
                # ignore any data from anonymous clients
            except WebSocketDisconnect:
                break
            except Exception:
                break
    finally:
        if websocket in anon_clients:
            anon_clients.remove(websocket)
        print("Anonymous client disconnected. total anon:", len(anon_clients))
        await broadcast_user_count()

# ---------------- authenticated websocket ----------------
@app.websocket("/ws/{token}")
async def websocket_auth(websocket: WebSocket, token: str):
    # validate token
    try:
        payload = decode_token(token)
        username = payload.get("username")
        if not username:
            await websocket.close()
            return
    except Exception:
        await websocket.close()
        return

    # Accept connection
    try:
        await websocket.accept()
    except Exception:
        return

    # send existing users list to new socket BEFORE adding them to auth_clients
    try:
        await send_existing_users_to_socket(websocket, exclude_username=username)
    except Exception:
        pass

    # Now register the new authenticated client
    auth_clients[username] = websocket
    print(f"{username} connected (auth). total auth:", len(auth_clients))

    # announce join to everyone
    try:
        join_msg = json.dumps({"type":"join","name": username, "color": None})
        await broadcast_raw(join_msg)
    except Exception:
        pass

    # update counts for all
    try:
        await broadcast_user_count()
    except Exception:
        pass

    try:
        while True:
            try:
                text = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except RuntimeError:
                break
            except Exception:
                break

            # process text
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and parsed.get("type"):
                    t = parsed.get("type")
                    # ignore client-sent join/leave; server handles them
                    if t in ("join", "leave"):
                        continue

                    # If it's a message event, save to DB
                    if t == "message":
                        mtext = parsed.get("text", "")
                        # save synchronously (quick commit)
                        try:
                            save_message_to_db(parsed.get("name", username), mtext)
                        except Exception as e:
                            print("error saving message:", e)

                        await broadcast_raw(text)
                        continue

                    # If it's a file event, save a simple representation to DB
                    if t == "file":
                        fname = parsed.get("filename") or ""
                        furl = parsed.get("url") or ""
                        # save as "[file] url|filename" to preserve info
                        content = f"[file] {furl} | {fname}"
                        try:
                            save_message_to_db(parsed.get("name", username), content)
                        except Exception as e:
                            print("error saving file message:", e)

                        await broadcast_raw(text)
                        continue

                    # typing or other events just broadcast (no DB save)
                    await broadcast_raw(text)
                else:
                    # fallback: not a typed object -> treat as plain message & save
                    plain = text
                    try:
                        save_message_to_db(username, plain)
                    except Exception as e:
                        print("error saving plain message:", e)
                    await broadcast_raw(wrap_message(username, text))
            except json.JSONDecodeError:
                # not JSON -> treat as plain chat message & save
                try:
                    save_message_to_db(username, text)
                except Exception as e:
                    print("error saving plain message:", e)
                await broadcast_raw(wrap_message(username, text))

    finally:
        if username in auth_clients and auth_clients[username] is websocket:
            del auth_clients[username]
        print(f"{username} disconnected (auth). total auth:", len(auth_clients))
        try:
            leave_msg = json.dumps({"type":"leave","name": username})
            await broadcast_raw(leave_msg)
        except Exception:
            pass
        try:
            await broadcast_user_count()
        except Exception:
            pass
