# server.py
import os
import json
import shutil
from datetime import datetime
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from passlib.context import CryptContext
from jose import jwt, JWTError
from dotenv import load_dotenv

# Local DB models / session
from database import SessionLocal, User, Message

load_dotenv()

SECRET_KEY = os.environ.get("SECRET_KEY", "replace-me-with-a-secure-secret")
ALGORITHM = "HS256"

app = FastAPI()

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure uploads folder exists and mount it
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Use bcrypt_sha256 to avoid bcrypt 72-byte limit (Passlib will pre-hash with sha256)
pwd_context = CryptContext(schemes=["bcrypt_sha256"], deprecated="auto")


def hash_password(p: str) -> str:
    # Defensive: ensure it's a string
    if p is None:
        p = ""
    return pwd_context.hash(str(p))


def verify_password(p: str, h: str) -> bool:
    try:
        return pwd_context.verify(str(p), h)
    except Exception:
        return False


def create_token(username: str) -> str:
    return jwt.encode({"username": username}, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# in-memory connection lists and recent messages for suggestions
auth_clients = {}        # username -> websocket
anon_clients = set()     # set of anonymous websockets

# keep recent world messages in memory (list of dicts: {name, text, ts})
MAX_RECENT = 200
recent_messages: List[dict] = []


def update_recent_messages(name: str, text: str):
    """
    Append a message record to recent_messages and trim to MAX_RECENT.
    Keep newest last (append).
    """
    if text is None:
        return
    recent_messages.append({"name": name, "text": text, "ts": datetime.utcnow().isoformat()})
    if len(recent_messages) > MAX_RECENT:
        # drop oldest
        del recent_messages[0: len(recent_messages) - MAX_RECENT]


# simple stop words / greetings to avoid in suggestions
COMMON_GREETINGS = {"hi", "hello", "hey", "hii", "hiya", "yo", "ok", "okay"}


def _clean_text_for_suggestion(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    return " ".join(s.split())  # collapse whitespace


def generate_suggestions(sender: str = None, limit: int = 3) -> List[str]:
    """
    Generate up to `limit` suggestion strings based on recent_messages.
    Prefer messages from other users. Skip very short messages and common greetings.
    Return newest distinct suggestions (most recent first).
    """
    suggestions = []
    seen = set()
    # iterate from newest to oldest
    for rec in reversed(recent_messages):
        if len(suggestions) >= limit:
            break
        author = rec.get("name")
        text = _clean_text_for_suggestion(rec.get("text", ""))
        if not text:
            continue
        # skip if it's the same user who just sent the message (we prefer other users' messages)
        if sender and author == sender:
            continue
        # normalize and skip trivial greetings or too short
        low = text.lower()
        if low in COMMON_GREETINGS:
            continue
        if len(text) < 3:
            continue
        if text in seen:
            continue
        seen.add(text)
        suggestions.append(text)
    # if not enough suggestions from other users, fallback to any recent messages (including sender)
    if len(suggestions) < limit:
        for rec in reversed(recent_messages):
            if len(suggestions) >= limit:
                break
            text = _clean_text_for_suggestion(rec.get("text", ""))
            if not text or text in seen:
                continue
            low = text.lower()
            if low in COMMON_GREETINGS or len(text) < 3:
                continue
            seen.add(text)
            suggestions.append(text)
    return suggestions[:limit]


# ---------------- helper broadcast / db helpers ----------------
async def broadcast_raw(text: str):
    """
    Broadcast a raw string (already JSON-stringified) to all connected clients.
    Clean up closed connections.
    """
    to_remove = []
    # auth clients
    for user, ws in list(auth_clients.items()):
        try:
            await ws.send_text(text)
        except Exception:
            to_remove.append(("auth", user))
    # anon clients
    for ws in list(anon_clients):
        try:
            await ws.send_text(text)
        except Exception:
            to_remove.append(("anon", ws))
    # cleanup
    for t, v in to_remove:
        if t == "auth" and v in auth_clients:
            del auth_clients[v]
        if t == "anon" and v in anon_clients:
            try:
                anon_clients.remove(v)
            except Exception:
                pass


def save_message_to_db(username: str, content: str):
    """
    Synchronous helper: open a short-lived session and save the message.
    """
    db = SessionLocal()
    try:
        msg = Message(username=username, content=content)
        db.add(msg)
        db.commit()
    except Exception as e:
        print("save_message_to_db error:", e)
        db.rollback()
    finally:
        db.close()


# ---------------- routes ----------------
@app.get("/")
def get_chat():
    # serve the main chat page
    return FileResponse("static/chat.html")


@app.post("/signup")
async def signup(data=Body(...)):
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")

    db = SessionLocal()
    try:
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")

        # hash (bcrypt_sha256 handles long input safely)
        hashed = hash_password(password)

        new_user = User(
            username=username,
            password=hashed
        )
        db.add(new_user)
        db.commit()
        return {"message": "Signup successful"}
    except HTTPException:
        raise
    except Exception as e:
        print("signup error:", e)
        db.rollback()
        # return the error text so you see it in dev; in prod you might hide it
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/login")
async def login(data=Body(...)):
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


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Basic file upload. In production sanitize filename and optionally use cloud storage.
    """
    filename = file.filename or "upload"
    base, ext = os.path.splitext(filename)
    safe_filename = filename
    counter = 1
    dest = os.path.join(UPLOAD_DIR, safe_filename)
    while os.path.exists(dest):
        safe_filename = f"{base}_{counter}{ext}"
        dest = os.path.join(UPLOAD_DIR, safe_filename)
        counter += 1

    try:
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        print("upload write error:", e)
        raise HTTPException(status_code=500, detail="Failed to save file")

    return {
        "url": f"/uploads/{os.path.basename(dest)}",
        "filename": os.path.basename(dest)
    }


# ---------------- anonymous readonly websocket ----------------
@app.websocket("/ws")
async def websocket_anonymous(websocket: WebSocket):
    await websocket.accept()
    anon_clients.add(websocket)
    print("Anonymous client connected. total anon:", len(anon_clients))
    # notify counts
    try:
        # broadcast user count to everyone (including this new anon)
        total_auth = len(auth_clients)
        total_anon = len(anon_clients)
        data = json.dumps({
            "type": "user_count",
            "auth": total_auth,
            "anon": total_anon,
            "total": total_auth + total_anon
        })
        await broadcast_raw(data)
    except Exception:
        pass

    try:
        while True:
            try:
                _ = await websocket.receive_text()
                # ignore anon input (read-only)
            except WebSocketDisconnect:
                break
            except Exception:
                break
    finally:
        try:
            if websocket in anon_clients:
                anon_clients.remove(websocket)
        except Exception:
            pass
        print("Anonymous client disconnected. total anon:", len(anon_clients))
        # update counts
        try:
            total_auth = len(auth_clients)
            total_anon = len(anon_clients)
            data = json.dumps({
                "type": "user_count",
                "auth": total_auth,
                "anon": total_anon,
                "total": total_auth + total_anon
            })
            await broadcast_raw(data)
        except Exception:
            pass


# ---------------- authenticated websocket ----------------
@app.websocket("/ws/{token}")
async def websocket_auth(websocket: WebSocket, token: str):
    # validate token
    payload = decode_token(token)
    if not payload or "username" not in payload:
        await websocket.close()
        return
    username = payload.get("username")

    try:
        await websocket.accept()
    except Exception:
        return

    # send existing users list BEFORE adding them to auth_clients
    try:
        for user in list(auth_clients.keys()):
            if user == username:
                continue
            data = json.dumps({
                "type": "join",
                "name": user,
                "color": None
            })
            try:
                await websocket.send_text(data)
            except Exception:
                pass
    except Exception:
        pass

    # register new authenticated client
    auth_clients[username] = websocket
    print(f"{username} connected (auth). total auth:", len(auth_clients))

    # announce join
    try:
        join_msg = json.dumps({"type": "join", "name": username, "color": None})
        await broadcast_raw(join_msg)
    except Exception:
        pass

    # update counts
    try:
        total_auth = len(auth_clients)
        total_anon = len(anon_clients)
        data = json.dumps({
            "type": "user_count",
            "auth": total_auth,
            "anon": total_anon,
            "total": total_auth + total_anon
        })
        await broadcast_raw(data)
    except Exception:
        pass

    try:
        while True:
            try:
                text = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                break

            # handle incoming text safely
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None

            try:
                if isinstance(parsed, dict) and parsed.get("type"):
                    t = parsed.get("type")

                    # ignore client-sent join/leave; server handles them
                    if t in ("join", "leave"):
                        continue

                    # message: save and broadcast
                    if t == "message":
                        mtext = parsed.get("text", "").strip()
                        author = parsed.get("name", username)
                        if mtext:
                            # update in-memory recent messages for suggestions
                            update_recent_messages(author, mtext)
                            # save to DB
                            try:
                                save_message_to_db(author, mtext)
                            except Exception as e:
                                print("error saving message:", e)
                            # broadcast original message unchanged
                            await broadcast_raw(json.dumps({
                                "type": "message",
                                "name": author,
                                "text": mtext
                            }))
                            # generate suggestions (prefer other users' messages)
                            try:
                                suggestions = generate_suggestions(sender=author, limit=3)
                                # broadcast suggestions to all clients
                                await broadcast_raw(json.dumps({
                                    "type": "suggestions",
                                    "suggestions": suggestions
                                }))
                            except Exception as e:
                                print("suggestion generation failed:", e)
                        continue

                    # file event: save a simple representation to DB and broadcast
                    if t == "file":
                        fname = parsed.get("filename") or ""
                        furl = parsed.get("url") or ""
                        author = parsed.get("name", username)
                        content = f"[file] {furl} | {fname}"
                        try:
                            save_message_to_db(author, content)
                        except Exception as e:
                            print("error saving file message:", e)
                        await broadcast_raw(json.dumps({
                            "type": "file",
                            "name": author,
                            "url": furl,
                            "filename": fname
                        }))
                        # update recent messages (so suggestions can be created from text messages later)
                        update_recent_messages(author, content)
                        continue

                    # typing event: broadcast to others
                    if t == "typing":
                        try:
                            await broadcast_raw(json.dumps({
                                "type": "typing",
                                "name": parsed.get("name", username)
                            }))
                        except Exception:
                            pass
                        continue

                    # unknown typed object -> just broadcast raw
                    await broadcast_raw(text)
                else:
                    # fallback: treat as plain message
                    plain = text
                    try:
                        save_message_to_db(username, plain)
                    except Exception as e:
                        print("error saving plain message:", e)
                    update_recent_messages(username, plain)
                    await broadcast_raw(json.dumps({"type": "message", "name": username, "text": plain}))
            except Exception as e:
                # don't crash websocket loop on any message error
                print("handling message error:", e)

    finally:
        # cleanup on disconnect
        try:
            if username in auth_clients and auth_clients[username] is websocket:
                del auth_clients[username]
        except Exception:
            pass
        print(f"{username} disconnected (auth). total auth:", len(auth_clients))
        # announce leave
        try:
            leave_msg = json.dumps({"type": "leave", "name": username})
            await broadcast_raw(leave_msg)
        except Exception:
            pass
        # broadcast updated counts
        try:
            total_auth = len(auth_clients)
            total_anon = len(anon_clients)
            data = json.dumps({
                "type": "user_count",
                "auth": total_auth,
                "anon": total_anon,
                "total": total_auth + total_anon
            })
            await broadcast_raw(data)
        except Exception:
            pass


# small debug endpoint to verify server is alive
@app.get("/ping")
def ping():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}


# run with uvicorn when executed directly (helps local dev and Render)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
