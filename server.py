# server.py
import os
import json
import shutil
import asyncio
from typing import List, Tuple, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from passlib.context import CryptContext
from jose import jwt
from dotenv import load_dotenv

# Optional SDKs - guarded (won't crash import if missing)
try:
    from groq import Groq
except Exception:
    Groq = None

# numpy may be used if you later enable embeddings locally. It's optional.
try:
    import numpy as np
except Exception:
    np = None

# database (make sure database.py is in same folder)
from database import SessionLocal, User, Message

load_dotenv()

# ------------- config -------------
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
ALGORITHM = "HS256"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

MAX_RECENT = 200
TOP_K = 3
AI_CONTEXT_LINES = 8
MAX_SUGGESTION_WORDS = 12
UPLOAD_DIR = "uploads"

# ------------- app init -------------
app = FastAPI()
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ------------- state -------------
auth_clients = {}         # username -> websocket
anon_clients = set()      # set of anonymous websockets

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

recent_messages: List[Tuple[str, str]] = []     # list of (username, text)
recent_embeddings: List = []                    # kept if you enable local embeddings later
current_suggestions: List[str] = []             # last broadcast suggestions

# ------------- optional Groq client -------------
groq_client: Optional[object] = None
if Groq is not None and GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("Groq client initialized")
    except Exception as e:
        groq_client = None
        print("Warning: Groq init failed:", e)
else:
    print("Groq not configured or SDK missing; suggestions fallback to similarity (if available).")

# ------------- helpers -------------
def hash_password(p: str) -> str:
    return pwd_context.hash(p)

def verify_password(p: str, h: str) -> bool:
    return pwd_context.verify(p, h)

def create_token(username: str) -> str:
    return jwt.encode({"username": username}, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

async def broadcast_raw(text: str):
    to_remove = []
    for user, ws in list(auth_clients.items()):
        try:
            await ws.send_text(text)
        except:
            to_remove.append(("auth", user))
    for ws in list(anon_clients):
        try:
            await ws.send_text(text)
        except:
            to_remove.append(("anon", ws))
    for t, v in to_remove:
        if t == "auth" and v in auth_clients:
            del auth_clients[v]
        if t == "anon" and v in anon_clients:
            anon_clients.remove(v)

def wrap_message(username, message_text):
    return json.dumps({"type":"message","name": username, "text": message_text})

async def broadcast_user_count():
    data = json.dumps({
        "type": "user_count",
        "auth": len(auth_clients),
        "anon": len(anon_clients),
        "total": len(auth_clients) + len(anon_clients)
    })
    await broadcast_raw(data)

async def send_existing_users_to_socket(websocket: WebSocket, exclude_username=None):
    for user in list(auth_clients.keys()):
        if exclude_username and user == exclude_username:
            continue
        try:
            await websocket.send_text(json.dumps({"type":"join","name": user, "color": None}))
        except:
            pass

def save_message_to_db(username: str, content: str):
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

# ------------- embeddings helpers (optional) -------------
def cosine_similarity(a, b) -> float:
    if np is None:
        return 0.0
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

def add_to_recent(username: str, text: str, embedding=None):
    recent_messages.append((username, text))
    if embedding is not None:
        recent_embeddings.append(embedding)
    if len(recent_messages) > MAX_RECENT:
        recent_messages.pop(0)
        if recent_embeddings:
            recent_embeddings.pop(0)

def find_top_k_similar_excluding(embedding, exclude_user: str, exclude_text: str, k: int = TOP_K) -> List[str]:
    if not recent_embeddings or np is None:
        return []
    sims = []
    for idx, emb in enumerate(recent_embeddings):
        sim = cosine_similarity(embedding, emb)
        sims.append((idx, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    out = []
    for idx, sim in sims:
        candidate_user, candidate_text = recent_messages[idx]
        if candidate_user == exclude_user:
            continue
        if candidate_text.strip().lower() == exclude_text.strip().lower():
            continue
        if candidate_text not in out:
            out.append(candidate_text)
        if len(out) >= k:
            break
    return out

# ------------- Groq suggestion generator -------------
async def generate_ai_suggestions_groq(message: str, recent_msgs: List[Tuple[str, str]], exclude_user: Optional[str] = None) -> List[str]:
    if groq_client is None:
        return []
    context_msgs = [ (u,t) for (u,t) in recent_msgs if u != exclude_user ]
    context_lines = []
    for user, text in context_msgs[-AI_CONTEXT_LINES:]:
        t = text if len(text) <= 300 else text[:300] + "..."
        context_lines.append(f"{user}: {t}")
    prompt = f"""
You are a concise smart-reply assistant. Given the conversation context (excluding the message sender) and the latest user message,
produce exactly {TOP_K} short conversational reply suggestions, one per line. Return replies only; no JSON or numbering.

Context (most recent first):
{chr(10).join(context_lines)}

Latest message:
{message}

Rules:
- 1-8 words each (prefer 2-5)
- Friendly, natural replies
- Each reply on its own line
"""
    try:
        loop = asyncio.get_event_loop()
        def call_groq():
            return groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a concise smart-reply generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=60
            )
        response = await loop.run_in_executor(None, call_groq)
        try:
            text_out = response.choices[0].message.content
        except Exception:
            try:
                text_out = response["choices"][0]["message"]["content"]
            except Exception as e:
                print("generate_ai_suggestions_groq: extract failed:", e)
                return []
        suggestions = []
        for line in text_out.splitlines():
            line = line.strip()
            if not line:
                continue
            if len(line) > 2 and line[0].isdigit() and line[1] == '.':
                parts = line.split('. ', 1)
                if len(parts) > 1:
                    line = parts[1].strip()
            if line.startswith(("-", "*")):
                line = line[1:].strip()
            if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
                line = line[1:-1].strip()
            if not line:
                continue
            words = line.split()
            if len(words) > MAX_SUGGESTION_WORDS:
                line = " ".join(words[:MAX_SUGGESTION_WORDS]) + "..."
            if line not in suggestions:
                suggestions.append(line)
            if len(suggestions) >= TOP_K:
                break
        return suggestions[:TOP_K]
    except Exception as e:
        print("Groq suggestion error:", e)
        return []

# ------------- debug endpoint -------------
@app.post("/debug_ai")
async def debug_ai(payload=Body(...)):
    msg = None
    if isinstance(payload, dict):
        msg = payload.get("message")
    if not msg:
        raise HTTPException(status_code=400, detail="message required")
    suggestions = []
    try:
        suggestions = await generate_ai_suggestions_groq(msg, recent_messages, exclude_user=None)
    except Exception as e:
        print("debug_ai groq failed:", e)
        suggestions = []
    return {"message": msg, "suggestions": suggestions, "recent_sample": recent_messages[-10:], "current_suggestions": current_suggestions}

# ------------- routes & websockets -------------
@app.get("/")
def get_chat():
    return FileResponse("static/chat.html")

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
        new_user = User(username=username, password=hash_password(password))
        db.add(new_user)
        db.commit()
        return {"message": "Signup successful"}
    finally:
        db.close()

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

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    base, ext = os.path.splitext(filename)
    safe_path = os.path.join(UPLOAD_DIR, filename)
    counter = 1
    while os.path.exists(safe_path):
        safe_path = os.path.join(UPLOAD_DIR, f"{base}_{counter}{ext}")
        counter += 1
    with open(safe_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"url": f"/uploads/{os.path.basename(safe_path)}", "filename": os.path.basename(safe_path)}

@app.websocket("/ws")
async def websocket_anonymous(websocket: WebSocket):
    await websocket.accept()
    anon_clients.add(websocket)
    await broadcast_user_count()
    try:
        while True:
            try:
                _ = await websocket.receive_text()
                # ignore any anonymous input (readonly)
            except WebSocketDisconnect:
                break
            except Exception:
                break
    finally:
        if websocket in anon_clients:
            anon_clients.remove(websocket)
        await broadcast_user_count()

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
    try:
        await websocket.accept()
    except Exception:
        return
    try:
        await send_existing_users_to_socket(websocket, exclude_username=username)
    except Exception:
        pass
    auth_clients[username] = websocket
    try:
        await broadcast_raw(json.dumps({"type":"join","name": username, "color": None}))
    except:
        pass
    try:
        await broadcast_user_count()
        if current_suggestions:
            await websocket.send_text(json.dumps({"type":"suggestions","suggestions": current_suggestions}))
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
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and parsed.get("type"):
                    t = parsed.get("type")
                    if t in ("join", "leave"):
                        continue
                    if t == "message":
                        mtext = parsed.get("text", "")
                        display_name = parsed.get("name", username)
                        try:
                            save_message_to_db(display_name, mtext)
                        except Exception as e:
                            print("save db error:", e)
                        await broadcast_raw(text)
                        # generate suggestions BEFORE adding this message to recent buffer
                        embedding = None
                        suggestions: List[str] = []
                        # primary: Groq (exclude sender)
                        try:
                            suggestions = await generate_ai_suggestions_groq(mtext, recent_messages, exclude_user=display_name)
                        except Exception as e:
                            print("Groq generation failed:", e)
                            suggestions = []
                        # fallback: similarity (if embeddings available)
                        if not suggestions and embedding is not None:
                            try:
                                suggestions = find_top_k_similar_excluding(embedding, exclude_user=display_name, exclude_text=mtext, k=TOP_K)
                            except Exception as e:
                                print("similarity fallback error:", e)
                                suggestions = []
                        # finally add current message to recent buffer
                        add_to_recent(display_name, mtext, embedding)
                        # broadcast suggestions if any (persist globally)
                        if suggestions:
                            current_suggestions.clear()
                            current_suggestions.extend([s.strip() for s in suggestions[:TOP_K] if s and s.strip()])
                            try:
                                await broadcast_raw(json.dumps({"type":"suggestions","suggestions": current_suggestions}))
                            except Exception as e:
                                print("broadcast suggestions failed:", e)
                        continue
                    if t == "file":
                        fname = parsed.get("filename") or ""
                        furl = parsed.get("url") or ""
                        content = f"[file] {furl} | {fname}"
                        try:
                            save_message_to_db(parsed.get("name", username), content)
                        except Exception as e:
                            print("error saving file message:", e)
                        await broadcast_raw(text)
                        continue
                    await broadcast_raw(text)
                else:
                    # treat plain string as message
                    try:
                        save_message_to_db(username, text)
                    except Exception as e:
                        print("error saving plain message:", e)
                    await broadcast_raw(wrap_message(username, text))
            except json.JSONDecodeError:
                try:
                    save_message_to_db(username, text)
                except Exception as e:
                    print("error saving plain message:", e)
                await broadcast_raw(wrap_message(username, text))
    finally:
        if username in auth_clients and auth_clients[username] is websocket:
            del auth_clients[username]
        try:
            await broadcast_raw(json.dumps({"type":"leave","name": username}))
        except:
            pass
        await broadcast_user_count()

@app.get("/health")
def health():
    return {"status": "ok"}
