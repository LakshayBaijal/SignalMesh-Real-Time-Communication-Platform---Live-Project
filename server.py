# server.py
import os
import json
import shutil
import hashlib
from datetime import datetime
from typing import List, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from passlib.context import CryptContext
from jose import jwt, JWTError
from dotenv import load_dotenv

# Optional AI/embedding imports (only used if installed)
try:
    from groq import Groq
except Exception:
    Groq = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

import asyncio
import numpy as np

# Local DB models / session
from database import SessionLocal, User, Message  # uses your existing database.py schema

load_dotenv()

# ----------------- config -----------------
SECRET_KEY = os.environ.get("SECRET_KEY", "replace-me-with-a-secure-secret")
ALGORITHM = "HS256"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# suggestion / embedding params
MAX_RECENT = int(os.environ.get("MAX_RECENT", 200))
TOP_K = int(os.environ.get("TOP_K", 3))
AI_CONTEXT_LINES = int(os.environ.get("AI_CONTEXT_LINES", 8))
MAX_SUGGESTION_WORDS = int(os.environ.get("MAX_SUGGESTION_WORDS", 12))
MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
MODEL_CACHE_FOLDER = os.environ.get("MODEL_CACHE_FOLDER", "./models")

app = FastAPI()

# static + uploads (same structure as before)
app.mount("/static", StaticFiles(directory="static"), name="static")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ----------------- password context (safe vs bcrypt 72-byte limit) -----------------
# Use bcrypt_sha256 which pre-hashes with sha256 before bcrypt to avoid 72-byte limit.
# Keep a defensive fallback: if hashing raises ValueError about length, pre-hash manually.
pwd_context = CryptContext(schemes=["bcrypt_sha256"], deprecated="auto")


def hash_password(p: str) -> str:
    """Hash safely and defensively against bcrypt 72-byte limit."""
    if p is None:
        p = ""
    p = str(p)
    try:
        return pwd_context.hash(p)
    except Exception as e:
        # Defensive fallback: pre-hash with sha256 then hash (should not generally be needed if bcrypt_sha256 is available)
        try:
            if "72" in str(e) or isinstance(e, ValueError):
                pre = hashlib.sha256(p.encode("utf-8")).hexdigest()
                return pwd_context.hash(pre)
        except Exception:
            pass
        # re-raise the original exception if fallback didn't work
        raise


def verify_password(p: str, h: str) -> bool:
    try:
        return pwd_context.verify(str(p), h)
    except Exception:
        # if verify fails for any reason, return False
        return False


def create_token(username: str) -> str:
    return jwt.encode({"username": username}, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# ----------------- in-memory state -----------------
auth_clients = {}        # username -> websocket
anon_clients = set()     # set of anonymous websockets

# Keep recent messages as list of dicts: {name, text, ts} (newest last)
recent_messages: List[dict] = []
# Embeddings aligned with a separate list (only messages encoded during runtime)
recent_embeddings: List[np.ndarray] = []

# Persisted latest suggestions so newly connected users immediately see them
current_suggestions: List[str] = []

# ----------------- optional AI/embeddings -----------------
embedding_model = None
groq_client = None

if SentenceTransformer is not None:
    try:
        print("Loading embedding model:", MODEL_NAME)
        embedding_model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE_FOLDER)
        print("Embedding model loaded")
    except Exception as e:
        print("Warning: failed to load embedding model:", e)
        embedding_model = None
else:
    print("sentence-transformers not installed; embedding fallback disabled.")

if Groq is not None and GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("Groq client initialized")
    except Exception as e:
        groq_client = None
        print("Warning: failed to initialize Groq client:", e)
elif Groq is None:
    print("groq package not installed; Groq integration disabled.")
else:
    print("GROQ_API_KEY not found; Groq integration disabled (will fallback to embeddings).")


# ----------------- helpers -----------------
def _clean_text_for_suggestion(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    return " ".join(s.split())


COMMON_GREETINGS = {"hi", "hello", "hey", "hii", "hiya", "yo", "ok", "okay"}


def generate_simple_suggestions(sender: str = None, limit: int = 3) -> List[str]:
    """
    Simple deterministic suggestion generator based on recent_messages in memory.
    Prefer messages from other users, skip greetings and too-short texts.
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
        if sender and author == sender:
            continue
        low = text.lower()
        if low in COMMON_GREETINGS:
            continue
        if len(text) < 3:
            continue
        if text in seen:
            continue
        seen.add(text)
        suggestions.append(text)
    # fallback allow messages from sender if still insufficient
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


async def broadcast_raw(text: str):
    """Broadcast a raw JSON string to all connected clients (auth + anon). Clean closed connections."""
    to_remove = []
    for user, ws in list(auth_clients.items()):
        try:
            await ws.send_text(text)
        except Exception:
            to_remove.append(("auth", user))
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
    """Synchronous DB save helper (short-lived session)."""
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


# ----------------- embeddings helpers (runtime only) -----------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


async def encode_text_async(text: str) -> np.ndarray:
    if embedding_model is None:
        raise RuntimeError("Embedding model not loaded")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: embedding_model.encode(text, convert_to_numpy=True))


def add_to_recent_runtime(name: str, text: str, embedding: Optional[np.ndarray] = None):
    """Add a message to the in-memory recent buffers. For runtime embeddings, keep alignment."""
    recent_messages.append({"name": name, "text": text, "ts": datetime.utcnow().isoformat()})
    if embedding is not None:
        recent_embeddings.append(embedding)
    # keep length bounded
    if len(recent_messages) > MAX_RECENT:
        # if we drop the oldest, possibly also drop the oldest embedding to keep alignment (pop left)
        # remove oldest message
        del recent_messages[0]
        if recent_embeddings:
            # keep embeddings roughly aligned (we only appended embeddings for messages encoded at runtime)
            # if embeddings length > recent_embeddings allowed, trim front
            if len(recent_embeddings) > len(recent_messages):
                # trim oldest embedding
                del recent_embeddings[0]


def find_top_k_similar_excluding(embedding: np.ndarray, exclude_user: str, exclude_text: str, k: int = TOP_K) -> List[str]:
    """Find top-k similar runtime-encoded messages excluding the sender and identical text."""
    if not recent_embeddings:
        return []
    sims = []
    for idx, emb in enumerate(recent_embeddings):
        sim = cosine_similarity(embedding, emb)
        sims.append((idx, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    suggestions = []
    for idx, sim in sims:
        # map idx to the most recent embeddings-aligned messages: embeddings correspond to a suffix of recent_messages
        # compute aligned index: consider embeddings aligned to the tail of recent_messages
        align_offset = len(recent_messages) - len(recent_embeddings)
        msg_idx = align_offset + idx
        if msg_idx < 0 or msg_idx >= len(recent_messages):
            continue
        candidate = recent_messages[msg_idx]
        candidate_user, candidate_text = candidate.get("name"), candidate.get("text")
        if candidate_user == exclude_user:
            continue
        if candidate_text.strip().lower() == exclude_text.strip().lower():
            continue
        if candidate_text not in suggestions:
            suggestions.append(candidate_text)
        if len(suggestions) >= k:
            break
    return suggestions


# ----------------- Groq suggestion helper (optional) -----------------
async def generate_ai_suggestions_groq(message: str, recent_msgs: List[dict], exclude_user: Optional[str] = None) -> List[str]:
    """
    Use Groq API to generate up to TOP_K short suggestions.
    Uses the recent messages (most recent last) but filters out exclude_user from context.
    """
    if groq_client is None:
        return []

    # build context excluding messages from sender
    context_msgs = [ (m["name"], m["text"]) for m in recent_msgs if m.get("name") != exclude_user ]
    context_lines = []
    for user, text in context_msgs[-AI_CONTEXT_LINES:]:
        t = text if len(text) <= 300 else text[:300] + "..."
        context_lines.append(f"{user}: {t}")

    prompt = f"""
You are a concise smart-reply assistant. Given the conversation context (excluding the message sender) and the latest user message,
produce exactly {TOP_K} short conversational reply suggestions, one per line. Return replies only; no JSON, no numbering, no explanations.

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

        # Extract content (robust)
        try:
            text_out = response.choices[0].message.content
        except Exception:
            try:
                text_out = response["choices"][0]["message"]["content"]
            except Exception as e:
                print("generate_ai_suggestions_groq: extract failed:", e)
                return []

        # parse lines
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


# ----------------- load recent messages from DB at startup -----------------
def load_recent_messages_from_db():
    """
    Populate recent_messages from DB so suggestions are available after restarts.
    This loads up to MAX_RECENT messages (oldest first).
    """
    db = SessionLocal()
    try:
        # detect timestamp field name used in DB model
        col = getattr(Message, "created_at", None) or getattr(Message, "timestamp", None)
        if col is None:
            return
        rows = db.query(Message).order_by(col.desc()).limit(MAX_RECENT).all()
        # rows are newest-first; we want oldest-first in recent_messages
        for r in reversed(rows):
            text = getattr(r, "content", None) or getattr(r, "content", "")
            recent_messages.append({"name": getattr(r, "username", None), "text": text, "ts": getattr(r, "created_at", None) or getattr(r, "timestamp", None)})
    except Exception as e:
        print("load_recent_messages_from_db failed:", e)
    finally:
        db.close()


# run load on module import / startup
try:
    load_recent_messages_from_db()
    print(f"Loaded {len(recent_messages)} recent messages from DB into memory.")
except Exception as e:
    print("Failed to preload recent messages from DB:", e)


# ----------------- routes -----------------
@app.get("/")
def get_chat():
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

        try:
            hashed = hash_password(password)
        except Exception as e:
            print("hashing error:", e)
            raise HTTPException(status_code=500, detail="Failed to hash password")

        new_user = User(username=username, password=hashed)
        db.add(new_user)
        db.commit()
        return {"message": "Signup successful"}
    except HTTPException:
        raise
    except Exception as e:
        print("signup error:", e)
        db.rollback()
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
    return {"url": f"/uploads/{os.path.basename(dest)}", "filename": os.path.basename(dest)}


# ----------------- anonymous websocket (read-only) -----------------
@app.websocket("/ws")
async def websocket_anonymous(websocket: WebSocket):
    await websocket.accept()
    anon_clients.add(websocket)
    print("Anonymous client connected. total anon:", len(anon_clients))
    # broadcast counts
    try:
        total_auth = len(auth_clients)
        total_anon = len(anon_clients)
        await broadcast_raw(json.dumps({
            "type": "user_count",
            "auth": total_auth,
            "anon": total_anon,
            "total": total_auth + total_anon
        }))
    except Exception:
        pass

    try:
        while True:
            try:
                _ = await websocket.receive_text()
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
            await broadcast_raw(json.dumps({
                "type": "user_count",
                "auth": total_auth,
                "anon": total_anon,
                "total": total_auth + total_anon
            }))
        except Exception:
            pass


# ----------------- authenticated websocket -----------------
@app.websocket("/ws/{token}")
async def websocket_auth(websocket: WebSocket, token: str):
    payload = decode_token(token)
    if not payload or "username" not in payload:
        await websocket.close()
        return
    username = payload.get("username")

    try:
        await websocket.accept()
    except Exception:
        return

    # send existing users list BEFORE adding
    try:
        for user in list(auth_clients.keys()):
            if user == username:
                continue
            data = json.dumps({"type": "join", "name": user, "color": None})
            try:
                await websocket.send_text(data)
            except Exception:
                pass
    except Exception:
        pass

    # register
    auth_clients[username] = websocket
    print(f"{username} connected (auth). total auth:", len(auth_clients))

    # announce join
    try:
        await broadcast_raw(json.dumps({"type": "join", "name": username, "color": None}))
    except Exception:
        pass

    # send counts and current suggestions to the new user
    try:
        total_auth = len(auth_clients)
        total_anon = len(anon_clients)
        await broadcast_raw(json.dumps({
            "type": "user_count",
            "auth": total_auth,
            "anon": total_anon,
            "total": total_auth + total_anon
        }))
        if current_suggestions:
            try:
                await websocket.send_text(json.dumps({"type": "suggestions", "suggestions": current_suggestions}))
            except Exception:
                pass
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

            # process payload
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None

            try:
                if isinstance(parsed, dict) and parsed.get("type"):
                    t = parsed.get("type")

                    if t in ("join", "leave"):
                        continue

                    if t == "message":
                        mtext = parsed.get("text", "").strip()
                        author = parsed.get("name", username)
                        if not mtext:
                            continue

                        # 1) persist to DB
                        try:
                            save_message_to_db(author, mtext)
                        except Exception as e:
                            print("error saving message:", e)

                        # 2) broadcast original message
                        await broadcast_raw(json.dumps({"type": "message", "name": author, "text": mtext}))

                        # 3) Try AI suggestion pipeline (Groq -> embed-similarity -> simple fallback)
                        suggestions: List[str] = []

                        # attempt Groq (excludes sender from context)
                        try:
                            suggestions = await generate_ai_suggestions_groq(mtext, recent_messages, exclude_user=author)
                        except Exception as e:
                            print("Groq generation failed:", e)
                            suggestions = []

                        # embedding fallback (runtime embeddings only)
                        embedding = None
                        if not suggestions and embedding_model is not None:
                            try:
                                embedding = await encode_text_async(mtext)
                                suggestions = find_top_k_similar_excluding(embedding, exclude_user=author, exclude_text=mtext, k=TOP_K)
                            except Exception as e:
                                print("Embedding similarity error:", e)
                                suggestions = []

                        # final simple fallback using recent_messages text-only
                        if not suggestions:
                            suggestions = generate_simple_suggestions(sender=author, limit=TOP_K)

                        # 4) Add current message to recent buffers (after generating suggestions to avoid echoing current message)
                        if embedding is not None:
                            try:
                                add_to_recent_runtime(author, mtext, embedding)
                            except Exception as e:
                                print("Failed to add to recent buffer with embedding:", e)
                                add_to_recent_runtime(author, mtext, None)
                        else:
                            add_to_recent_runtime(author, mtext, None)

                        # 5) persist and broadcast suggestions (if any)
                        if suggestions:
                            # sanitize & dedupe & trim
                            cleaned = []
                            for s in suggestions:
                                s2 = _clean_text_for_suggestion(s)
                                if not s2:
                                    continue
                                if s2 not in cleaned:
                                    cleaned.append(s2)
                                if len(cleaned) >= TOP_K:
                                    break
                            if cleaned:
                                # update global persisted suggestions
                                current_suggestions[:] = cleaned[:TOP_K]
                                try:
                                    await broadcast_raw(json.dumps({"type": "suggestions", "suggestions": current_suggestions}))
                                except Exception as e:
                                    print("Failed to broadcast suggestions:", e)
                        continue

                    if t == "file":
                        fname = parsed.get("filename") or ""
                        furl = parsed.get("url") or ""
                        author = parsed.get("name", username)
                        content = f"[file] {furl} | {fname}"
                        try:
                            save_message_to_db(author, content)
                        except Exception as e:
                            print("error saving file message:", e)
                        # update recent messages (text form)
                        add_to_recent_runtime(author, content, None)
                        await broadcast_raw(json.dumps({"type": "file", "name": author, "url": furl, "filename": fname}))
                        continue

                    if t == "typing":
                        try:
                            await broadcast_raw(json.dumps({"type": "typing", "name": parsed.get("name", username)}))
                        except Exception:
                            pass
                        continue

                    # unknown typed item -> broadcast
                    await broadcast_raw(text)
                else:
                    # fallback plain text
                    plain = text
                    try:
                        save_message_to_db(username, plain)
                    except Exception as e:
                        print("error saving plain message:", e)
                    add_to_recent_runtime(username, plain, None)
                    await broadcast_raw(json.dumps({"type": "message", "name": username, "text": plain}))
            except Exception as e:
                print("handling message error:", e)

    finally:
        try:
            if username in auth_clients and auth_clients[username] is websocket:
                del auth_clients[username]
        except Exception:
            pass
        print(f"{username} disconnected (auth). total auth:", len(auth_clients))
        # announce leave
        try:
            await broadcast_raw(json.dumps({"type": "leave", "name": username}))
        except Exception:
            pass
        # broadcast updated counts
        try:
            total_auth = len(auth_clients)
            total_anon = len(anon_clients)
            await broadcast_raw(json.dumps({
                "type": "user_count",
                "auth": total_auth,
                "anon": total_anon,
                "total": total_auth + total_anon
            }))
        except Exception:
            pass


# health
@app.get("/ping")
def ping():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}


# local run helper
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
