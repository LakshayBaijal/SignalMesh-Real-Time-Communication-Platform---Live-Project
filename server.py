# server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from passlib.context import CryptContext
from jose import jwt
import json
import os
import shutil
import asyncio
import numpy as np
from typing import List, Tuple, Optional

# Try Groq SDK (optional)
try:
    from groq import Groq
except Exception:
    Groq = None

# Try sentence-transformers (optional)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from dotenv import load_dotenv
from database import SessionLocal, User, Message  # your DB module

load_dotenv()

# -----------------------
# Config
# -----------------------
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
ALGORITHM = "HS256"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "all-MiniLM-L6-v2"    # local embeddings (free)
MODEL_CACHE_FOLDER = "./models"
MAX_RECENT = 200
TOP_K = 3
AI_CONTEXT_LINES = 8               # how many prior messages to send to LLM (we exclude sender)
MAX_SUGGESTION_WORDS = 12

# -----------------------
# App init
# -----------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# -----------------------
# State
# -----------------------
auth_clients = {}        # username -> websocket
anon_clients = set()     # set of anonymous websockets

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory recent buffers (most recent last)
recent_messages: List[Tuple[str, str]] = []    # (username, text)
recent_embeddings: List[np.ndarray] = []      # aligned with recent_messages

# Keep last broadcasted suggestions (persist for frontend)
current_suggestions: List[str] = []

# -----------------------
# Load optional embedding model
# -----------------------
embedding_model = None
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

# -----------------------
# Initialize Groq client (optional)
# -----------------------
groq_client: Optional[object] = None
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

# -----------------------
# Helpers: auth, broadcast, DB
# -----------------------
def hash_password(p: str) -> str:
    return pwd_context.hash(p)

def verify_password(p: str, h: str) -> bool:
    return pwd_context.verify(p, h)

def create_token(username: str) -> str:
    return jwt.encode({"username": username}, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

async def broadcast_raw(text: str):
    """Broadcast raw json text to all connected clients (auth + anon)."""
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

# -----------------------
# Embedding helpers
# -----------------------
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

def add_to_recent(username: str, text: str, embedding: np.ndarray):
    recent_messages.append((username, text))
    recent_embeddings.append(embedding)
    if len(recent_messages) > MAX_RECENT:
        recent_messages.pop(0)
        recent_embeddings.pop(0)

def find_top_k_similar_excluding(embedding: np.ndarray, exclude_user: str, exclude_text: str, k: int = TOP_K) -> List[str]:
    if not recent_embeddings:
        return []
    sims = []
    for idx, emb in enumerate(recent_embeddings):
        sim = cosine_similarity(embedding, emb)
        sims.append((idx, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    suggestions = []
    for idx, sim in sims:
        candidate_user, candidate_text = recent_messages[idx][0], recent_messages[idx][1]
        if candidate_user == exclude_user:
            continue
        if candidate_text.strip().lower() == exclude_text.strip().lower():
            continue
        if candidate_text not in suggestions:
            suggestions.append(candidate_text)
        if len(suggestions) >= k:
            break
    return suggestions

# -----------------------
# Groq suggestion generator (EXCLUDES sender from context)
# -----------------------
async def generate_ai_suggestions_groq(message: str, recent_msgs: List[Tuple[str, str]], exclude_user: Optional[str] = None) -> List[str]:
    """
    Use Groq to generate up to TOP_K short replies.
    Excludes messages by `exclude_user` from the context to avoid echoing sender's messages.
    """
    if groq_client is None:
        return []

    # Filter out messages from exclude_user when building context
    context_msgs = [ (u,t) for (u,t) in recent_msgs if u != exclude_user ]
    # take last AI_CONTEXT_LINES from filtered context
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

        # Extract text (SDK returns object; handle both attribute/dict access)
        try:
            text_out = response.choices[0].message.content
        except Exception:
            try:
                text_out = response["choices"][0]["message"]["content"]
            except Exception as e:
                print("generate_ai_suggestions_groq: extract failed:", e)
                return []

        # Debug log
        try:
            print("Groq raw content:", text_out)
        except:
            pass

        # Parse lines to suggestions
        suggestions = []
        for line in text_out.splitlines():
            line = line.strip()
            if not line:
                continue
            # remove numeric bullets like "1. "
            if len(line) > 2 and line[0].isdigit() and line[1] == '.':
                parts = line.split('. ', 1)
                if len(parts) > 1:
                    line = parts[1].strip()
            if line.startswith(("-", "*")):
                line = line[1:].strip()
            # unquote
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

# -----------------------
# Debug endpoint
# -----------------------
@app.post("/debug_ai")
async def debug_ai(payload=Body(...)):
    """
    POST {"message":"..."} -> quick check of Groq (or embedding fallback)
    """
    msg = None
    if isinstance(payload, dict):
        msg = payload.get("message")
    if not msg:
        raise HTTPException(status_code=400, detail="message required")

    suggestions = []
    try:
        # Call Groq excluding no user (just test)
        suggestions = await generate_ai_suggestions_groq(msg, recent_messages, exclude_user=None)
    except Exception as e:
        print("debug_ai: Groq call failed:", e)
        suggestions = []

    if not suggestions and embedding_model is not None:
        try:
            emb = await encode_text_async(msg)
            suggestions = find_top_k_similar_excluding(emb, exclude_user="", exclude_text=msg, k=TOP_K)
        except Exception as e:
            print("debug_ai: embedding fallback failed:", e)
            suggestions = []

    return {"message": msg, "suggestions": suggestions, "recent_sample": recent_messages[-10:], "current_suggestions": current_suggestions}

# -----------------------
# Web routes & websockets
# -----------------------
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

# anonymous readonly websocket
@app.websocket("/ws")
async def websocket_anonymous(websocket: WebSocket):
    await websocket.accept()
    anon_clients.add(websocket)
    print("Anonymous client connected. total anon:", len(anon_clients))
    await broadcast_user_count()
    try:
        while True:
            try:
                _ = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                break
    finally:
        if websocket in anon_clients:
            anon_clients.remove(websocket)
        print("Anonymous client disconnected. total anon:", len(anon_clients))
        await broadcast_user_count()

# authenticated websocket
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

    # send existing user list to new socket
    try:
        await send_existing_users_to_socket(websocket, exclude_username=username)
    except Exception:
        pass

    # register
    auth_clients[username] = websocket
    print(f"{username} connected (auth). total auth:", len(auth_clients))

    # announce join
    try:
        join_msg = json.dumps({"type":"join","name": username, "color": None})
        await broadcast_raw(join_msg)
    except Exception:
        pass

    # send current user counts and current suggestions to the new user
    try:
        await broadcast_user_count()
        # send current suggestions (persisted) to newly connected user so they see them immediately
        if current_suggestions:
            await websocket.send_text(json.dumps({
                "type": "suggestions",
                "suggestions": current_suggestions
            }))
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

            # process incoming payload
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and parsed.get("type"):
                    t = parsed.get("type")
                    if t in ("join", "leave"):
                        continue

                    # ---------- MESSAGE HANDLING: suggestions are computed BEFORE adding current message ----------
                    if t == "message":
                        mtext = parsed.get("text", "")
                        display_name = parsed.get("name", username)

                        # persist message
                        try:
                            save_message_to_db(display_name, mtext)
                        except Exception as e:
                            print("error saving message:", e)

                        # broadcast the original message to all
                        await broadcast_raw(text)

                        # prepare embedding for fallback similarity
                        embedding = None
                        if embedding_model is not None:
                            try:
                                embedding = await encode_text_async(mtext)
                            except Exception as e:
                                print("Embedding generation error:", e)
                                embedding = None

                        suggestions: List[str] = []

                        # 1) Primary: Groq generation excluding sender's prior messages from context
                        try:
                            suggestions = await generate_ai_suggestions_groq(mtext, recent_messages, exclude_user=display_name)
                        except Exception as e:
                            print("Groq generation failed:", e)
                            suggestions = []

                        # 2) Fallback: embedding similarity that excludes sender and identical text
                        if not suggestions and embedding is not None and recent_embeddings:
                            try:
                                suggestions = find_top_k_similar_excluding(embedding, exclude_user=display_name, exclude_text=mtext, k=TOP_K)
                            except Exception as e:
                                print("Similarity fallback error:", e)
                                suggestions = []

                        # 3) If still nothing and embeddings not available, try Groq without change (it will use prior msgs except sender)
                        if not suggestions and embedding is None and groq_client is not None:
                            try:
                                suggestions = await generate_ai_suggestions_groq(mtext, recent_messages, exclude_user=display_name)
                            except Exception as e:
                                print("Groq fallback (no embedding) failed:", e)
                                suggestions = []

                        # 4) ADD current message to recent buffers AFTER generating suggestions
                        if embedding is not None:
                            try:
                                add_to_recent(display_name, mtext, embedding)
                            except Exception as e:
                                print("Failed to add to recent buffer:", e)
                        else:
                            # optionally we could store text-only history when no embeddings available
                            pass

                        # 5) Update persisted suggestions and broadcast them to ALL clients (so suggestions are "permanent" and visible to everyone)
                        if suggestions:
                            # sanitize & limit
                            suggestions = [s.strip() for s in suggestions if s and s.strip()]
                            if suggestions:
                                current_suggestions = suggestions[:TOP_K]  # update persisted suggestions
                                try:
                                    await broadcast_raw(json.dumps({
                                        "type": "suggestions",
                                        "suggestions": current_suggestions
                                    }))
                                except Exception as e:
                                    print("Failed to broadcast suggestions:", e)
                        else:
                            # if no suggestions, we can optionally clear current_suggestions or keep previous; here we keep previous
                            pass

                        continue

                    # ---------- FILE HANDLING ----------
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

                    # other events (typing, etc)
                    await broadcast_raw(text)
                else:
                    # fallback: plain message
                    plain = text
                    try:
                        save_message_to_db(username, plain)
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

# health
@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
