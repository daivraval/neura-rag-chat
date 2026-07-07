"""
NEURA — an immersive RAG chat experience.

FastAPI backend wrapping the existing Chroma + HuggingFace RAG pipeline,
with SQLite-persisted chat history and a fully custom animated frontend
(particle field canvas, aurora gradients, glassmorphism).

Run:  uvicorn app:app --reload   (or)   python app.py
Then open http://127.0.0.1:8000
"""

import os
import sqlite3
import uuid
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.db")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# ---------------------------------------------------------------- RAG setup
rag = {}


def build_rag():
    from langchain_huggingface import (
        ChatHuggingFace,
        HuggingFaceEmbeddings,
        HuggingFaceEndpoint,
    )
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import ChatPromptTemplate

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
    )
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    llm = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="text-generation",
            max_new_tokens=400,
            do_sample=False,
            huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        )
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                'You are a helpful AI assistant. Use ONLY the provided context to '
                'answer the question. If the answer is not present in the context, '
                'say: "I could not find the answer in the document."',
            ),
            ("human", "Context: {context}\nQuestion: {question}"),
        ]
    )
    rag["retriever"] = retriever
    rag["llm"] = llm
    rag["prompt"] = prompt


# ------------------------------------------------------------- history store
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                created_at TEXT NOT NULL
            );
            """
        )


def now():
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------- server
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    build_rag()
    yield


app = FastAPI(title="NEURA", lifespan=lifespan)


class ChatIn(BaseModel):
    session_id: str | None = None
    message: str


class SessionIn(BaseModel):
    title: str | None = None


@app.get("/api/sessions")
def list_sessions():
    with db() as conn:
        rows = conn.execute(
            "SELECT s.id, s.title, s.created_at,"
            " (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n"
            " FROM sessions s ORDER BY s.created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/sessions")
def create_session(body: SessionIn):
    sid = str(uuid.uuid4())
    with db() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
            (sid, body.title or "New chat", now()),
        )
    return {"id": sid, "title": body.title or "New chat"}


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    with db() as conn:
        session = conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        if not session:
            raise HTTPException(404, "session not found")
        msgs = conn.execute(
            "SELECT role, content, sources, created_at FROM messages"
            " WHERE session_id = ? ORDER BY id",
            (sid,),
        ).fetchall()
    return {"id": sid, "title": session["title"], "messages": [dict(m) for m in msgs]}


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    with db() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    return {"ok": True}


@app.post("/api/chat")
def chat(body: ChatIn):
    query = body.message.strip()
    if not query:
        raise HTTPException(400, "empty message")

    sid = body.session_id
    with db() as conn:
        if not sid or not conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (sid,)
        ).fetchone():
            sid = str(uuid.uuid4())
            title = query[:48] + ("…" if len(query) > 48 else "")
            conn.execute(
                "INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
                (sid, title, now()),
            )
        else:
            # first message names the session
            n = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (sid,)
            ).fetchone()[0]
            if n == 0:
                title = query[:48] + ("…" if len(query) > 48 else "")
                conn.execute(
                    "UPDATE sessions SET title = ? WHERE id = ?", (title, sid)
                )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at)"
            " VALUES (?, 'user', ?, ?)",
            (sid, query, now()),
        )

    docs = rag["retriever"].invoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    final_prompt = rag["prompt"].invoke({"context": context, "question": query})
    try:
        answer = rag["llm"].invoke(final_prompt).content
    except Exception as exc:
        raise HTTPException(502, f"LLM error: {exc}") from exc

    source_snips = " ||| ".join(d.page_content[:220] for d in docs[:3])
    with db() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, sources, created_at)"
            " VALUES (?, 'ai', ?, ?, ?)",
            (sid, answer, source_snips, now()),
        )

    return {
        "session_id": sid,
        "answer": answer,
        "sources": [d.page_content[:220] for d in docs[:3]],
    }


# ----------------------------------------------------------------- frontend
@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>NEURA — Document Intelligence</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@500;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#05060f; --bg2:#0a0d1f;
  --ink:#eef0ff; --ink-dim:#8b90b5; --ink-faint:#4c5170;
  --acc:#7c5cff; --acc2:#00e5c7; --acc3:#ff4d8d;
  --glass:rgba(18,21,44,.55); --glass-brd:rgba(140,150,255,.14);
  --r:18px;
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%}
body{
  background:var(--bg); color:var(--ink);
  font-family:'Inter',sans-serif; overflow:hidden;
  -webkit-font-smoothing:antialiased;
}

/* ---------- background layers ---------- */
#stars{position:fixed;inset:0;z-index:0}
.aurora{position:fixed;border-radius:50%;filter:blur(110px);opacity:.5;z-index:0;pointer-events:none;
  animation:drift 26s ease-in-out infinite alternate}
.a1{width:60vw;height:60vw;background:radial-gradient(circle,#3b1f8f 0%,transparent 65%);top:-22vw;left:-14vw}
.a2{width:52vw;height:52vw;background:radial-gradient(circle,#003f4d 0%,transparent 65%);bottom:-20vw;right:-10vw;animation-delay:-9s}
.a3{width:34vw;height:34vw;background:radial-gradient(circle,#5d1040 0%,transparent 65%);top:32%;left:56%;animation-delay:-17s}
@keyframes drift{
  0%{transform:translate(0,0) scale(1)}
  50%{transform:translate(5vw,-4vh) scale(1.12)}
  100%{transform:translate(-4vw,4vh) scale(.95)}
}
.grain{position:fixed;inset:0;z-index:1;pointer-events:none;opacity:.05;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")}

/* ---------- intro overlay ---------- */
#intro{position:fixed;inset:0;z-index:50;display:flex;flex-direction:column;
  align-items:center;justify-content:center;background:var(--bg);
  transition:opacity 1s ease,visibility 1s}
#intro.gone{opacity:0;visibility:hidden}
#intro .logo{font-family:'Syne',sans-serif;font-weight:800;font-size:clamp(3rem,9vw,7rem);
  letter-spacing:.35em;padding-left:.35em;
  background:linear-gradient(100deg,var(--acc),var(--acc2) 55%,var(--acc3));
  -webkit-background-clip:text;background-clip:text;color:transparent;
  animation:pulse 2.4s ease-in-out infinite}
#intro .sub{margin-top:1rem;color:var(--ink-dim);letter-spacing:.5em;padding-left:.5em;
  font-size:.7rem;text-transform:uppercase;animation:fadeUp 1.4s ease .4s both}
#intro .bar{margin-top:2.6rem;width:180px;height:2px;background:rgba(255,255,255,.08);
  border-radius:2px;overflow:hidden}
#intro .bar i{display:block;height:100%;width:40%;border-radius:2px;
  background:linear-gradient(90deg,var(--acc),var(--acc2));
  animation:load 1.4s ease-in-out infinite}
@keyframes load{0%{transform:translateX(-100%)}100%{transform:translateX(450%)}}
@keyframes pulse{0%,100%{filter:brightness(1)}50%{filter:brightness(1.35)}}
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}

/* ---------- layout ---------- */
#shell{position:relative;z-index:2;display:grid;grid-template-columns:290px 1fr;
  height:100vh;gap:0}

/* ---------- sidebar ---------- */
aside{display:flex;flex-direction:column;border-right:1px solid var(--glass-brd);
  background:linear-gradient(180deg,rgba(10,12,28,.72),rgba(8,9,22,.85));
  backdrop-filter:blur(22px)}
.brand{display:flex;align-items:center;gap:.7rem;padding:1.35rem 1.3rem 1.1rem}
.orb{width:34px;height:34px;border-radius:50%;position:relative;flex:none;
  background:conic-gradient(from 0deg,var(--acc),var(--acc2),var(--acc3),var(--acc));
  animation:spin 7s linear infinite}
.orb::after{content:"";position:absolute;inset:3px;border-radius:50%;background:var(--bg2)}
@keyframes spin{to{transform:rotate(360deg)}}
.brand b{font-family:'Syne',sans-serif;font-weight:800;font-size:1.15rem;letter-spacing:.28em}
.brand span{font-size:.6rem;color:var(--ink-faint);letter-spacing:.24em;display:block;margin-top:2px}
#newChat{margin:0 1.1rem .9rem;padding:.72rem 1rem;border-radius:12px;border:1px solid var(--glass-brd);
  background:linear-gradient(120deg,rgba(124,92,255,.2),rgba(0,229,199,.12));
  color:var(--ink);font-family:'Inter',sans-serif;font-weight:600;font-size:.83rem;
  cursor:pointer;display:flex;align-items:center;gap:.55rem;
  transition:transform .18s ease,box-shadow .25s ease}
#newChat:hover{transform:translateY(-2px);box-shadow:0 8px 28px -8px rgba(124,92,255,.55)}
#newChat svg{width:15px;height:15px}
.hist-label{padding:.4rem 1.35rem .5rem;font-size:.62rem;letter-spacing:.26em;
  color:var(--ink-faint);text-transform:uppercase}
#sessions{flex:1;overflow-y:auto;padding:0 .8rem 1rem;display:flex;flex-direction:column;gap:4px}
#sessions::-webkit-scrollbar{width:4px}
#sessions::-webkit-scrollbar-thumb{background:rgba(140,150,255,.18);border-radius:4px}
.sess{position:relative;padding:.66rem .8rem;border-radius:11px;cursor:pointer;
  border:1px solid transparent;transition:all .2s ease;animation:fadeUp .4s ease both}
.sess:hover{background:rgba(124,92,255,.08);border-color:var(--glass-brd)}
.sess.active{background:rgba(124,92,255,.14);border-color:rgba(124,92,255,.35)}
.sess .t{font-size:.82rem;font-weight:500;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;padding-right:1.4rem}
.sess .m{font-size:.65rem;color:var(--ink-faint);margin-top:2px}
.sess .del{position:absolute;right:.5rem;top:50%;transform:translateY(-50%);
  width:22px;height:22px;border:none;border-radius:6px;background:transparent;
  color:var(--ink-faint);cursor:pointer;opacity:0;transition:all .18s;display:grid;place-items:center}
.sess:hover .del{opacity:1}
.sess .del:hover{color:var(--acc3);background:rgba(255,77,141,.12)}
.side-foot{padding:.9rem 1.3rem;border-top:1px solid var(--glass-brd);
  font-size:.62rem;color:var(--ink-faint);letter-spacing:.12em;display:flex;gap:.5rem;align-items:center}
.dot{width:7px;height:7px;border-radius:50%;background:var(--acc2);
  box-shadow:0 0 10px var(--acc2);animation:blink 2.2s ease infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.35}}

/* ---------- main ---------- */
main{display:flex;flex-direction:column;height:100vh;position:relative}
header{padding:1.15rem 2.2rem;display:flex;justify-content:space-between;align-items:center;
  border-bottom:1px solid var(--glass-brd);background:rgba(8,10,24,.35);backdrop-filter:blur(14px)}
header .title{font-family:'Syne',sans-serif;font-weight:700;font-size:.95rem;letter-spacing:.06em;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:52vw}
header .tag{font-size:.62rem;letter-spacing:.22em;color:var(--ink-faint);text-transform:uppercase;
  border:1px solid var(--glass-brd);padding:.32rem .7rem;border-radius:99px}

#feed{flex:1;overflow-y:auto;padding:2.2rem 2.2rem 1rem;scroll-behavior:smooth}
#feed::-webkit-scrollbar{width:5px}
#feed::-webkit-scrollbar-thumb{background:rgba(140,150,255,.18);border-radius:4px}
.feed-inner{max-width:820px;margin:0 auto;display:flex;flex-direction:column;gap:1.4rem}

/* hero (empty state) */
#hero{max-width:820px;margin:0 auto;padding-top:9vh;text-align:left;
  animation:fadeUp .8s cubic-bezier(.2,.7,.2,1) both}
#hero h1{font-family:'Syne',sans-serif;font-weight:800;line-height:1.04;
  font-size:clamp(2.4rem,5.2vw,4.3rem);letter-spacing:-.01em}
#hero h1 .grad{background:linear-gradient(95deg,var(--acc) 0%,var(--acc2) 55%,var(--acc3) 110%);
  -webkit-background-clip:text;background-clip:text;color:transparent;
  background-size:220% 100%;animation:sheen 6s linear infinite}
@keyframes sheen{to{background-position:220% 0}}
#hero p{margin-top:1.1rem;color:var(--ink-dim);font-weight:300;max-width:34rem;line-height:1.7;font-size:.98rem}
.chips{display:flex;gap:.6rem;flex-wrap:wrap;margin-top:2rem}
.chip{padding:.6rem 1.05rem;border-radius:99px;border:1px solid var(--glass-brd);
  background:var(--glass);backdrop-filter:blur(8px);font-size:.8rem;color:var(--ink-dim);
  cursor:pointer;transition:all .22s ease}
.chip:hover{color:var(--ink);border-color:rgba(124,92,255,.5);transform:translateY(-2px);
  box-shadow:0 10px 30px -12px rgba(124,92,255,.5)}

/* messages */
.msg{display:flex;gap:.95rem;animation:msgIn .5s cubic-bezier(.2,.8,.25,1) both}
@keyframes msgIn{from{opacity:0;transform:translateY(18px) scale(.985)}to{opacity:1;transform:none}}
.avatar{width:36px;height:36px;border-radius:12px;flex:none;display:grid;place-items:center;
  font-family:'Syne',sans-serif;font-weight:700;font-size:.72rem}
.msg.user .avatar{background:linear-gradient(135deg,#2b2f55,#1a1d38);border:1px solid var(--glass-brd)}
.msg.ai .avatar{background:conic-gradient(from 40deg,var(--acc),var(--acc2),var(--acc));color:#04060f}
.bubble{padding:1rem 1.25rem;border-radius:var(--r);line-height:1.72;font-size:.93rem;
  max-width:86%;position:relative}
.msg.user .bubble{background:linear-gradient(120deg,rgba(124,92,255,.16),rgba(124,92,255,.06));
  border:1px solid rgba(124,92,255,.25)}
.msg.ai .bubble{background:var(--glass);border:1px solid var(--glass-brd);backdrop-filter:blur(14px)}
.caret{display:inline-block;width:8px;height:1.05em;vertical-align:-3px;margin-left:2px;
  background:var(--acc2);animation:blink 1s steps(1) infinite}
details.src{margin-top:.9rem;border-top:1px dashed rgba(140,150,255,.18);padding-top:.7rem}
details.src summary{cursor:pointer;font-size:.68rem;letter-spacing:.18em;text-transform:uppercase;
  color:var(--ink-faint);list-style:none;display:flex;align-items:center;gap:.45rem}
details.src summary::before{content:"▸";transition:transform .2s}
details.src[open] summary::before{transform:rotate(90deg)}
details.src .snip{margin-top:.6rem;font-family:'JetBrains Mono',monospace;font-size:.7rem;
  color:var(--ink-dim);background:rgba(0,0,0,.28);border-left:2px solid var(--acc2);
  padding:.6rem .8rem;border-radius:0 8px 8px 0;line-height:1.6}
.thinking{display:flex;gap:6px;padding:.35rem 0}
.thinking i{width:7px;height:7px;border-radius:50%;background:var(--acc2);
  animation:bob 1.2s ease-in-out infinite}
.thinking i:nth-child(2){animation-delay:.15s;background:var(--acc)}
.thinking i:nth-child(3){animation-delay:.3s;background:var(--acc3)}
@keyframes bob{0%,100%{transform:translateY(0);opacity:.4}50%{transform:translateY(-7px);opacity:1}}

/* composer */
#composer{padding:1.1rem 2.2rem 1.6rem}
.comp-inner{max-width:820px;margin:0 auto;position:relative;border-radius:20px;padding:1.5px;
  background:linear-gradient(120deg,rgba(124,92,255,.55),rgba(0,229,199,.4) 50%,rgba(255,77,141,.45));
  background-size:250% 100%;animation:sheen 8s linear infinite}
.comp-box{display:flex;align-items:flex-end;gap:.7rem;background:rgba(8,10,24,.92);
  border-radius:19px;padding:.75rem .8rem .75rem 1.25rem;backdrop-filter:blur(20px)}
#input{flex:1;background:transparent;border:none;outline:none;resize:none;color:var(--ink);
  font-family:'Inter',sans-serif;font-size:.95rem;line-height:1.55;max-height:140px;padding:.35rem 0}
#input::placeholder{color:var(--ink-faint)}
#send{width:42px;height:42px;flex:none;border:none;border-radius:13px;cursor:pointer;
  display:grid;place-items:center;color:#04060f;
  background:linear-gradient(135deg,var(--acc2),var(--acc));
  transition:transform .16s ease,box-shadow .25s ease,opacity .2s}
#send:hover{transform:scale(1.07);box-shadow:0 6px 26px -6px rgba(0,229,199,.6)}
#send:disabled{opacity:.35;cursor:not-allowed;transform:none;box-shadow:none}
.hint{max-width:820px;margin:.6rem auto 0;text-align:center;font-size:.64rem;
  color:var(--ink-faint);letter-spacing:.14em}

@media (max-width:840px){
  #shell{grid-template-columns:1fr}
  aside{display:none}
  #feed,#composer{padding-left:1.1rem;padding-right:1.1rem}
}
</style>
</head>
<body>

<div id="intro">
  <div class="logo">NEURA</div>
  <div class="sub">document intelligence</div>
  <div class="bar"><i></i></div>
</div>

<canvas id="stars"></canvas>
<div class="aurora a1"></div><div class="aurora a2"></div><div class="aurora a3"></div>
<div class="grain"></div>

<div id="shell">
  <aside>
    <div class="brand">
      <div class="orb"></div>
      <div><b>NEURA</b><span>RAG&nbsp;ENGINE</span></div>
    </div>
    <button id="newChat">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>
      New conversation
    </button>
    <div class="hist-label">History</div>
    <div id="sessions"></div>
    <div class="side-foot"><span class="dot"></span> Qwen 2.5 · MiniLM · Chroma</div>
  </aside>

  <main>
    <header>
      <div class="title" id="chatTitle">New conversation</div>
      <div class="tag">grounded · document-only</div>
    </header>

    <div id="feed">
      <div id="hero">
        <h1>Ask your documents<br><span class="grad">anything at all.</span></h1>
        <p>NEURA retrieves the most relevant passages from your indexed PDFs using
           maximal-marginal-relevance search over a Chroma vector store, then answers
           with a grounded language model. No hallucinations  if it isn't in the
           document, it says so.</p>
        <div class="chips">
          <div class="chip">Summarize the document</div>
          <div class="chip">What are the key findings?</div>
          <div class="chip">Explain the methodology</div>
          <div class="chip">List the conclusions</div>
        </div>
      </div>
      <div class="feed-inner" id="feedInner"></div>
    </div>

    <div id="composer">
      <div class="comp-inner">
        <div class="comp-box">
          <textarea id="input" rows="1" placeholder="Ask something about your document…"></textarea>
          <button id="send" aria-label="send">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg>
          </button>
        </div>
      </div>
      <div class="hint">ENTER TO SEND · SHIFT+ENTER FOR NEW LINE</div>
    </div>
  </main>
</div>

<script>
/* ============ particle constellation background ============ */
const cv = document.getElementById('stars'), cx = cv.getContext('2d');
let W, H, pts = [], mouse = {x:-1e4, y:-1e4};
function resize(){ W = cv.width = innerWidth; H = cv.height = innerHeight; }
resize(); addEventListener('resize', resize);
addEventListener('mousemove', e => { mouse.x = e.clientX; mouse.y = e.clientY; });
const N = Math.min(110, innerWidth / 12);
for (let i = 0; i < N; i++) pts.push({
  x: Math.random()*innerWidth, y: Math.random()*innerHeight,
  vx: (Math.random()-.5)*.35, vy: (Math.random()-.5)*.35,
  r: Math.random()*1.6 + .4
});
function tick(){
  cx.clearRect(0,0,W,H);
  for (const p of pts){
    p.x += p.vx; p.y += p.vy;
    if (p.x < 0 || p.x > W) p.vx *= -1;
    if (p.y < 0 || p.y > H) p.vy *= -1;
    const dm = Math.hypot(p.x-mouse.x, p.y-mouse.y);
    if (dm < 160){ p.x += (p.x-mouse.x)/dm*.6; p.y += (p.y-mouse.y)/dm*.6; }
    cx.beginPath(); cx.arc(p.x, p.y, p.r, 0, 7);
    cx.fillStyle = 'rgba(160,170,255,.5)'; cx.fill();
  }
  for (let i = 0; i < pts.length; i++) for (let j = i+1; j < pts.length; j++){
    const a = pts[i], b = pts[j], d = Math.hypot(a.x-b.x, a.y-b.y);
    if (d < 130){
      cx.beginPath(); cx.moveTo(a.x,a.y); cx.lineTo(b.x,b.y);
      cx.strokeStyle = `rgba(124,92,255,${(1-d/130)*.16})`; cx.stroke();
    }
  }
  requestAnimationFrame(tick);
}
tick();

/* ============ app state ============ */
const $ = s => document.querySelector(s);
const feed = $('#feedInner'), feedWrap = $('#feed'), hero = $('#hero');
const input = $('#input'), sendBtn = $('#send');
let sessionId = null, busy = false;

setTimeout(() => $('#intro').classList.add('gone'), 1600);

/* ---------- sessions ---------- */
async function loadSessions(){
  const rows = await (await fetch('/api/sessions')).json();
  const box = $('#sessions'); box.innerHTML = '';
  rows.forEach((s, i) => {
    const el = document.createElement('div');
    el.className = 'sess' + (s.id === sessionId ? ' active' : '');
    el.style.animationDelay = (i * 40) + 'ms';
    el.innerHTML = `<div class="t"></div><div class="m">${s.n} messages</div>
      <button class="del" title="delete">✕</button>`;
    el.querySelector('.t').textContent = s.title;
    el.onclick = () => openSession(s.id, s.title);
    el.querySelector('.del').onclick = async e => {
      e.stopPropagation();
      await fetch('/api/sessions/' + s.id, {method:'DELETE'});
      if (s.id === sessionId) newChat();
      loadSessions();
    };
    box.appendChild(el);
  });
}

async function openSession(id, title){
  sessionId = id; busy = false;
  $('#chatTitle').textContent = title;
  hero.style.display = 'none'; feed.innerHTML = '';
  const data = await (await fetch('/api/sessions/' + id)).json();
  for (const m of data.messages)
    addMsg(m.role, m.content, m.sources ? m.sources.split(' ||| ') : null, false);
  loadSessions();
  feedWrap.scrollTop = feedWrap.scrollHeight;
}

function newChat(){
  sessionId = null; feed.innerHTML = '';
  hero.style.display = 'block';
  $('#chatTitle').textContent = 'New conversation';
  loadSessions();
}
$('#newChat').onclick = newChat;

/* ---------- messages ---------- */
function addMsg(role, text, sources, animate = true){
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  if (!animate) el.style.animation = 'none';
  el.innerHTML = `<div class="avatar">${role === 'user' ? 'YOU' : 'AI'}</div>
                  <div class="bubble"></div>`;
  const bub = el.querySelector('.bubble');
  bub.textContent = text;
  if (sources && sources.length && sources[0]){
    const det = document.createElement('details');
    det.className = 'src';
    det.innerHTML = '<summary>retrieved context</summary>';
    sources.forEach(s => {
      const d = document.createElement('div');
      d.className = 'snip'; d.textContent = s + '…';
      det.appendChild(d);
    });
    bub.appendChild(det);
  }
  feed.appendChild(el);
  feedWrap.scrollTop = feedWrap.scrollHeight;
  return bub;
}

function typeInto(bub, text, sources){
  return new Promise(res => {
    bub.textContent = '';
    const caret = document.createElement('span'); caret.className = 'caret';
    bub.appendChild(caret);
    let i = 0;
    const step = Math.max(1, Math.round(text.length / 220));
    (function go(){
      if (i < text.length){
        caret.before(document.createTextNode(text.slice(i, i + step)));
        i += step;
        feedWrap.scrollTop = feedWrap.scrollHeight;
        setTimeout(go, 12);
      } else {
        caret.remove();
        if (sources && sources.length){
          const det = document.createElement('details');
          det.className = 'src';
          det.innerHTML = '<summary>retrieved context</summary>';
          sources.forEach(s => {
            const d = document.createElement('div');
            d.className = 'snip'; d.textContent = s + '…';
            det.appendChild(d);
          });
          bub.appendChild(det);
        }
        feedWrap.scrollTop = feedWrap.scrollHeight;
        res();
      }
    })();
  });
}

async function send(){
  const q = input.value.trim();
  if (!q || busy) return;
  busy = true; sendBtn.disabled = true;
  input.value = ''; input.style.height = 'auto';
  hero.style.display = 'none';
  addMsg('user', q);
  const bub = addMsg('ai', '');
  bub.innerHTML = '<div class="thinking"><i></i><i></i><i></i></div>';
  try {
    const r = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({session_id: sessionId, message: q})
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();
    sessionId = data.session_id;
    await typeInto(bub, data.answer, data.sources);
    loadSessions();
  } catch (err) {
    bub.textContent = '⚠ ' + err.message;
  }
  busy = false; sendBtn.disabled = false; input.focus();
}

sendBtn.onclick = send;
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); send(); }
});
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 140) + 'px';
});
document.querySelectorAll('.chip').forEach(c =>
  c.onclick = () => { input.value = c.textContent; send(); });

loadSessions();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
