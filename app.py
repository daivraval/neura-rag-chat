"""
NEURA — an immersive RAG chat experience.

FastAPI backend wrapping the existing Chroma + HuggingFace RAG pipeline,
with SQLite-persisted chat history and a hand-written frontend laid out
like a listening-library dashboard: the paper is the library, questions
are titles on its shelf, and the composer is the player bar.

Run:  uvicorn app:app --reload   (or)   python app.py
Then open http://127.0.0.1:8000
"""

import json
import os
import sqlite3
import uuid
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.db")

# ---------------------------------------------------------------- RAG setup
rag = {}


def build_rag():
    from neura import pipeline

    store = pipeline.open_index()
    rag["retriever"] = pipeline.make_retriever(store, **pipeline.RETRIEVAL)
    rag["llm"] = pipeline.make_llm()
    rag["prompt"] = pipeline.make_prompt()
    rag["generate"] = pipeline.generate
    rag["model"] = pipeline.resolve_llm()[1].split("/")[-1]
    rag["info"] = {
        "document": pipeline.DOCUMENT,
        "chunks": store._collection.count(),
        "chunk_size": pipeline.CHUNK_SIZE,
        "chunk_overlap": pipeline.CHUNK_OVERLAP,
        "retrieval": pipeline.RETRIEVAL,
        "embeddings": pipeline.EMBED_MODEL.split("/")[-1],
        "model": rag["model"],
        "provider": pipeline.resolve_llm()[0],
    }


# ------------------------------------------------------------- eval results
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals", "results")


def load_evals():
    """Headline numbers from the saved eval reports, so the UI never quotes
    a figure the repo can't back up. Missing reports just leave gaps."""
    import glob

    out = {"retrieval": None, "answers": None}
    path = os.path.join(RESULTS_DIR, "retrieval.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        n = sum(q.get("answerable", True) for q in next(iter(r["questions"].values()), []))
        out["retrieval"] = {
            "generated": r["generated"],
            "configs": [
                {"name": name, "config": s["config"], "hit": s["hit_at_k"], "mrr": s["mrr"],
                 "page_hit": s["page_hit_at_k"]}
                for name, s in r["summary"].items()
            ],
            "answerable": n,
        }
    runs = sorted(glob.glob(os.path.join(RESULTS_DIR, "generation_app_*.json")), key=os.path.getmtime)
    if runs:
        with open(runs[-1], encoding="utf-8") as f:
            g = json.load(f)
        s = g["summary"]
        questions = g["questions"]
        out["answers"] = {
            "generated": g["generated"],
            "model": s["model"],
            "judge": s.get("judge"),
            "questions": len(questions),
            "traps": sum(1 for q in questions if not q.get("answerable", True)),
            "answer_accuracy": s["answer_accuracy"],
            "refusal_accuracy": s["refusal_accuracy"],
            "hallucination_rate": None if s["refusal_accuracy"] is None else 1 - s["refusal_accuracy"],
            "false_refusal_rate": s["false_refusal_rate"],
            "faithfulness": s.get("faithfulness"),
        }
    return out


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


@app.get("/api/info")
def info():
    return rag["info"]


@app.get("/api/evals")
def evals():
    return load_evals()


@app.get("/api/sessions")
def list_sessions():
    with db() as conn:
        rows = conn.execute(
            "SELECT s.id, s.title, s.created_at,"
            " (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n,"
            " (SELECT substr(content, 1, 200) FROM messages m WHERE m.session_id = s.id"
            "  AND m.role = 'ai' ORDER BY m.id DESC LIMIT 1) AS preview"
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
    messages = []
    for m in msgs:
        m = dict(m)
        m["sources"] = parse_sources(m["sources"])
        messages.append(m)
    return {"id": sid, "title": session["title"], "messages": messages}


def parse_sources(raw):
    """Sources are stored as JSON [{text, page}]; older rows used ' ||| '."""
    if not raw:
        return []
    if raw.startswith("["):
        return json.loads(raw)
    return [{"text": t, "page": None} for t in raw.split(" ||| ")]


def snippet(doc):
    page = doc.metadata.get("page")
    return {"text": doc.page_content[:220], "page": None if page is None else page + 1}


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
    try:
        answer = rag["generate"](rag["llm"], rag["prompt"], query, docs)
    except Exception as exc:
        if "429" in str(exc) or "rate_limit" in str(exc):
            raise HTTPException(
                429, "the LLM provider's rate limit was reached. Wait a few seconds"
            ) from exc
        raise HTTPException(502, f"LLM error: {exc}") from exc

    sources = [snippet(d) for d in docs[:3]]
    with db() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, sources, created_at)"
            " VALUES (?, 'ai', ?, ?, ?)",
            (sid, answer, json.dumps(sources), now()),
        )

    return {
        "session_id": sid,
        "answer": answer,
        "sources": sources,
    }


@app.get("/pdf")
def pdf():
    """The indexed document, so page links in the UI open the source."""
    from neura import pipeline

    return FileResponse(pipeline.PDF_PATH, media_type="application/pdf")


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
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&family=Hanken+Grotesk:wght@400;500;600;800&display=swap" rel="stylesheet">
<style>
:root{
  --ground:#16130E; --oxblood:#5B1408; --rust:#9F2E10; --umber:#4B3F29;
  --amber:#E8892F; --apricot:#F1B978;
  --raise:color-mix(in srgb,var(--umber) 16%,var(--ground));
  --text:var(--apricot);
  --muted:color-mix(in srgb,var(--apricot) 68%,var(--ground));
  --line:var(--umber);
  --line-strong:color-mix(in srgb,var(--apricot) 42%,var(--ground));
  --on-amber:var(--ground);
  --sans:'Hanken Grotesk',system-ui,sans-serif;
  --mono:'Courier Prime',ui-monospace,Consolas,monospace;
  --rail:84px; --bar:84px;
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%}
body{background:var(--ground);color:var(--text);font-family:var(--mono);font-size:14px;line-height:1.5;
  -webkit-font-smoothing:antialiased;overflow:hidden}
button,input,textarea{font:inherit;color:inherit}
button{background:none;border:0;cursor:pointer}
a{color:inherit;text-underline-offset:3px;text-decoration-thickness:1px}
a:hover{color:var(--amber)}
::selection{background:var(--rust);color:var(--apricot)}
:focus-visible{outline:2px solid var(--amber);outline-offset:2px}
.icon{width:22px;height:22px;flex:none;fill:none;stroke:currentColor;stroke-width:1.6;
  stroke-linecap:round;stroke-linejoin:round}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
*{scrollbar-color:var(--umber) transparent;scrollbar-width:thin}

/* ---------- frame ---------- */
#app{display:grid;grid-template-columns:var(--rail) minmax(0,1fr);grid-template-rows:var(--bar) minmax(0,1fr);
  height:calc(100dvh - 32px);margin:16px;border:1px solid var(--line);background:var(--ground)}

/* ---------- rail ---------- */
.rail{grid-row:1 / span 2;display:flex;flex-direction:column;border-right:1px solid var(--line)}
.mark{height:var(--bar);display:grid;place-items:center;border-bottom:1px solid var(--line)}
.mark svg{width:44px;height:44px}
.rail nav{display:flex;flex-direction:column;gap:6px;padding-top:22px}
.rail .spacer{flex:1}
.rail .bottom{padding-bottom:16px}
.rail-btn{display:flex;flex-direction:column;align-items:center;gap:5px;padding:10px 0;width:100%;
  font-size:11px;color:var(--muted);text-decoration:none}
.rail-btn:hover{color:var(--apricot)}
.rail-btn[aria-current="page"]{color:var(--amber)}

/* ---------- top bar ---------- */
.topbar{display:flex;align-items:stretch;background:var(--amber);color:var(--on-amber);
  border-bottom:1px solid var(--line)}
.topbar :focus-visible{outline-color:var(--ground)}
.search{display:flex;align-items:center;gap:10px;margin:18px 24px;width:min(380px,40%);
  border:1.5px solid var(--ground);padding:0 14px;height:48px}
.search input{flex:1;min-width:0;background:none;border:0;outline:0;font-size:15px;color:var(--ground)}
.search input::placeholder{color:color-mix(in srgb,var(--ground) 80%,var(--amber))}
.topbar .grow{flex:1}
.cell{border-left:1px solid var(--ground);display:flex;align-items:center}
.doc-thumb{width:var(--bar);justify-content:center}
.doc-thumb svg{width:52px;height:64px;border:1px solid var(--ground)}
.doc-name{gap:14px;padding:0 20px 0 18px;min-width:0;text-align:left;cursor:pointer;width:300px}
.doc-name span{display:block;min-width:0}
.doc-name b{display:block;font-family:var(--sans);font-weight:600;font-size:16px;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.doc-name small{font-size:12px;white-space:nowrap}
.doc-name .narrow{display:none}
.proof{padding:0 22px;font-size:13px;font-variant-numeric:tabular-nums;white-space:nowrap}
.proof:hover{background:color-mix(in srgb,var(--amber) 82%,var(--ground))}
.proof[hidden]{display:none}
.doc-name .icon{margin-left:auto;width:18px;height:18px}
.bell{width:var(--bar);justify-content:center}
.bell:hover,.doc-name:hover{background:color-mix(in srgb,var(--amber) 82%,var(--ground))}

/* ---------- main ---------- */
main{display:grid;grid-template-rows:auto auto minmax(0,1fr) auto;min-height:0}
.toolbar{grid-row:1}.types{grid-row:2}.scroll{grid-row:3}.player{grid-row:4}
.toolbar{display:flex;align-items:center;gap:12px;padding:26px 32px 22px;flex-wrap:wrap}
.tabs{display:flex;gap:10px;flex-wrap:wrap}
.tab{border:1px solid var(--line-strong);padding:6px 14px;font-family:var(--sans);font-size:14px;font-weight:500}
.tab:hover{border-color:var(--apricot)}
.tab[aria-selected="true"]{background:var(--rust);border-color:var(--rust);color:var(--apricot)}
.tab[hidden]{display:none}
.toolbar .grow{flex:1}
.filter{display:flex;align-items:center;gap:10px;border:1px solid var(--line-strong);height:46px;padding:0 14px;
  width:min(300px,100%)}
.filter input{flex:1;min-width:0;background:none;border:0;outline:0;font-size:15px}
.filter input::placeholder{color:var(--muted)}
.filter:focus-within,.search:focus-within{outline:2px solid var(--amber);outline-offset:2px}
.search:focus-within{outline-color:var(--ground)}
.filter-btn{width:46px;height:46px;border:1px solid var(--line-strong);display:grid;place-items:center}
.filter-btn[aria-expanded="true"]{background:var(--rust);border-color:var(--rust)}
.types{display:flex;gap:8px;flex-wrap:wrap;padding:0 32px 18px;margin-top:-6px}
.types[hidden]{display:none}
.type-chip{border:1px dashed var(--line-strong);padding:4px 10px;font-size:12px;color:var(--muted)}
.type-chip[aria-pressed="true"]{border-style:solid;border-color:var(--amber);color:var(--amber)}

.scroll{min-height:0;overflow-y:auto;padding:0 32px 28px}
.view[hidden]{display:none}

/* ---------- shelf cards ---------- */
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:30px 34px}
.card{display:flex;flex-direction:column;border-bottom:1px solid var(--line-strong);padding-bottom:12px}
.card[hidden]{display:none}
.card-top{display:grid;grid-template-columns:84px minmax(0,1fr);gap:14px;margin-bottom:22px;flex:1}
.cover-col svg{display:block;width:84px;height:112px}
.cover-col .meta{display:block;margin-top:12px;font-size:12px;line-height:1.35;color:var(--muted)}
.card h3{font-family:var(--sans);font-weight:500;font-size:17px;line-height:1.3;color:var(--apricot);
  text-wrap:balance}
.by{font-size:12px;color:var(--muted);margin:6px 0 10px}
.blurb{font-size:12.5px;line-height:1.55;color:var(--muted);display:-webkit-box;-webkit-line-clamp:6;
  -webkit-box-orient:vertical;overflow:hidden}
.primary{background:var(--oxblood);color:var(--apricot);height:50px;width:100%;font-size:15px;
  border:1px solid var(--oxblood)}
.primary:hover{background:var(--rust);border-color:var(--rust)}
.primary:active{transform:translateY(1px)}
.actions{display:grid;grid-template-columns:1fr 1fr;margin-top:6px}
.actions.one{grid-template-columns:1fr}
.act{display:flex;align-items:center;justify-content:center;gap:8px;padding:9px 0;font-size:12.5px;color:var(--muted)}
.act .icon{width:18px;height:18px}
.act:hover{color:var(--apricot)}
.act.danger:hover{color:var(--rust)}
.empty{grid-column:1 / -1;border:1px dashed var(--line-strong);padding:36px;color:var(--muted);max-width:640px}
.empty b{display:block;font-family:var(--sans);font-weight:500;font-size:18px;color:var(--apricot);margin-bottom:6px}

/* ---------- thread ---------- */
.thread{max-width:860px;display:flex;flex-direction:column;gap:26px}
.thread h2{font-family:var(--sans);font-weight:600;font-size:24px;line-height:1.25;text-wrap:balance}
.turn{display:grid;grid-template-columns:52px minmax(0,1fr);gap:14px;align-items:start}
.turn svg{width:52px;height:68px;display:block}
.turn h3{font-family:var(--sans);font-weight:500;font-size:18px;line-height:1.35}
.answer{border:1px solid var(--line-strong);background:var(--raise);margin-left:66px}
.answer header{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:12px 18px;
  border-bottom:1px solid var(--line);font-size:12px;color:var(--muted)}
.state{font-size:11.5px;padding:3px 10px;border:1px solid var(--line-strong);color:var(--muted)}
.answer.refused{border-color:var(--rust)}
.answer.refused .state{background:var(--rust);border-color:var(--rust);color:var(--apricot)}
.answer.failed .state{background:var(--amber);border-color:var(--amber);color:var(--ground)}
.answer-text{padding:16px 18px 18px;font-size:14.5px;line-height:1.7;white-space:pre-wrap;overflow-wrap:anywhere;max-width:75ch}
.answer.refused .answer-text{color:var(--apricot)}
.caret{display:inline-block;width:8px;height:1.1em;vertical-align:-3px;background:var(--amber);
  animation:blink 1s steps(1) infinite}
@keyframes blink{50%{opacity:0}}
.wait{color:var(--muted)}
.passages{border-top:1px solid var(--line)}
.passages summary{cursor:pointer;list-style:none;display:flex;align-items:center;gap:8px;padding:11px 18px;
  font-size:12.5px;color:var(--muted)}
.passages summary::-webkit-details-marker{display:none}
.passages summary .icon{width:16px;height:16px;transition:transform .2s}
.passages[open] summary .icon{transform:rotate(180deg)}
.passages summary:hover{color:var(--apricot)}
.passages ol{list-style:none;counter-reset:p;padding:0 18px 16px;display:flex;flex-direction:column;gap:8px}
.passages li{counter-increment:p;display:grid;grid-template-columns:30px 76px minmax(0,1fr);gap:10px;
  border:1px solid var(--line);background:var(--ground);padding:10px 12px;font-size:12.5px;line-height:1.55;color:var(--muted)}
.passages li::before{content:counter(p);color:var(--amber);font-weight:700}
.passages .pg{white-space:nowrap}

/* ---------- ledger (evals / document) ---------- */
.ledger-wrap{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.25fr);gap:34px;align-items:start}
.panel{border:1px solid var(--line-strong);background:var(--raise)}
.panel h2{font-family:var(--sans);font-weight:600;font-size:18px;padding:16px 20px;border-bottom:1px solid var(--line)}
.panel .sub{font-size:12px;color:var(--muted);padding:12px 20px;border-top:1px solid var(--line)}
.rows{display:flex;flex-direction:column}
.row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px 16px;padding:14px 20px;border-top:1px solid var(--line)}
.row:first-child{border-top:0}
.row b{grid-column:1;font-family:var(--sans);font-weight:500;font-size:15px}
.row span{grid-column:1;font-size:12px;color:var(--muted)}
.row .v{grid-column:2;grid-row:1 / span 2;align-self:center;font-style:normal;font-size:24px;font-weight:700;
  color:var(--amber);font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:11px 14px;border-top:1px solid var(--line);font-size:12.5px;vertical-align:top}
th{font-weight:400;color:var(--muted);border-top:0}
td.n,th.n{text-align:right}
tr.ship td{color:var(--apricot);background:color-mix(in srgb,var(--amber) 9%,var(--raise))}
tr.ship td:first-child{color:var(--amber)}
td small{display:block;color:var(--muted)}
.doc-view{display:grid;grid-template-columns:220px minmax(0,1fr);gap:34px;align-items:start;max-width:980px}
.doc-view > svg{width:220px;height:293px;display:block}
.doc-view h2{font-family:var(--sans);font-weight:600;font-size:26px;line-height:1.2;text-wrap:balance;margin-bottom:6px}
dl{display:grid;grid-template-columns:150px minmax(0,1fr);border-top:1px solid var(--line);margin-top:18px}
dt,dd{padding:10px 0;border-bottom:1px solid var(--line);font-size:13px}
dt{color:var(--muted)}
.doc-actions{display:flex;gap:14px;margin-top:22px;flex-wrap:wrap}
.doc-actions .primary{width:auto;padding:0 26px;display:inline-flex;align-items:center;gap:10px;text-decoration:none}
.doc-actions .primary:hover{color:var(--apricot)}

/* ---------- player bar composer ---------- */
.player{margin:0 32px 22px;background:var(--oxblood);border:1px solid var(--rust);
  display:grid;grid-template-columns:auto minmax(180px,300px) minmax(0,1fr) auto;align-items:center;gap:18px;
  padding:10px 16px 10px 10px}
.player :focus-visible{outline-color:var(--apricot)}
.player .thumb svg{width:44px;height:58px;display:block;border:1px solid var(--rust)}
.now b{display:block;font-family:var(--mono);font-weight:700;font-size:15px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.now small{display:block;font-size:11.5px;color:color-mix(in srgb,var(--apricot) 75%,var(--oxblood));white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.deck{display:flex;flex-direction:column;gap:8px;min-width:0}
.deck-row{display:flex;align-items:center;gap:12px}
.deck textarea{flex:1;min-width:0;resize:none;background:color-mix(in srgb,var(--ground) 55%,var(--oxblood));
  border:1px solid color-mix(in srgb,var(--apricot) 35%,var(--oxblood));padding:10px 14px;color:var(--apricot);
  font-size:14.5px;line-height:1.45;max-height:120px;outline:0;caret-color:var(--amber)}
.deck textarea::placeholder{color:color-mix(in srgb,var(--apricot) 62%,var(--oxblood))}
.deck textarea:focus{border-color:var(--apricot)}
.send{width:46px;height:46px;border-radius:50%;background:var(--apricot);color:var(--oxblood);display:grid;place-items:center;flex:none}
.send .icon{stroke-width:2.2}
.send:hover{background:var(--amber)}
.send:disabled{opacity:.45;cursor:not-allowed}
.track{position:relative;height:3px;background:color-mix(in srgb,var(--apricot) 26%,var(--oxblood));margin:0 58px 0 2px;overflow:visible}
.track i{position:absolute;inset:0;background:var(--apricot);transform-origin:left;
  transform:scaleX(var(--p,0));transition:transform .12s linear}
.track b{position:absolute;inset:0;transform:translateX(calc(var(--p,0) * 100%));transition:transform .12s linear}
.track b::after{content:"";position:absolute;left:-5px;top:-4px;width:11px;height:11px;border-radius:50%;background:var(--apricot)}
.track.busy i{transition:none;animation:scan 1.3s cubic-bezier(.65,0,.35,1) infinite alternate}
.track.busy b{display:none}
@keyframes scan{from{transform:translateX(0) scaleX(.28)}to{transform:translateX(72%) scaleX(.28)}}
.chips{display:flex;align-items:center;gap:14px}
.readout{font-size:12px;color:color-mix(in srgb,var(--apricot) 75%,var(--oxblood));white-space:nowrap;font-variant-numeric:tabular-nums}
.player .new{width:42px;height:42px;display:grid;place-items:center;border:1px solid color-mix(in srgb,var(--apricot) 35%,var(--oxblood))}
.player .new:hover{background:var(--rust)}

/* ---------- responsive ---------- */
@media (max-width:1360px){
  .proof .more{display:none}
}
@media (max-width:1100px){
  .topbar .proof{display:none}
  .doc-name .wide{display:none}
  .doc-name .narrow{display:inline}
}
@media (max-width:1180px){
  .grid{grid-template-columns:repeat(2,minmax(0,1fr))}
  .readout{display:none}
  .ledger-wrap{grid-template-columns:1fr}
}
@media (max-width:900px){
  body{overflow:hidden}
  #app{margin:0;height:100dvh;border:0;grid-template-columns:minmax(0,1fr);grid-template-rows:auto minmax(0,1fr)}
  .rail{display:none}
  .topbar{height:68px}
  .topbar .search,.topbar .grow,.topbar .proof{display:none}
  .doc-name .wide{display:none}
  .doc-name .narrow{display:inline}
  .doc-thumb{width:68px}
  .doc-thumb svg{width:40px;height:52px}
  .doc-name{width:auto;flex:1;padding:0 14px;border-left:1px solid var(--ground)}
  .doc-name small{display:block;overflow:hidden;text-overflow:ellipsis}
  .bell{width:68px}
  .toolbar{padding:16px 16px 14px;gap:10px}
  .tabs{flex-wrap:nowrap;overflow-x:auto;width:100%}
  .tab{flex:none}
  .toolbar .grow{display:none}
  .filter{flex:1;width:auto}
  .types{padding:0 16px 14px}
  .scroll{padding:0 16px 24px}
  .player{margin:0;border-width:1px 0 0;grid-template-columns:minmax(0,1fr);gap:8px;padding:10px}
  .player .thumb,.player .now,.chips{display:none}
  .track{margin-right:58px}
  .answer{margin-left:0}
  .doc-view{grid-template-columns:1fr}
  .doc-view > svg{width:150px;height:200px}
  dl{grid-template-columns:120px minmax(0,1fr)}
}
@media (max-width:640px){
  .grid{grid-template-columns:minmax(0,1fr)}
  .passages li{grid-template-columns:22px minmax(0,1fr)}
  .passages .pg{grid-column:2}
  .passages li p{grid-column:2}
}
@media (prefers-reduced-motion:reduce){
  .caret{animation:none}
  .track.busy i{animation:none;transform:none;opacity:.5}
  .passages summary .icon{transition:none}
}
</style>
</head>
<body>

<svg width="0" height="0" style="position:absolute" aria-hidden="true">
  <symbol id="i-ask" viewBox="0 0 24 24"><path d="M3 5.5c3-1.3 6-1.3 9 .7 3-2 6-2 9-.7V19c-3-1.3-6-1.3-9 .7-3-2-6-2-9-.7z"/><path d="M12 6.2v13.5"/></symbol>
  <symbol id="i-chat" viewBox="0 0 24 24"><path d="M4 5h16v11H9.5L4 20z"/><path d="M8 9.5h8M8 12.5h5"/></symbol>
  <symbol id="i-history" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/></symbol>
  <symbol id="i-evals" viewBox="0 0 24 24"><path d="M4 20h16M7 16.5v-5M12 16.5V7M17 16.5v-8"/></symbol>
  <symbol id="i-doc" viewBox="0 0 24 24"><path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4M9 12h6M9 16h6"/></symbol>
  <symbol id="i-new" viewBox="0 0 24 24"><path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13 7l4 4"/></symbol>
  <symbol id="i-api" viewBox="0 0 24 24"><path d="M8 4C6 4 6 5 6 7v2c0 1.5-1 2.3-2 3 1 .7 2 1.5 2 3v2c0 2 0 3 2 3M16 4c2 0 2 1 2 3v2c0 1.5 1 2.3 2 3-1 .7-2 1.5-2 3v2c0 2 0 3-2 3"/></symbol>
  <symbol id="i-source" viewBox="0 0 24 24"><circle cx="6" cy="5.5" r="2"/><circle cx="6" cy="18.5" r="2"/><circle cx="18" cy="8" r="2"/><path d="M6 7.5v9M18 10c0 4-4 4.5-10.5 7"/></symbol>
  <symbol id="i-search" viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/></symbol>
  <symbol id="i-filter" viewBox="0 0 24 24"><path d="M4 8h9M17 8h3M4 16h3M11 16h9"/><circle cx="15" cy="8" r="2"/><circle cx="9" cy="16" r="2"/></symbol>
  <symbol id="i-send" viewBox="0 0 24 24"><path d="M5 12h13M13 6l6 6-6 6"/></symbol>
  <symbol id="i-copy" viewBox="0 0 24 24"><rect x="8.5" y="8.5" width="11" height="11"/><path d="M15.5 8.5V4.5h-11v11h4"/></symbol>
  <symbol id="i-edit" viewBox="0 0 24 24"><path d="M4 20h4L19 9l-4-4L4 16z"/></symbol>
  <symbol id="i-trash" viewBox="0 0 24 24"><path d="M4 7h16M9 7V4h6v3M6.5 7l1 13h9l1-13"/></symbol>
  <symbol id="i-down" viewBox="0 0 24 24"><path d="M6 9l6 6 6-6"/></symbol>
  <symbol id="i-open" viewBox="0 0 24 24"><path d="M14 4h6v6M20 4l-9 9M18 14v6H4V6h6"/></symbol>
</svg>

<div id="app">
  <aside class="rail" aria-label="Main">
    <div class="mark" role="img" aria-label="NEURA">
      <svg viewBox="0 0 34 34" aria-hidden="true"><text x="17" y="28" text-anchor="middle" font-family="Hanken Grotesk, sans-serif" font-weight="800" font-size="31" fill="#E8892F">N</text></svg>
    </div>
    <nav>
      <button class="rail-btn" data-go="ask"><svg class="icon"><use href="#i-ask"/></svg>Ask</button>
      <button class="rail-btn" data-go="history"><svg class="icon"><use href="#i-history"/></svg>History</button>
      <button class="rail-btn" data-go="evals"><svg class="icon"><use href="#i-evals"/></svg>Evals</button>
      <button class="rail-btn" data-go="doc"><svg class="icon"><use href="#i-doc"/></svg>Document</button>
    </nav>
    <div class="spacer"></div>
    <nav class="bottom">
      <button class="rail-btn" id="railNew"><svg class="icon"><use href="#i-new"/></svg>New chat</button>
      <a class="rail-btn" href="/docs" target="_blank" rel="noopener"><svg class="icon"><use href="#i-api"/></svg>API</a>
      <a class="rail-btn" href="https://github.com/daivraval/neura-rag-chat" target="_blank" rel="noopener"><svg class="icon"><use href="#i-source"/></svg>Source</a>
    </nav>
  </aside>

  <header class="topbar">
    <label class="search">
      <svg class="icon"><use href="#i-search"/></svg>
      <span class="sr">Search your chats</span>
      <input id="chatSearch" type="search" placeholder="Search your chats" autocomplete="off"/>
    </label>
    <div class="grow"></div>
    <button class="cell proof" id="proof" data-go="evals" hidden title="Open the eval results"><span id="proofText"></span></button>
    <div class="cell doc-thumb" aria-hidden="true"><svg viewBox="0 0 84 112" id="thumbTop"></svg></div>
    <button class="cell doc-name" data-go="doc" aria-label="Open document details">
      <span><b id="docTitle">Loading document…</b><small id="docMeta"><span class="wide">&nbsp;</span><span class="narrow"></span></small></span>
      <svg class="icon"><use href="#i-down"/></svg>
    </button>
    <button class="cell bell" id="topNew" aria-label="New chat" title="New chat"><svg class="icon"><use href="#i-new"/></svg></button>
  </header>

  <main>
    <div class="toolbar">
      <div class="tabs" role="tablist" aria-label="Views">
        <button class="tab" role="tab" data-view="ask" aria-selected="true">Ask</button>
        <button class="tab" role="tab" data-view="chat" aria-selected="false" hidden>Chat</button>
        <button class="tab" role="tab" data-view="history" aria-selected="false">History</button>
        <button class="tab" role="tab" data-view="evals" aria-selected="false">Evals</button>
        <button class="tab" role="tab" data-view="doc" aria-selected="false">Document</button>
      </div>
      <div class="grow"></div>
      <label class="filter" id="filterBox">
        <svg class="icon"><use href="#i-search"/></svg>
        <span class="sr">Filter</span>
        <input id="filter" type="search" placeholder="Search in questions" autocomplete="off"/>
      </label>
      <button class="filter-btn" id="filterBtn" aria-expanded="false" aria-controls="types" title="Filter by question type">
        <svg class="icon"><use href="#i-filter"/></svg><span class="sr">Filter by type</span>
      </button>
    </div>
    <div class="types" id="types" hidden></div>

    <div class="scroll" id="scroll">
      <section class="view" id="view-ask" aria-label="Suggested questions"><div class="grid" id="shelf"></div></section>
      <section class="view" id="view-chat" hidden aria-label="Conversation"><div class="thread" id="thread"></div></section>
      <section class="view" id="view-history" hidden aria-label="Chat history"><div class="grid" id="history"></div></section>
      <section class="view" id="view-evals" hidden aria-label="Eval results"><div id="evals"></div></section>
      <section class="view" id="view-doc" hidden aria-label="Document"><div id="doc"></div></section>
    </div>

    <form class="player" id="composer" autocomplete="off">
      <div class="thumb" aria-hidden="true"><svg viewBox="0 0 84 112" id="thumbPlayer"></svg></div>
      <div class="now"><b id="nowTitle">Ask the paper</b><small id="nowBy">&nbsp;</small></div>
      <div class="deck">
        <div class="deck-row">
          <label class="sr" for="input">Your question</label>
          <textarea id="input" rows="1" placeholder="Ask about the document…" title="Enter to ask, Shift+Enter for a new line"></textarea>
          <button class="send" id="send" type="submit" aria-label="Ask NEURA"><svg class="icon"><use href="#i-send"/></svg></button>
        </div>
        <div class="track" id="track" aria-hidden="true"><i></i><b></b></div>
      </div>
      <div class="chips">
        <span class="readout" id="chipK">k 4 · temp 0</span>
        <button class="new" type="button" id="playerNew" aria-label="New chat" title="New chat"><svg class="icon"><use href="#i-new"/></svg></button>
      </div>
    </form>
  </main>
</div>

<script>
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const REFUSAL = /could not find the answer in the document/i;
const pct = v => v == null ? '—' : (v * 100).toFixed(1) + '%';

/* ---------- covers: one authored tile per question type ---------- */
const COVERS = {
  definition: {bg:'#9F2E10', ink:'#F1B978', motif:'<path d="M20 26h-6v26h6M64 26h6v26h-6" fill="none" stroke-width="3"/><path d="M27 34h30M27 41h22M27 48h26" stroke-width="2"/>'},
  method:     {bg:'#4B3F29', ink:'#E8892F', motif:'<rect x="14" y="24" width="14" height="10" fill="none" stroke-width="2.5"/><rect x="35" y="36" width="14" height="10" fill="none" stroke-width="2.5"/><rect x="56" y="48" width="14" height="10" fill="none" stroke-width="2.5"/><path d="M28 29h14v7M49 41h14v7" fill="none" stroke-width="2"/>'},
  numeric:    {bg:'#E8892F', ink:'#16130E', motif:[0,1,2,3].map(r => [0,1,2,3,4].map(c => `<circle cx="${18+c*12}" cy="${26+r*10}" r="${(r*5+c)%3===0?3:1.6}"/>`).join('')).join('')},
  table:      {bg:'#F1B978', ink:'#5B1408', motif:'<path d="M12 24h60M12 34h60M12 44h60M12 54h60M32 24v30M52 24v30" fill="none" stroke-width="2"/><path d="M12 24h60" stroke-width="4"/>'},
  trap:       {bg:'#16130E', ink:'#9F2E10', motif:'<rect x="22" y="20" width="40" height="40" fill="none" stroke-width="2.5"/><path d="M16 66L68 14" stroke-width="3.5"/>'},
  chat:       {bg:'#4B3F29', ink:'#F1B978', motif:'<path d="M14 22h56v26H34l-12 10V48h-8z" fill="none" stroke-width="2.5"/><path d="M22 31h38M22 39h26" stroke-width="2"/>'},
  doc:        {bg:'#5B1408', ink:'#F1B978', motif:''},
};
function cover(kind, big){
  const c = COVERS[kind] || COVERS.chat;
  return `<rect width="84" height="112" fill="${c.bg}"/>`
    + (kind === 'trap' ? `<rect x=".75" y=".75" width="82.5" height="110.5" fill="none" stroke="#9F2E10" stroke-width="1.5"/>` : '')
    + `<g fill="${c.ink}" stroke="${c.ink}">${c.motif}</g>`
    + `<text x="10" y="98" fill="${c.ink}" font-family="Hanken Grotesk, sans-serif" font-weight="800" font-size="26" letter-spacing="-1">${esc(big)}</text>`;
}
function docCover(){
  return `<rect width="84" height="112" fill="#5B1408"/>`
    + `<rect x="7" y="7" width="70" height="98" fill="none" stroke="#9F2E10" stroke-width="1.2"/>`
    + `<text x="13" y="44" fill="#F1B978" font-family="Hanken Grotesk, sans-serif" font-weight="800" font-size="27" letter-spacing="-1.4">DPR</text>`
    + `<path d="M13 52h58M13 57h44M13 62h52" stroke="#E8892F" stroke-width="1.6"/>`
    + `<rect x="13" y="70" width="34" height="4" fill="#E8892F"/>`
    + `<text x="13" y="96" fill="#F1B978" font-family="Courier Prime, monospace" font-size="8" letter-spacing=".8">EMNLP 2020</text>`;
}
['#thumbTop', '#thumbPlayer'].forEach(s => $(s).innerHTML = docCover());

/* ---------- the shelf: real questions from the golden set ---------- */
const SHELF = [
  {q:'How many passages does the final Wikipedia corpus contain?', type:'numeric', page:4, sec:'§4.1 Wikipedia Data',
   blurb:'One exact figure from the data section. A rounded answer still has to name the right count.'},
  {q:'How does DPR measure the similarity between a question and a passage?', type:'definition', page:3, sec:'§3.1 Overview',
   blurb:'The retriever’s core design choice, defined as Equation 1 in the paper.'},
  {q:'What top-20 retrieval accuracy does the multi-dataset DPR reach on TREC?', type:'table', page:5, sec:'Table 2',
   blurb:'One cell from the main results table, read from a row of ten numbers.'},
  {q:'What exact match score does DPR get on HotpotQA?', type:'trap', page:null, sec:null,
   blurb:'The paper never evaluates on HotpotQA. The only correct answer is a refusal.'},
  {q:'Which three types of negative passages does the paper consider?', type:'method', page:3, sec:'§3.2 Training',
   blurb:'A three-part answer from the training section. Leaving one out counts as wrong.'},
  {q:'How long does it take to build the FAISS index for 21 million passages?', type:'numeric', page:7, sec:'§5.4 Run-time Efficiency',
   blurb:'Indexing cost from the efficiency section, set next to the Lucene comparison.'},
  {q:'Why does the paper think DPR performs worse on SQuAD?', type:'method', page:5, sec:'§5.1 Main Results',
   blurb:'An explanation, not a number. The answer is the paper’s own reasoning about how the dataset was built.'},
  {q:'What exact match score does jointly training the retriever and reader get on Natural Questions?', type:'table', page:8, sec:'§6.2 Results',
   blurb:'The ablation behind the paper’s pipeline choice: training the two parts together instead of apart.'},
  {q:'What does the self-attention mechanism in a transformer compute?', type:'trap', page:null, sec:null,
   blurb:'The model knows this one, but the paper never explains it. NEURA should still refuse.'},
];
const TYPES = ['definition', 'method', 'numeric', 'table', 'trap'];

function shelfCard(item){
  const el = document.createElement('article');
  el.className = 'card';
  el.dataset.type = item.type;
  el.dataset.text = item.q.toLowerCase();
  const trap = item.type === 'trap';
  el.innerHTML = `
    <div class="card-top">
      <div class="cover-col">
        <svg viewBox="0 0 84 112" aria-hidden="true">${cover(item.type, trap ? 'n/a' : 'p.' + item.page)}</svg>
        <span class="meta">${trap ? 'not in<br>the paper' : 'page ' + item.page + '<br>' + item.type}</span>
      </div>
      <div>
        <h3>${esc(item.q)}</h3>
        <p class="by">${trap ? 'Trap question' : `From <a href="/pdf#page=${item.page}" target="_blank" rel="noopener">${esc(item.sec)}</a>`}</p>
        <p class="blurb">${esc(item.blurb)}</p>
      </div>
    </div>
    <button class="primary" type="button">Ask NEURA</button>
    <div class="actions">
      <button class="act" type="button" data-act="edit"><svg class="icon"><use href="#i-edit"/></svg>Edit first</button>
      <button class="act" type="button" data-act="copy"><svg class="icon"><use href="#i-copy"/></svg><span>Copy</span></button>
    </div>`;
  el.querySelector('.primary').onclick = () => ask(item.q);
  el.querySelector('[data-act="edit"]').onclick = () => { input.value = item.q; grow(); input.focus(); };
  el.querySelector('[data-act="copy"]').onclick = async e => {
    const label = e.currentTarget.querySelector('span');
    try { await navigator.clipboard.writeText(item.q); label.textContent = 'Copied'; }
    catch { label.textContent = 'Copy failed'; }
    setTimeout(() => label.textContent = 'Copy', 1400);
  };
  return el;
}
const shelf = $('#shelf');
SHELF.forEach(item => shelf.appendChild(shelfCard(item)));

/* ---------- type filter chips ---------- */
const activeTypes = new Set();
TYPES.forEach(t => {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'type-chip'; b.textContent = t; b.setAttribute('aria-pressed', 'false');
  b.onclick = () => {
    activeTypes.has(t) ? activeTypes.delete(t) : activeTypes.add(t);
    b.setAttribute('aria-pressed', activeTypes.has(t));
    applyFilter();
  };
  $('#types').appendChild(b);
});
$('#filterBtn').onclick = () => {
  const open = $('#types').hidden;
  $('#types').hidden = !open;
  $('#filterBtn').setAttribute('aria-expanded', open);
};

/* ---------- views ---------- */
let view = 'ask', sessionId = null, sessionTitle = '', busy = false;
const scroller = $('#scroll');
const FILTER_HINT = {ask:'Search in questions', history:'Search your chats'};
function setView(v){
  view = v;
  document.querySelectorAll('.tab').forEach(t => t.setAttribute('aria-selected', t.dataset.view === v));
  document.querySelectorAll('.view').forEach(s => s.hidden = s.id !== 'view-' + v);
  document.querySelectorAll('.rail-btn[data-go]').forEach(b => {
    if (b.dataset.go === v) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current');
  });
  const filterable = v === 'ask' || v === 'history';
  $('#filterBox').style.visibility = filterable ? 'visible' : 'hidden';
  $('#filterBtn').style.visibility = v === 'ask' ? 'visible' : 'hidden';
  if (v !== 'ask') { $('#types').hidden = true; $('#filterBtn').setAttribute('aria-expanded', 'false'); }
  if (filterable) $('#filter').placeholder = FILTER_HINT[v];
  if (v === 'history') loadSessions();
  if (v === 'evals') loadEvals();
  if (v !== 'chat') scroller.scrollTop = 0;
  applyFilter();
}
document.querySelectorAll('[data-view]').forEach(t => t.onclick = () => setView(t.dataset.view));
document.querySelectorAll('[data-go]').forEach(b => b.onclick = () => setView(b.dataset.go));

function applyFilter(){
  const q = $('#filter').value.trim().toLowerCase();
  const grid = view === 'history' ? $('#history') : shelf;
  grid.querySelectorAll('.card').forEach(c => {
    const typeOk = view !== 'ask' || !activeTypes.size || activeTypes.has(c.dataset.type);
    c.hidden = !(typeOk && (!q || c.dataset.text.includes(q)));
  });
}
$('#filter').addEventListener('input', applyFilter);
$('#chatSearch').addEventListener('input', e => {
  $('#filter').value = e.target.value;
  if (view !== 'history') setView('history'); else applyFilter();
});

/* ---------- info + evals ---------- */
let info = null;
async function loadInfo(){
  try { info = await (await fetch('/api/info')).json(); } catch { return; }
  const d = info.document;
  $('#docTitle').textContent = d.short;
  $('#docMeta .wide').textContent = `${d.byline} · ${info.chunks} chunks`;
  $('#nowBy').textContent = `By: ${d.byline} | Model: ${info.model}`;
  $('#chipK').textContent = `k ${info.retrieval.k} · temp 0`;
  renderDoc();
}

function renderDoc(){
  const d = info.document;
  const r = info.retrieval;
  $('#doc').innerHTML = `
    <div class="doc-view">
      <svg viewBox="0 0 84 112" aria-hidden="true">${docCover()}</svg>
      <div>
        <h2>${esc(d.title)}</h2>
        <p class="by">${esc(d.authors)}</p>
        <dl>
          <dt>Published</dt><dd>${esc(d.venue)}</dd>
          <dt>License</dt><dd>${esc(d.license)}</dd>
          <dt>Indexed as</dt><dd>${info.chunks} chunks of ${info.chunk_size} characters, ${info.chunk_overlap} overlap</dd>
          <dt>Embeddings</dt><dd>${esc(info.embeddings)}, run locally</dd>
          <dt>Retrieval</dt><dd>${esc(r.search_type)} search, top ${r.k} passages per question</dd>
          <dt>Answers</dt><dd>${esc(info.model)} on ${esc(info.provider)}, temperature 0</dd>
        </dl>
        <div class="doc-actions">
          <a class="primary" href="/pdf" target="_blank" rel="noopener"><svg class="icon"><use href="#i-open"/></svg>Open the PDF</a>
          <a class="act" href="${esc(d.url)}" target="_blank" rel="noopener">ACL Anthology page</a>
        </div>
      </div>
    </div>`;
}

let evalsP = null, evalsLoaded = false;
const getEvals = () => evalsP || (evalsP = fetch('/api/evals').then(r => r.json()));
const refusedCount = a => Math.round(a.refusal_accuracy * a.traps);
async function loadProof(){
  let a;
  try { a = (await getEvals()).answers; } catch { return; }
  if (!a) return;
  $('#proofText').innerHTML = `Traps refused ${refusedCount(a)}/${a.traps}<span class="more"> · Answers correct ${pct(a.answer_accuracy)}</span>`;
  $('#proof').hidden = false;
  $('#docMeta .narrow').textContent = `${refusedCount(a)}/${a.traps} traps refused`;
}
async function loadEvals(){
  if (evalsLoaded) return;
  let e;
  try { e = await getEvals(); } catch { evalsP = null; $('#evals').innerHTML = '<p class="empty">Could not load the eval reports.</p>'; return; }
  evalsLoaded = true;
  const a = e.answers, r = e.retrieval;
  const answers = a ? `
    <section class="panel">
      <h2>Answers</h2>
      <div class="rows">
        <div class="row"><b>Hallucination rate</b><span>${a.traps - refusedCount(a)} of ${a.traps} traps answered anyway</span><em class="v">${pct(a.hallucination_rate)}</em></div>
        <div class="row"><b>Refusal accuracy</b><span>${refusedCount(a)} of ${a.traps} trap questions refused</span><em class="v">${pct(a.refusal_accuracy)}</em></div>
        <div class="row"><b>Answer accuracy</b><span>Every expected fact present</span><em class="v">${pct(a.answer_accuracy)}</em></div>
        <div class="row"><b>False refusals</b><span>Answerable questions refused</span><em class="v">${pct(a.false_refusal_rate)}</em></div>
        <div class="row"><b>Faithfulness</b><span>Judged by ${esc((a.judge || 'an LLM judge').split(' on ')[0])}</span><em class="v">${pct(a.faithfulness)}</em></div>
      </div>
      <p class="sub">${a.questions} questions · ${esc(a.model)} · ${esc(a.generated)}</p>
    </section>` : `
    <section class="panel"><h2>Answers</h2>
      <p class="sub">No answer eval saved yet. Run <code>python -m evals.run_eval generation --judge</code>.</p></section>`;
  const retrieval = r ? `
    <section class="panel">
      <h2>Retrieval</h2>
      <table>
        <thead><tr><th>Config</th><th class="n">Evidence hit</th><th class="n">MRR</th><th class="n">Page hit</th></tr></thead>
        <tbody>${r.configs.map(c => `
          <tr class="${c.name === 'app' ? 'ship' : ''}">
            <td>${c.name === 'app' ? 'shipped' : esc(c.name)}<small>${esc(c.config)}</small></td>
            <td class="n">${pct(c.hit)}</td><td class="n">${c.mrr.toFixed(3)}</td><td class="n">${pct(c.page_hit)}</td>
          </tr>`).join('')}</tbody>
      </table>
      <p class="sub">${r.answerable} answerable questions · offline, no LLM calls · ${esc(r.generated)}</p>
    </section>` : '';
  $('#evals').innerHTML = `<div class="ledger-wrap">${answers}${retrieval}</div>`;
}

/* ---------- history ---------- */
async function loadSessions(){
  const rows = await (await fetch('/api/sessions')).json();
  const box = $('#history'); box.innerHTML = '';
  if (!rows.length){
    box.innerHTML = '<div class="empty"><b>No chats yet</b>Pick a question from the Ask shelf, or type one in the player bar below.</div>';
    return;
  }
  rows.forEach(s => {
    const el = document.createElement('article');
    el.className = 'card';
    el.dataset.text = s.title.toLowerCase();
    const when = new Date(s.created_at);
    const answers = Math.ceil(s.n / 2);
    el.innerHTML = `
      <div class="card-top">
        <div class="cover-col">
          <svg viewBox="0 0 84 112" aria-hidden="true">${cover('chat', String(answers))}</svg>
          <span class="meta">${s.n} messages</span>
        </div>
        <div>
          <h3></h3>
          <p class="by">Started ${when.toLocaleDateString(undefined, {month:'short', day:'numeric'})}, ${when.toLocaleTimeString(undefined, {hour:'2-digit', minute:'2-digit'})}</p>
          <p class="blurb"></p>
        </div>
      </div>
      <button class="primary" type="button">Open chat</button>
      <div class="actions one"><button class="act danger" type="button"><svg class="icon"><use href="#i-trash"/></svg>Delete chat</button></div>`;
    el.querySelector('h3').textContent = s.title;
    el.querySelector('.blurb').textContent = s.preview || 'No answer yet.';
    el.querySelector('.primary').onclick = () => openSession(s.id, s.title);
    el.querySelector('.danger').onclick = async () => {
      await fetch('/api/sessions/' + s.id, {method:'DELETE'});
      if (s.id === sessionId) resetChat();
      loadSessions();
    };
    box.appendChild(el);
  });
  applyFilter();
}

/* ---------- thread ---------- */
const thread = $('#thread');
function showChatTab(title){
  const tab = $('.tab[data-view="chat"]');
  tab.hidden = false;
  sessionTitle = title;
  $('#nowTitle').textContent = title || 'Ask the paper';
}
function threadHeading(title){
  thread.innerHTML = '';
  const h = document.createElement('h2');
  h.textContent = title;
  thread.appendChild(h);
}
function addQuestion(text){
  const el = document.createElement('div');
  el.className = 'turn';
  el.innerHTML = `<svg viewBox="0 0 84 112" aria-hidden="true">${cover('chat', 'Q')}</svg><div><h3></h3></div>`;
  el.querySelector('h3').textContent = text;
  thread.appendChild(el);
}
function addAnswer(){
  const el = document.createElement('article');
  el.className = 'answer';
  el.innerHTML = `<header><span>NEURA · ${esc(info ? info.model : 'model')}</span><span class="state">Answer</span></header>
    <div class="answer-text"></div>`;
  thread.appendChild(el);
  return el;
}
function finishAnswer(el, text, sources){
  const refused = REFUSAL.test(text);
  el.classList.toggle('refused', refused);
  el.querySelector('.state').textContent = refused ? 'Not in the document' : 'Grounded answer';
  if (sources && sources.length){
    const det = document.createElement('details');
    det.className = 'passages';
    det.innerHTML = `<summary>Retrieved passages · ${sources.length}<svg class="icon"><use href="#i-down"/></svg></summary><ol></ol>`;
    const ol = det.querySelector('ol');
    sources.forEach(s => {
      const src = typeof s === 'string' ? {text:s, page:null} : s;
      const li = document.createElement('li');
      li.innerHTML = `<span class="pg">${src.page ? `<a href="/pdf#page=${src.page}" target="_blank" rel="noopener">page ${src.page}</a>` : ''}</span><p></p>`;
      li.querySelector('p').textContent = src.text + '…';
      ol.appendChild(li);
    });
    el.appendChild(det);
  }
}
async function openSession(id, title){
  const data = await (await fetch('/api/sessions/' + id)).json();
  sessionId = id;
  showChatTab(data.title);
  threadHeading(data.title);
  for (const m of data.messages){
    if (m.role === 'user') addQuestion(m.content);
    else { const a = addAnswer(); a.querySelector('.answer-text').textContent = m.content; finishAnswer(a, m.content, m.sources); }
  }
  setView('chat');
  scroller.scrollTop = scroller.scrollHeight;
}
function resetChat(){
  sessionId = null;
  thread.innerHTML = '';
  $('.tab[data-view="chat"]').hidden = true;
  $('#nowTitle').textContent = 'Ask the paper';
  setTrack('idle');
  setView('ask');
  input.focus();
}
['#railNew', '#topNew', '#playerNew'].forEach(s => $(s).onclick = resetChat);

/* ---------- player track: the signature moment ---------- */
const track = $('#track');
function setTrack(mode, p = 0){
  track.classList.toggle('busy', mode === 'busy');
  track.style.setProperty('--p', mode === 'play' ? p / 100 : 0);
}

function typeInto(box, text){
  return new Promise(done => {
    box.textContent = '';
    const caret = document.createElement('span'); caret.className = 'caret';
    box.appendChild(caret);
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const step = reduce ? text.length : Math.max(1, Math.round(text.length / 200));
    let i = 0;
    (function tick(){
      if (i < text.length){
        caret.before(document.createTextNode(text.slice(i, i + step)));
        i += step;
        setTrack('play', Math.min(100, i / text.length * 100));
        scroller.scrollTop = scroller.scrollHeight;
        setTimeout(tick, 12);
      } else { caret.remove(); setTrack('play', 100); done(); }
    })();
  });
}

/* ---------- asking ---------- */
const input = $('#input'), sendBtn = $('#send');
function grow(){ input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 120) + 'px'; }
input.addEventListener('input', grow);
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); ask(input.value); }
});
$('#composer').addEventListener('submit', e => { e.preventDefault(); ask(input.value); });

async function ask(text){
  const q = text.trim();
  if (!q || busy) return;
  busy = true; sendBtn.disabled = true;
  input.value = ''; grow();
  if (!sessionId){ showChatTab(q); threadHeading(q); }
  setView('chat');
  addQuestion(q);
  const a = addAnswer();
  const box = a.querySelector('.answer-text');
  box.innerHTML = '<span class="wait">Retrieving passages and writing an answer…</span>';
  a.querySelector('.state').textContent = 'Working';
  scroller.scrollTop = scroller.scrollHeight;
  setTrack('busy');
  try {
    const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({session_id: sessionId, message: q})});
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    const data = await r.json();
    sessionId = data.session_id;
    await typeInto(box, data.answer);
    finishAnswer(a, data.answer, data.sources);
  } catch (err) {
    setTrack('idle');
    a.classList.add('failed');
    a.querySelector('.state').textContent = 'Request failed';
    box.textContent = `NEURA couldn’t answer (${err.message}). Wait a moment and ask again; the server log has details.`;
  }
  busy = false; sendBtn.disabled = false; input.focus();
}

const startView = location.hash.slice(1);
setView(['history', 'evals', 'doc'].includes(startView) ? startView : 'ask');
loadInfo();
loadProof();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
