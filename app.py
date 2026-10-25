"""Web service for The Librarian: serves the chat page and answers questions with the RAG pipeline.

Run locally (from the repo root):
    uvicorn app:app --port 8000            # then open http://localhost:8000
On Render, render.yaml starts it with the port Render provides.

Environment:
    OPENAI_API_KEY          required (read from .env locally, the host's secret settings when deployed)
    LIBRARIAN_MIN_SCORE     "I don't know" when the best vector score is below this (default 0: the LLM decides)
    LIBRARIAN_K             chunks sent to the LLM (default 5)
    LIBRARIAN_RATE_PER_MIN  questions per visitor per minute (default 10)
    LIBRARIAN_DAILY_CAP     questions per day for the whole service (default 300), protects the OpenAI bill
    LIBRARIAN_CACHE_SIZE    answers kept in memory; a repeated question is served free (default 256, 0 = off)
    HF_LOGS_REPO, HF_TOKEN  if set, logs/ is pushed every 5 minutes to that private Hugging Face dataset,
                            for hosts whose disk is wiped on every restart

Every request is appended to logs/requests-<run>.jsonl (latency, tokens, cost: the Week 8 "before" numbers) and
every 👍/👎 rating to logs/feedback-<run>.jsonl. One file per server run, so a restarted
server never overwrites the files it pushed before. logs/ holds user questions, so it is never committed.
"""
import json, os, threading, time, uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from generate import IDK, LLM_MODEL, Librarian

ROOT = Path(__file__).parent
LOGS = ROOT / "logs"
RUN = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
MIN_SCORE = float(os.getenv("LIBRARIAN_MIN_SCORE", "0"))
K = int(os.getenv("LIBRARIAN_K", "5"))
CACHE_SIZE = int(os.getenv("LIBRARIAN_CACHE_SIZE", "256"))
lib = scheduler = None
log_lock, cache_lock = threading.Lock(), threading.Lock()
cache = OrderedDict()                       # (question, rewrite, types) -> answer, least recently used first


class Limiter:
    """In-memory caps for the public URL: questions per visitor per minute, and per day for everyone."""
    def __init__(self, per_min, per_day):
        self.per_min, self.per_day = per_min, per_day
        self.recent, self.day, self.count, self.lock = {}, None, 0, threading.Lock()

    def check(self, visitor):
        """Returns an error message, or None if the question may go ahead."""
        t, today = time.time(), time.strftime("%Y-%m-%d", time.gmtime())
        with self.lock:
            if today != self.day:
                self.day, self.count = today, 0
            if self.count >= self.per_day:
                return "This demo has reached its daily question limit. Please come back tomorrow."
            hits = [x for x in self.recent.get(visitor, []) if t - x < 60]
            if len(hits) >= self.per_min:
                return "Too many questions in a minute. Please wait a moment and try again."
            self.recent[visitor] = hits + [t]
            self.count += 1
            if len(self.recent) > 5000:        # forget visitors idle for a minute
                self.recent = {v: h for v, h in self.recent.items() if t - h[-1] < 60}
        return None


limiter = Limiter(int(os.getenv("LIBRARIAN_RATE_PER_MIN", "10")), int(os.getenv("LIBRARIAN_DAILY_CAP", "300")))


@asynccontextmanager
async def lifespan(app):
    global lib, scheduler
    LOGS.mkdir(exist_ok=True)
    if os.getenv("HF_LOGS_REPO"):
        from huggingface_hub import CommitScheduler
        scheduler = CommitScheduler(repo_id=os.environ["HF_LOGS_REPO"], repo_type="dataset", folder_path=LOGS,
                                    every=5, private=True, token=os.getenv("HF_TOKEN"))
    lib = Librarian(str(ROOT / "index"))   # load the index and the embedding model once, not per request
    yield
    if scheduler:                          # push what is left before the container stops
        scheduler.trigger().result(timeout=60)


app = FastAPI(title="The Librarian", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    rewrite: bool = False
    types: list[Literal["doc", "issue", "release"]] | None = None   # metadata filter; None searches everything


class Feedback(BaseModel):
    request_id: str = Field(max_length=64)
    helpful: bool
    comment: str = Field(default="", max_length=1000)


def log(name, row):
    # the scheduler's lock stops a push from reading a half-written line
    with (scheduler.lock if scheduler else log_lock), open(LOGS / f"{name}-{RUN}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"ok": lib is not None, "model": LLM_MODEL, "chunks": len(lib.chunks), "k": K, "min_score": MIN_SCORE,
            "logs_synced_to_hub": scheduler is not None, "questions_today": limiter.count,
            "daily_cap": limiter.per_day}


@app.post("/api/ask")
def ask(req: Ask, request: Request):      # sync endpoint: FastAPI runs it in a worker thread
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Please type a question.")
    refusal = limiter.check(request.client.host if request.client else "unknown")
    if refusal:
        raise HTTPException(429, refusal)
    rid = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    key = (" ".join(question.lower().split()), req.rewrite, tuple(sorted(req.types or [])))
    with cache_lock:
        res = cache.get(key)
        if res:
            cache.move_to_end(key)
    cached = res is not None
    if cached:                              # same question asked before: no search, no LLM call, no cost
        res = dict(res, prompt_tokens=0, completion_tokens=0, tokens=0, cost_usd=0.0)
    else:
        try:
            res = lib.answer(question, k=K, rewrite=req.rewrite, min_score=MIN_SCORE, types=req.types)
        except Exception as e:              # OpenAI errors, missing key: log details, show only the type
            log("requests", dict(id=rid, ts=now(), question=question, error=repr(e)[:500]))
            raise HTTPException(502, f"The language model call failed ({type(e).__name__}). Check the server log.")
        if CACHE_SIZE:
            with cache_lock:
                cache[key] = res
                if len(cache) > CACHE_SIZE:
                    cache.popitem(last=False)
    latency = time.perf_counter() - t0

    cited = {c["n"] for c in res["citations"]}
    sources = [dict(n=n, id=c["id"], type=c["type"], title=c["title"], section=c["section"], url=c["url"],
                    date=c["date"], score=round(c["score"], 3), cited=n in cited, text=c["text"])
               for n, c in enumerate(res["sources"], 1)]
    log("requests", dict(id=rid, ts=now(), question=question, query=res["query"], rewrite=req.rewrite,
                         types=req.types, cached=cached, idk=res["answer"].strip() == IDK,
                         latency_s=round(latency, 3),
                         prompt_tokens=res["prompt_tokens"], completion_tokens=res["completion_tokens"],
                         cost_usd=res["cost_usd"], best_score=res["best_score"],
                         source_ids=[s["id"] for s in sources], cited=sorted(cited),
                         bad_citations=res["bad_citations"], model=LLM_MODEL, k=K, min_score=MIN_SCORE))
    return dict(request_id=rid, answer=res["answer"], query=res["query"], sources=sources,
                bad_citations=res["bad_citations"], latency_s=round(latency, 2), tokens=res["tokens"],
                cost_usd=res["cost_usd"], model=LLM_MODEL, cached=cached)


@app.post("/api/feedback")
def feedback(fb: Feedback):
    log("feedback", dict(fb.model_dump(), ts=now()))
    return {"ok": True}
