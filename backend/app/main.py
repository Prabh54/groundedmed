"""FastAPI app (ARCHITECTURE §8): /health, /query (baseline mode), /sources/{pmid}.

Run locally: uv run uvicorn backend.app.main:app --reload
Errors: 422 invalid input; 429 rate limit or daily cap; 503 provider or database unavailable.
Never a mocked answer. One JSON log line per query goes to stdout (ARCHITECTURE §10).
"""

import functools
import json
import os
import subprocess
import uuid

import psycopg
from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.baseline.pipeline import answer_baseline
from backend.app.config import (
    DEFAULT_DAILY_QUERY_CAP,
    LOCAL_ORIGINS,
    PER_IP_QUERIES_PER_HOUR,
    REPO_ROOT,
    redact,
)
from backend.app.guardrails.rate_limit import RateLimiter
from backend.app.llm.client import anthropic_client
from backend.app.models.answer import FinalAnswer
from backend.app.models.api import (
    Citation,
    QueryRequest,
    QueryResponse,
    SourceResponse,
    Stage,
    pubmed_url,
)
from backend.app.models.retrieval import RetrievalError
from backend.app.retrieval.embed import voyage_client
from backend.app.retrieval.vector_search import connect, get_document

LIMIT_MESSAGE = "Demo limit reached, please try again later."
UNAVAILABLE_MESSAGE = "The service is temporarily unavailable, please try again later."


def allowed_origins(env_value: str) -> list[str]:
    """Comma-separated ALLOWED_ORIGINS (production Vercel domain) plus localhost (§14.8)."""
    return [o.strip().rstrip("/") for o in env_value.split(",") if o.strip()] + LOCAL_ORIGINS


def daily_cap(env_value: str | None) -> int:
    """DAILY_QUERY_CAP if it is a positive integer, else the default (never crash at start-up)."""
    try:
        cap = int(env_value or "")
    except ValueError:
        return DEFAULT_DAILY_QUERY_CAP
    return cap if cap > 0 else DEFAULT_DAILY_QUERY_CAP


app = FastAPI(title="GroundedMed API", description="Research demonstration, not medical advice.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(os.environ.get("ALLOWED_ORIGINS", "")),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
# In-memory limits: run a single uvicorn worker, or each worker gets its own counters.
limiter = RateLimiter(PER_IP_QUERIES_PER_HOUR, daily_cap(os.environ.get("DAILY_QUERY_CAP")))


@functools.cache
def version() -> str:
    sha = os.environ.get("RAILWAY_GIT_COMMIT_SHA")
    if sha:
        return sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@functools.cache
def clients():
    return voyage_client(), anthropic_client()


@app.get("/health")
def health():
    try:
        with connect() as conn:
            conn.execute("select 1 from documents limit 1")
    except Exception:
        body = {"status": "error", "version": version(), "db": "unreachable"}
        return JSONResponse(status_code=503, content=body)
    return {"status": "ok", "version": version(), "db": "ok"}


def to_response(request_id: str, a: FinalAnswer) -> QueryResponse:
    retrieval_ms = a.timing_ms.get("retrieval", 0.0)
    stages = [
        Stage(name="retriever", status="done", ms=retrieval_ms,
              message=f"Selected {len(a.evidence)} evidence passages"
              + (" (reranker unavailable: similarity order)" if a.rerank_skipped else "")),
        Stage(name="synthesizer", status="done" if a.gate_passed else "skipped",
              ms=a.timing_ms.get("synthesis", 0.0),
              message=f"Wrote {len(a.claims)} cited claims" if a.gate_passed
              else "Skipped: evidence too weak to answer"),
    ]  # fmt: skip
    return QueryResponse(
        request_id=request_id,
        mode="baseline",
        verdict=a.verdict,
        answer=a.answer,
        claims=a.claims,
        citations=[
            Citation(
                id=c.citation_id,
                pmid=c.pmid,
                passage=c.passage,
                sections=c.sections,
                pubmed_url=pubmed_url(c.pmid),
                rerank_score=c.rerank_score,
            )
            for c in a.evidence
        ],  # fmt: skip
        limitations=a.limitations,
        abstained=a.abstained,
        abstain_reason=a.abstain_reason,
        rerank_skipped=a.rerank_skipped,
        stages=stages,
        timing_ms={
            "planner": 0.0,
            "retrieval": retrieval_ms,
            "synthesis": a.timing_ms.get("synthesis", 0.0),
            "critic": 0.0,
            "total": a.timing_ms.get("total", 0.0),
        },
        estimated_cost_usd=round(a.estimated_cost_usd, 6),
        prompt_versions=a.prompt_versions,
    )


def log(record: dict) -> None:
    print(json.dumps(record), flush=True)


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request):
    request_id = str(uuid.uuid4())
    ip = request.client.host if request.client else "unknown"
    refused = limiter.check(ip)
    if refused:
        log({"request_id": request_id, "event": "rate_limited", "reason": refused})
        raise HTTPException(status_code=429, detail=LIMIT_MESSAGE)

    try:
        voyage, llm = clients()
        with connect() as conn:
            answer = answer_baseline(body.question, conn=conn, voyage=voyage, llm=llm)
    except (RetrievalError, psycopg.Error) as exc:  # embedding or database failure
        log({"request_id": request_id, "event": "error", "stage": "retrieval",
             "error": redact(f"{type(exc).__name__}: {exc}")[:300]})  # fmt: skip
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE) from None

    log({
        "request_id": request_id, "mode": "baseline", "question": body.question,
        "verdict": answer.verdict, "abstained": answer.abstained,
        "abstain_reason": answer.abstain_reason, "evidence": [c.pmid for c in answer.evidence],
        "rerank_skipped": answer.rerank_skipped, "rerank_error": answer.rerank_error,
        "timing_ms": answer.timing_ms,
        "usage": [u.model_dump() for u in answer.usage], "cost_usd": answer.estimated_cost_usd,
        "validation": answer.validation.model_dump(), "error": answer.error,
    })  # fmt: skip
    if answer.abstain_reason == "error":  # LLM failed: plain message, never a partial answer
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE)
    return to_response(request_id, answer)


@app.get("/sources/{pmid}", response_model=SourceResponse)
def source(pmid: str = Path(pattern=r"^\d{1,9}$")):
    try:
        with connect() as conn:
            doc = get_document(conn, pmid)
    except (RetrievalError, psycopg.Error):
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE) from None
    if doc is None:
        raise HTTPException(status_code=404, detail="No such document in the corpus.")
    return SourceResponse(id=f"PMID:{pmid}", pubmed_url=pubmed_url(pmid), **doc)
