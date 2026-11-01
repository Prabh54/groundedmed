"""FastAPI app (ARCHITECTURE §8): /health, /query and /query/stream (baseline or agent mode),
/sources/{pmid}.

Run locally: uv run uvicorn backend.app.main:app --reload
Errors: 422 invalid input; 429 rate limit or daily cap; 503 provider or database unavailable.
Never a mocked answer. One JSON log line per query goes to stdout (ARCHITECTURE §10).
"""

import asyncio
import functools
import json
import os
import queue
import subprocess
import threading
import uuid

import psycopg
from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.app.agents.graph import NODE_ORDER
from backend.app.agents.pipeline import Emit, ShouldStop, StageEvent, answer_agent
from backend.app.baseline.pipeline import answer_baseline
from backend.app.config import (
    DEFAULT_DAILY_QUERY_CAP,
    LOCAL_ORIGINS,
    PER_IP_QUERIES_PER_HOUR,
    REPO_ROOT,
    redact,
)
from backend.app.guardrails.rate_limit import (
    CANDIDATE_IP_HEADERS,
    RateLimiter,
    client_key,
    is_trusted,
    pseudonym,
    trusted_networks,
)
from backend.app.llm.client import Cancelled, anthropic_client
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
INTERNAL_MESSAGE = "Something went wrong while answering, please try again later."
STREAM_POLL_SECONDS = 0.05  # how often the stream checks for a new event or a departed visitor


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
# The header holding the visitor's address, believed only from these proxy addresses (ADR-039,
# ADR-041; both measured on the host at M6-T4). Either one empty = believe no header.
CLIENT_IP_HEADER = os.environ.get("CLIENT_IP_HEADER", "").strip().lower()
TRUSTED_PROXIES, _invalid_proxies = trusted_networks(os.environ.get("TRUSTED_PROXY_CIDRS", ""))
# Off unless DEBUG_CLIENT_IP=1: lets a caller see how the server sees their own request (M6-T4).
DEBUG_CLIENT_IP = os.environ.get("DEBUG_CLIENT_IP", "").strip() == "1"


@functools.cache
def version() -> str:
    sha = os.environ.get("RENDER_GIT_COMMIT")  # set by Render for each deploy
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


def citations(a: FinalAnswer) -> list[Citation]:
    return [
        Citation(id=c.citation_id, pmid=c.pmid, passage=c.passage, sections=c.sections,
                 pubmed_url=pubmed_url(c.pmid), rerank_score=c.rerank_score)
        for c in a.evidence
    ]  # fmt: skip


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
        citations=citations(a),
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


def agent_stages(a: FinalAnswer, events: list[StageEvent]) -> list[Stage]:
    """One entry per node in graph order: its final event, or `skipped` if it never ran."""
    last = {name: (status, message, ms) for name, status, message, ms in events}
    stages = []
    for name in NODE_ORDER:
        if name in last and last[name][0] != "started":
            status, message, ms = last[name]
            stages.append(Stage(name=name, status=status, message=message, ms=ms))
        else:
            why = "evidence too weak to answer" if not a.gate_passed else "an earlier step failed"
            stages.append(Stage(name=name, status="skipped", message=f"Skipped: {why}", ms=0.0))
    return stages


def to_agent_response(request_id: str, a: FinalAnswer, events: list[StageEvent]) -> QueryResponse:
    t = a.timing_ms
    return QueryResponse(
        request_id=request_id, mode="agent", verdict=a.verdict, answer=a.answer, claims=a.claims,
        citations=citations(a), limitations=a.limitations, abstained=a.abstained,
        abstain_reason=a.abstain_reason, rerank_skipped=a.rerank_skipped,
        stages=agent_stages(a, events),
        timing_ms={"planner": t.get("planner", 0.0), "retrieval": t.get("retriever", 0.0),
                   "rerank": t.get("retriever_rerank", 0.0), "synthesis": t.get("synthesizer", 0.0),
                   "critic": t.get("critic", 0.0), "total": t.get("total", 0.0)},
        estimated_cost_usd=round(a.estimated_cost_usd, 6), prompt_versions=a.prompt_versions,
    )  # fmt: skip


def log(record: dict) -> None:
    print(json.dumps(record), flush=True)


if _invalid_proxies:
    log({"event": "config_warning", "setting": "TRUSTED_PROXY_CIDRS",
         "ignored_entries": _invalid_proxies})  # fmt: skip


def limit_key(request: Request) -> tuple[str | None, str]:
    """(connecting peer, the address the per-IP limit counts) for one request."""
    peer = request.client.host if request.client else None
    # Two copies may mean the proxy added its own after a caller's: believe neither.
    values = request.headers.getlist(CLIENT_IP_HEADER) if CLIENT_IP_HEADER else []
    return peer, client_key(peer, values[0] if len(values) == 1 else None, TRUSTED_PROXIES)


def ip_header_counts(request: Request) -> dict[str, int]:
    names = dict.fromkeys(CANDIDATE_IP_HEADERS + ((CLIENT_IP_HEADER,) if CLIENT_IP_HEADER else ()))
    return {n: len(request.headers.getlist(n)) for n in names if request.headers.getlist(n)}


def check_limits(request: Request, request_id: str) -> None:
    peer, ip = limit_key(request)
    # The peer is the hosting proxy's address in production (how TRUSTED_PROXY_CIDRS is measured,
    # M6-T4); header values and the visitor's own address are never logged, only a pseudonym.
    xff = [e for v in request.headers.getlist("x-forwarded-for") for e in v.split(",") if e.strip()]
    log({"request_id": request_id, "event": "client", "peer": peer,
         "trusted_peer": is_trusted(peer, TRUSTED_PROXIES), "ip_headers": ip_header_counts(request),
         "xff_entries": len(xff), "client_id": pseudonym(ip)})  # fmt: skip
    refused = limiter.check(ip)
    if refused:
        log({"request_id": request_id, "event": "rate_limited", "reason": refused})
        raise HTTPException(status_code=429, detail=LIMIT_MESSAGE)


def answer(
    body: QueryRequest,
    request_id: str,
    emit: Emit | None = None,
    should_stop: ShouldStop | None = None,
) -> QueryResponse:
    """Run one question in the requested mode, log it, and build the response. Raises 503 for a
    retrieval/database failure or a model failure (never a partial answer), and Cancelled (logged
    with the cost so far) when `should_stop()` turned true before a step started."""
    events: list[StageEvent] = []

    def record(event: StageEvent) -> None:
        events.append(event)
        if emit:
            emit(event)

    state = None
    try:
        voyage, llm = clients()
        with connect() as conn:
            if body.mode == "agent":
                result, state = answer_agent(body.question, conn=conn, voyage=voyage, llm=llm,
                                             request_id=request_id, emit=record,
                                             should_stop=should_stop)  # fmt: skip
            else:
                result = answer_baseline(body.question, conn=conn, voyage=voyage, llm=llm,
                                         should_stop=should_stop)  # fmt: skip
    except Cancelled as exc:  # visitor left: no further paid call was started
        log({"request_id": request_id, "mode": body.mode, "event": "cancelled",
             "stopped_before": exc.before, "usage": [u.model_dump() for u in exc.usage],
             "cost_usd": sum(u.cost_usd for u in exc.usage)})  # fmt: skip
        raise
    except (RetrievalError, psycopg.Error) as exc:  # embedding or database failure
        spent = getattr(exc, "usage", [])  # calls made before the failure (agent planner)
        log({"request_id": request_id, "mode": body.mode, "event": "error", "stage": "retrieval",
             "error": redact(f"{type(exc).__name__}: {exc}")[:300],
             "usage": [u.model_dump() for u in spent],
             "cost_usd": sum(u.cost_usd for u in spent)})  # fmt: skip
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE) from None

    record_line = {
        "request_id": request_id, "mode": body.mode, "question": body.question,
        "verdict": result.verdict, "abstained": result.abstained,
        "abstain_reason": result.abstain_reason, "evidence": [c.pmid for c in result.evidence],
        "rerank_skipped": result.rerank_skipped, "rerank_error": result.rerank_error,
        "timing_ms": result.timing_ms,
        "usage": [u.model_dump() for u in result.usage], "cost_usd": result.estimated_cost_usd,
        "validation": result.validation.model_dump(), "error": result.error,
    }  # fmt: skip
    if state is not None:
        record_line |= {
            "subqueries": [s.query for s in state["subquestions"]],
            "candidates": len(state["retrieved_candidates"]),
            "planner_added_evidence": state["planner_added_evidence"],
            "critic": [{"id": r.claim_id, "action": r.action} for r in state["critic_results"]],
            "critic_removed": state["critic_removed"],
        }
    log(record_line)
    if result.abstain_reason == "error":  # LLM failed: plain message, never a partial answer
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE)
    if body.mode == "agent":
        return to_agent_response(request_id, result, events)
    response = to_response(request_id, result)
    if emit:  # baseline has no live stages: report them once the answer is ready
        for s in response.stages:
            emit((s.name, s.status, s.message, s.ms))
    return response


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request):
    request_id = str(uuid.uuid4())
    check_limits(request, request_id)
    return answer(body, request_id)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class StopOnExit(StreamingResponse):
    """Sets `stop` however the response ends: answer sent, visitor disconnected (the server
    cancels the response) or a failed write (M6-F2)."""

    def __init__(self, content, stop: threading.Event, **kwargs):
        super().__init__(content, **kwargs)
        self.stop = stop

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.stop.set()


@app.post("/query/stream")
def query_stream(body: QueryRequest, request: Request):
    """Server-Sent Events: `stage` events ({name, status, message, ms}), then one `result` event
    (the /query response) or one `error` event with a plain message. Limits and input errors are
    still answered as HTTP 429/422 before the stream starts.

    If the visitor disconnects, `stop` is set: the step already running finishes, no later step
    (so no further model call) starts, and the cost so far is logged (M6-F2)."""
    request_id = str(uuid.uuid4())
    check_limits(request, request_id)
    events: queue.Queue = queue.Queue()
    stop = threading.Event()

    def emit(event: StageEvent) -> None:
        name, status, message, ms = event
        events.put(sse("stage", {"name": name, "status": status, "message": message,
                                 "ms": round(ms, 1)}))  # fmt: skip

    def work() -> None:
        try:
            events.put(sse("result", answer(body, request_id, emit, stop.is_set).model_dump()))
        except Cancelled:  # nobody to tell; answer() logged it
            pass
        except HTTPException as exc:
            events.put(sse("error", {"message": exc.detail}))
        except Exception as exc:  # a bug: no traceback or secrets to the client, logged here
            log({"request_id": request_id, "event": "error", "stage": "stream",
                 "error": redact(f"{type(exc).__name__}: {exc}")[:300]})  # fmt: skip
            events.put(sse("error", {"message": INTERNAL_MESSAGE}))
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    async def stream():
        # Never block the event loop on the queue: a blocked read would delay noticing the
        # disconnect until the next event, by which time the next step has already started.
        while not await request.is_disconnected():
            try:
                chunk = events.get_nowait()
            except queue.Empty:
                await asyncio.sleep(STREAM_POLL_SECONDS)
                continue
            if chunk is None:
                return
            yield chunk

    return StopOnExit(stream(), stop, media_type="text/event-stream",
                      headers={"Cache-Control": "no-cache"})  # fmt: skip


@app.get("/debug/client-ip")
def debug_client_ip(request: Request):
    """The caller's own request as the server sees it, for measuring the proxy (M6-T4). Returns
    only what the caller sent or their own proxy chain; nothing is logged or counted. 404 unless
    DEBUG_CLIENT_IP=1."""
    if not DEBUG_CLIENT_IP:
        raise HTTPException(status_code=404, detail="Not Found")
    peer, ip = limit_key(request)
    headers = {n: request.headers.getlist(n) for n in ip_header_counts(request)}
    return {"peer": peer, "trusted_peer": is_trusted(peer, TRUSTED_PROXIES),
            "client_ip_header": CLIENT_IP_HEADER or None, "limit_key": ip,
            "headers": headers}  # fmt: skip


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
