# Architecture

Status: starter version from the playbook. The architecture milestone
(docs/milestones/00-architecture.md) expands this into schemas, endpoints and flows.

## Principle
Keep production compute lightweight.
Use hosted model inference.
Keep evaluation reproducible and mostly offline.

## Components
Frontend:
- Next.js App Router
- Vercel

Backend:
- Python
- FastAPI
- LangGraph

Storage:
- Supabase Postgres
- pgvector

Retrieval:
1. hosted query embedding
2. pgvector candidate retrieval
3. hosted reranker
4. top evidence passed to generation

Generation:
- hosted LLM

Independent evaluation:
- separate evaluator model/provider where practical
- executed offline
- results persisted for dashboard

## Source data (PubMedQA, profiled 2026-10-21)
Hugging Face `qiaojin/PubMedQA`:
- `pqa_labeled`: 1,000 rows — `pubid`, `question`, `context{contexts, labels, meshes,
  reasoning_required_pred, reasoning_free_pred}`, `long_answer`, `final_decision`.
- `pqa_unlabeled`: 61,249 rows — same minus `final_decision` and the two `reasoning_*` fields.

Field handling:

| Field | Indexed for retrieval? | Use |
|---|---|---|
| `context.contexts` | YES (the only indexed text) | Evidence passages |
| `context.labels` (BACKGROUND, METHODS, RESULTS…) | Display metadata only | Passage section names |
| `context.meshes` | Not embedded | Distractor selection, topic display |
| `pubid` | Metadata | Citation ID / PubMed link |
| `question` | NEVER as document text | Benchmark query only |
| `long_answer` (conclusion) | NEVER | Not used at runtime |
| `final_decision` | NEVER | Eval gold label only |
| `reasoning_required_pred`, `reasoning_free_pred` | NEVER | Annotator answers — drop at load time |

Contexts are short (median ~200 words, max ~600), so no long-document chunking is needed;
whether a passage is a whole abstract or one section is an architecture-milestone decision.

## Request flow
User
-> Next.js
-> POST /query or streaming equivalent
-> planner
-> retriever
-> evidence gate
-> synthesizer
-> critic
-> deterministic citation validator
-> response

## Graph state
ResearchState:
- request_id
- question
- subquestions
- retrieved_candidates
- selected_evidence
- retrieval_scores
- draft_answer
- claims
- critic_results
- final_answer
- abstained
- timings
- usage
- errors

## Agent nodes
Planner:
Input: question
Output: 1-3 evidence needs
Must not answer question.

Retriever:
Input: evidence needs
Output: ranked evidence

Synthesizer:
Input: question + evidence
Output: structured cited answer

Critic:
Input: answer claims + cited evidence
Output: support decision per claim

## Deterministic gates
Evidence gate:
- applies frozen threshold/rule chosen on development data
- may return insufficient_evidence

Citation validator:
- all citation IDs exist
- all citations came from retrieval
- claims requiring evidence contain citations

## API
GET /health
POST /query
GET /eval/summary
GET /sources/{source_id}

Optional:
GET /query/{request_id}/events

## Query response shape
{
  "request_id": "...",
  "verdict": "yes|no|maybe|insufficient_evidence",
  "answer": "...",
  "claims": [],
  "citations": [],
  "abstained": false,
  "stages": [],
  "timing_ms": {},
  "estimated_cost_usd": 0.0
}

## Evaluation architecture
Frozen test manifest
-> baseline runner
-> agent runner
-> independent judge
-> metric calculator
-> confidence intervals
-> raw JSONL
-> summary JSON
-> dashboard

Never calculate benchmark metrics in the browser.

## Production constraint
No sentence-transformer or cross-encoder model needs to be loaded
in the public web process unless hosted APIs become unavailable.

## Environment variables
DATABASE_URL=
EMBEDDING_API_KEY=
RERANK_API_KEY=
GENERATION_API_KEY=
EVALUATOR_API_KEY=
NEXT_PUBLIC_API_BASE_URL=

Do not place secret values in this file.

## Observability
Log:
- request ID
- stage timings
- retrieval count
- abstention
- token/API usage
- model names
- errors

Never log secrets.

## Deployment
Frontend:
Vercel

Backend:
Railway by default

Database:
Supabase

Model inference:
hosted APIs

## Fallbacks
If pgvector blocks progress:
use Chroma locally while preserving the retrieval interface.

If reranker API blocks progress:
ship dense retrieval first.

If LangGraph blocks progress:
validate the four functions sequentially in plain Python,
then wrap the already-working functions in LangGraph.

If public backend deployment blocks progress:
preserve a local working demo and recorded evidence,
but do not represent a static mock as a live system.
