// Fake GroundedMed API for $0 frontend testing (no model, database or network calls).
// Mirrors POST /query/stream from backend/app/main.py. Every answer is labelled MOCK.
// Run: npm run mock-api   (port 8001; start the app with NEXT_PUBLIC_API_BASE_URL=http://localhost:8001)
// Trigger tokens in the question (never natural words): __weak__ -> evidence gate refusal,
// __abstain__ -> model abstains but keeps cited claims, __crash__ -> error event,
// __limit__ -> HTTP 429, __slow__ -> 3 s per stage.
import http from "node:http";

const PORT = Number(process.env.MOCK_PORT || 8001);
const ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const sse = (event, data) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

const citation = (pmid) => ({
  id: `PMID:${pmid}`,
  pmid,
  passage: `MOCK passage text for PMID ${pmid}.`,
  sections: [{ label: "RESULTS", text: `MOCK passage text for PMID ${pmid}.` }],
  pubmed_url: `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`,
  rerank_score: 0.9,
});

function result(question, kind) {
  const base = { request_id: "mock", mode: "agent", rerank_skipped: false, stages: [], timing_ms: {}, estimated_cost_usd: 0, prompt_versions: {} };
  if (kind === "weak") {
    return { ...base, verdict: "insufficient_evidence", answer: "MOCK: The retrieved evidence scored below the relevance threshold.", claims: [], citations: [], limitations: [], abstained: true, abstain_reason: "weak_evidence" };
  }
  if (kind === "abstain") {
    // validator.py keeps validated claims when the model itself answers insufficient_evidence
    return {
      ...base, verdict: "insufficient_evidence", answer: "MOCK claim text joined", abstained: true, abstain_reason: "model_abstained",
      claims: [{ id: "c1", text: "MOCK: the study measured something related, not the question asked.", citation_ids: ["PMID:33333333"] }],
      citations: [citation("33333333")], limitations: [],
    };
  }
  return {
    ...base,
    verdict: "no",
    answer: "MOCK answer",
    claims: [
      { id: "c1", text: `MOCK finding one for: ${question}`, citation_ids: ["PMID:11111111"] },
      { id: "c2", text: "MOCK finding two, supported by two sources.", citation_ids: ["PMID:11111111", "PMID:22222222"] },
    ],
    citations: [citation("11111111"), citation("22222222")],
    limitations: [{ id: "l1", text: "MOCK limitation: small sample.", citation_ids: ["PMID:22222222"] }],
    abstained: false,
    abstain_reason: null,
  };
}

// [name, started message, done message] — same wording style as backend/app/agents/pipeline.py
const STAGES = [
  ["planner", "Planning evidence needs", "Planned 3 search queries"],
  ["retriever", "Searching the corpus", "Retrieved 36 candidate passages; selected 5 evidence passages"],
  ["gate", "Checking evidence relevance", "Evidence passed the relevance threshold"],
  ["synthesizer", "Writing a cited answer", "Drafted 2 claims and 1 limitations"],
  ["critic", "Checking 3 claims and limitations", "Checked 3 claims and limitations"],
  ["validator", "Validating citations", "Kept 2 cited claims; removed 0 unsupported items"],
];

const server = http.createServer(async (req, res) => {
  const origin = req.headers.origin;
  if (ORIGINS.includes(origin)) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST");
  }
  if (req.method === "OPTIONS") return res.writeHead(204).end();
  if (req.method === "GET" && req.url === "/health") {
    return res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({ status: "ok", version: "mock" }));
  }
  if (req.method !== "POST" || req.url !== "/query/stream") return res.writeHead(404).end();

  let raw = "";
  for await (const chunk of req) raw += chunk;
  let question = "";
  try {
    question = String(JSON.parse(raw).question ?? "").trim();
  } catch {
    // fall through to 422
  }
  const json = (status, body) => res.writeHead(status, { "Content-Type": "application/json" }).end(JSON.stringify(body));
  if (question.length < 5 || question.length > 500) return json(422, { detail: [{ msg: "invalid question" }] });
  if (question.includes("__limit__")) return json(429, { detail: "Demo limit reached, please try again later." });

  res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
  const delay = question.includes("__slow__") ? 3000 : 500;
  for (const [i, [name, startMessage, message]] of STAGES.entries()) {
    res.write(sse("stage", { name, status: "started", message: startMessage, ms: 0 }));
    await sleep(delay);
    if (question.includes("__crash__") && i === 3) {
      res.write(sse("error", { message: "The service is temporarily unavailable, please try again later." }));
      return res.end();
    }
    if (question.includes("__weak__") && name === "gate") {
      // real graph: failed gate goes straight to the validator (no synthesizer/critic)
      res.write(sse("stage", { name, status: "done", message: "Evidence scored below the relevance threshold", ms: delay }));
      res.write(sse("stage", { name: "validator", status: "started", message: "Validating citations", ms: 0 }));
      res.write(sse("stage", { name: "validator", status: "done", message: "No answer: the evidence scored below the relevance threshold", ms: 1 }));
      res.write(sse("result", result(question, "weak")));
      return res.end();
    }
    res.write(sse("stage", { name, status: "done", message, ms: delay }));
  }
  res.write(sse("result", result(question, question.includes("__abstain__") ? "abstain" : "answer")));
  res.end();
});

server.listen(PORT, () => console.log(`MOCK GroundedMed API on http://localhost:${PORT}`));
