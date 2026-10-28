// Client for POST /query/stream (ARCHITECTURE §8): Server-Sent Events over a fetch body,
// because EventSource cannot send a POST. Stage events, then one `result` or one `error`.

export type Verdict = "yes" | "no" | "maybe" | "insufficient_evidence";

export type Claim = { id: string; text: string; citation_ids: string[] };

export type Citation = {
  id: string;
  pmid: string;
  passage: string;
  sections: { label: string; text: string }[];
  pubmed_url: string;
  rerank_score: number | null;
};

export type QueryResponse = {
  request_id: string;
  mode: "baseline" | "agent";
  verdict: Verdict;
  answer: string;
  claims: Claim[];
  citations: Citation[];
  limitations: Claim[];
  abstained: boolean;
  abstain_reason: string | null;
};

export type StageEvent = {
  name: string;
  status: "started" | "done" | "skipped" | "failed";
  message: string;
  ms: number;
};

export type SseEvent = { event: string; data: string };

/** Splits SSE text into complete events; returns them plus the unfinished remainder. */
export function parseSse(buffer: string): { events: SseEvent[]; rest: string } {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const blocks = normalized.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: SseEvent[] = [];
  for (const block of blocks) {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (data.length) events.push({ event, data: data.join("\n") });
  }
  return { events, rest };
}

/** Plain-English message for a refused request (422/429/503) or an unreachable API. */
export class AskError extends Error {}

const FALLBACK_MESSAGE = "Something went wrong while answering, please try again later.";

export async function askStream(
  apiBase: string,
  question: string,
  onStage: (stage: StageEvent) => void,
  signal?: AbortSignal,
): Promise<QueryResponse> {
  let response: Response;
  try {
    response = await fetch(`${apiBase}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode: "agent" }),
      signal,
    });
  } catch (err) {
    if (signal?.aborted) throw err;
    throw new AskError("Can't reach the research service right now. Please try again later.");
  }
  if (!response.ok || !response.body) {
    throw new AskError(await refusalMessage(response));
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    const { events, rest } = parseSse(buffer);
    buffer = rest;
    for (const { event, data } of events) {
      if (event === "stage") onStage(JSON.parse(data) as StageEvent);
      else if (event === "result") {
        await reader.cancel();
        return JSON.parse(data) as QueryResponse;
      }
      else if (event === "error") throw new AskError(JSON.parse(data).message || FALLBACK_MESSAGE);
    }
  }
  throw new AskError(FALLBACK_MESSAGE); // stream ended without a result
}

async function refusalMessage(response: Response): Promise<string> {
  if (response.status === 422) return "Please write a question between 5 and 500 characters.";
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail; // 429 / 503 carry a friendly message
  } catch {
    // not JSON
  }
  return FALLBACK_MESSAGE;
}
