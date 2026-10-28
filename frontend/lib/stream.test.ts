import { test } from "node:test";
import assert from "node:assert/strict";
import { AskError, askStream, parseSse } from "./stream.ts";

test("parseSse returns complete events and keeps the unfinished tail", () => {
  const text = 'event: stage\ndata: {"a":1}\n\nevent: result\ndata: {"b"';
  const { events, rest } = parseSse(text);
  assert.deepEqual(events, [{ event: "stage", data: '{"a":1}' }]);
  assert.equal(rest, 'event: result\ndata: {"b"');
});

test("parseSse handles CRLF line endings and multi-line data", () => {
  const { events, rest } = parseSse("event: x\r\ndata: one\r\ndata: two\r\n\r\n");
  assert.deepEqual(events, [{ event: "x", data: "one\ntwo" }]);
  assert.equal(rest, "");
});

function streamResponse(chunks: string[], status = 200): Response {
  const body = new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(new TextEncoder().encode(c));
      controller.close();
    },
  });
  return new Response(body, { status, headers: { "Content-Type": "text/event-stream" } });
}

function withFetch(fake: typeof fetch, run: () => Promise<void>) {
  const real = globalThis.fetch;
  globalThis.fetch = fake;
  return run().finally(() => {
    globalThis.fetch = real;
  });
}

test("askStream reports stages and returns the result, even when split mid-event", async () => {
  const result = { verdict: "yes", claims: [], citations: [], limitations: [], abstained: false };
  const full = `event: stage\ndata: {"name":"planner","status":"done","message":"Planned 3 searches","ms":1}\n\nevent: result\ndata: ${JSON.stringify(result)}\n\n`;
  const chunks = [full.slice(0, 17), full.slice(17, 90), full.slice(90)];
  let sentBody = "";
  await withFetch(
    (async (_url: string, init: RequestInit) => {
      sentBody = String(init.body);
      return streamResponse(chunks);
    }) as typeof fetch,
    async () => {
      const stages: string[] = [];
      const answer = await askStream("http://api", "Q?", (s) => stages.push(s.message));
      assert.deepEqual(stages, ["Planned 3 searches"]);
      assert.equal(answer.verdict, "yes");
    },
  );
  assert.deepEqual(JSON.parse(sentBody), { question: "Q?", mode: "agent" });
});

test("askStream turns an error event into an AskError with its message", async () => {
  await withFetch(
    (async () => streamResponse(['event: error\ndata: {"message":"Service down"}\n\n'])) as typeof fetch,
    async () => {
      await assert.rejects(askStream("http://api", "Q?", () => {}), (e) => e instanceof AskError && e.message === "Service down");
    },
  );
});

test("askStream uses the API's friendly 429 message", async () => {
  await withFetch(
    (async () => new Response(JSON.stringify({ detail: "Demo limit reached, please try again later." }), { status: 429 })) as typeof fetch,
    async () => {
      await assert.rejects(askStream("http://api", "Q?", () => {}), /Demo limit reached/);
    },
  );
});

test("askStream explains an unreachable API and a stream that ends with no result", async () => {
  await withFetch(
    (async () => {
      throw new TypeError("fetch failed");
    }) as typeof fetch,
    async () => {
      await assert.rejects(askStream("http://api", "Q?", () => {}), /Can't reach the research service/);
    },
  );
  await withFetch(
    (async () => streamResponse(['event: stage\ndata: {"name":"planner","status":"started","message":"","ms":0}\n\n'])) as typeof fetch,
    async () => {
      await assert.rejects(askStream("http://api", "Q?", () => {}), /Something went wrong/);
    },
  );
});

function byteResponse(parts: Uint8Array[]): Response {
  const body = new ReadableStream({
    start(controller) {
      for (const p of parts) controller.enqueue(p);
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

test("askStream decodes a multi-byte character split across chunks and a CRLF split across chunks", async () => {
  const text = `event: result\r\ndata: {"verdict":"no","answer":"Sjögren","claims":[],"citations":[],"limitations":[],"abstained":false}\r\n\r\n`;
  const bytes = new TextEncoder().encode(text);
  const cutInsideO = bytes.indexOf(0xc3) + 1; // between the two bytes of "ö"
  const cutInsideCrlf = bytes.indexOf(0x0d) + 1; // between \r and \n
  const parts = [bytes.slice(0, cutInsideCrlf), bytes.slice(cutInsideCrlf, cutInsideO), bytes.slice(cutInsideO)];
  await withFetch(
    (async () => byteResponse(parts)) as typeof fetch,
    async () => {
      const answer = await askStream("http://api", "Q?", () => {});
      assert.equal(answer.answer, "Sjögren");
    },
  );
});

test("askStream gives a plain message for a 422 whose detail is a list", async () => {
  await withFetch(
    (async () => new Response(JSON.stringify({ detail: [{ msg: "String should have at least 5 characters" }] }), { status: 422 })) as typeof fetch,
    async () => {
      await assert.rejects(askStream("http://api", "Q?", () => {}), /between 5 and 500 characters/);
    },
  );
});

test("askStream passes an abort through instead of reporting a connection problem", async () => {
  const ctrl = new AbortController();
  await withFetch(
    (async (_url: string, init: RequestInit) => {
      ctrl.abort();
      throw init.signal?.reason ?? new DOMException("aborted", "AbortError");
    }) as typeof fetch,
    async () => {
      await assert.rejects(askStream("http://api", "Q?", () => {}, ctrl.signal), (e) => !(e instanceof AskError));
    },
  );
});
