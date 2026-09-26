# Mini RAG Chatbot Backend — Submission

A FastAPI backend that answers company-policy questions over WebSocket using
Retrieval-Augmented Generation (RAG) and enforces role-based access control
(RBAC) at the vector-query layer. High/low-level design is in
[app/docs/ARCHITECTURE.md](app/docs/ARCHITECTURE.md); the intent router contract
is in [app/docs/INTENT_ROUTING.md](app/docs/INTENT_ROUTING.md).

## 1. How to run the ingestion script

The scanned executive document (`app/docs/exec/strategic_plan.pdf`) needs the
Tesseract binary in addition to the `pytesseract` wheel:

```bash
brew install tesseract          # macOS
# apt install tesseract-ocr     # Debian/Ubuntu
```

Ingestion extracts text (with OCR fallback), chunks it, embeds each chunk with a
local `sentence-transformers/all-MiniLM-L6-v2` model, and stores vectors plus
metadata in a Qdrant collection. Stop any process holding the embedded store
(the chat server keeps `./app/qdrant_db/.lock` once retrieval has run), then:

```bash
python -m app.data.ingestion --source-dir app/docs --qdrant-path ./app/qdrant_db --recreate-collection
```

`--qdrant-path` always uses the local embedded database. Extraction and chunking
run first; a missing PDF or OCR failure aborts before the collection is dropped.

## 2. How to run the server

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env          # then set JWT_SECRET (>= 32 chars) and an LLM key
python3 mint_tokens.py        # prints one JWT per mock user
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/` — the server also serves `test_client.html`, so it
uses the same local origin and the default `ws://localhost:8000/ws/chat`. Paste
a token and chat. The LLM provider is pluggable (OpenRouter / Groq / Gemini / Ollama). The
answer-generation model is `meta-llama/llama-3.3-70b-instruct` through
OpenRouter by default. Intent classification uses
`qwen/qwen3-30b-a3b-instruct-2507`, with a bounded output-token limit. Both paths
share the provider interface in `app/orchestration/llm.py`.

The generation model is selected for grounded response quality and streaming;
the smaller non-thinking Qwen instruct model is used for the router's narrow
JSON classification task. Override the provider and model through environment
settings where supported.
## 3. How to test

Run the automated suite:

```bash
pytest # (or) .venv/bin/python -m pytest
```

Or drive the WebSocket by hand. The endpoint is `ws://localhost:8000/ws/chat`
and expects the auth frame first, then `message` frames, streamed back token by
token:

```
Client -> {"type": "auth", "token": "<jwt>"}
Server <- {"type": "auth_success", "user_id": "emp-001", "department": "hr", "level": 1}

Client -> {"type": "message", "text": "How many leave days do I get?"}
Server <- {"type": "stream", "text": "You are entitled to "}
Server <- {"type": "stream", "text": "22 working days..."}
Server <- {"type": "done"}
```

Invalid or expired tokens get `auth_failed` and a close code `1008`. A
tool/LLM failure sends a single `{"type": "error", "message": "..."}` frame and
keeps the connection open for the next message. The same protocol is spoken by
`test_client.html`, which is the client used for evaluation.

## 4. Chunking strategy

Chunks follow document headings rather than a fixed sliding window across the
whole file:

- The parser keeps heading lines intact and splits numbered headings that
  OCR/PDF layout glued onto the following sentence (`2. Membership The
  committee...`).
- The chunker starts a new chunk at each `N. Title` heading and each
  `Priority N:` line; the title/preamble is its own opening section.
- **Chunk size 450 words, overlap 75 words** — but overlap is applied only when
  a single section exceeds ~450 words, and a continuation never crosses a
  heading (it repeats the heading instead).

Why: these policy documents are organised into numbered clauses, and each clause
is a self-contained unit. Cutting on heading boundaries keeps a whole clause in
one chunk (so the LLM sees the rule plus its conditions together), while the
fallback size/overlap handles unheaded or oversized text. The 450/75 numbers
are a starting point, not a magic constant — they keep chunks small enough to
embed precisely and give a modest overlap so a clause split across a page break
retains its context.

## 5. RBAC approach

Authorization is applied **at the vector-database query**, not by filtering a
Python list after retrieval.

`app/auth/rbac.py::AccessScope.for_user(user)` turns a verified identity into a
scope:

- departments: `("hr",)` for HR users, otherwise `("hr", user.department)` — HR
  content is company-wide, and a user's own department is the only other scope;
- `maximum_access_level = user.level`.

`scope.qdrant_filter()` produces a Qdrant `must` filter:

```json
{
  "must": [
    {"key": "department", "match": {"any": ["hr", "exec"]}},
    {"key": "access_level", "range": {"lte": 3}}
  ]
}
```

`Retriever.search` passes that filter straight into `QdrantStore.search`, so
unauthorized chunks are never returned at all. Doing it in the DB layer matters
for two reasons: it is defense-in-depth (even if downstream code is wrong, the
data was never fetched), and it is cheaper than fetching the top-k then
discarding most of them — and it can't accidentally leak an unauthorized chunk
into the prompt. The JWT is the only source of the scope; the router's
`domain_category` is never used for access control.

## 6. Async / parallel approach

The personal-assistant tool exposes one operation,
`get_employee_context(user_id)`, which must fetch three pieces of mock data.
Each mock MCP operation simulates a one-second call and reads its own table
under `app/sample_data`:

- `get_profile(user_id)` → `{name, grade}` (`profile.xlsx`)
- `get_manager(user_id)` → `{manager}` (`manager.xlsx`)
- `get_team(user_id)` → `{team_size, team_name}` (`team info.xlsx`)

They are run concurrently with `asyncio.gather`, not awaited one after another:

```python
profile, manager, team = await asyncio.gather(
    get_profile(user_id),
    get_manager(user_id),
    get_team(user_id),
)
return {**profile, **manager, **team}
```

So three 1-second calls finish in **~1 second instead of ~3 seconds**. Two
things make this correct in an async context: the fan-out uses `asyncio.gather`
(not sequential `await`s), and each table read runs through `asyncio.to_thread`
so parsing the spreadsheet never blocks the WebSocket event loop. The same
"offload blocking work" rule is applied to RAG retrieval, where embedding and
the Qdrant query also run via `asyncio.to_thread`.

## 7. One thing you would improve given more time

Replace the mock Excel-backed employee data with an authenticated HR-system
integration, and build a retrieval evaluation set from representative policy
questions so changes to chunking and prompts can be measured.

## 8. One tradeoff you made and why

The project uses local sentence-transformer embeddings and embedded Qdrant to
keep the demo inexpensive and self-contained. The tradeoff is CPU latency and a
one-time model download, plus a single-process local vector store rather than a
shared production service.

## 9. Turn logs (Optional)

Every chat turn appends one JSON object per line to `logs/chat_turns.jsonl`
(configurable with `CHAT_LOG_FILE`) and mirrors the same lines to stderr, so a
request can be followed end to end:

| Event | Answers |
| --- | --- |
| `turn.start` | who asked what, and how much history was in play |
| `router.result` | the route and domain category the router chose, and why |
| `tool.selected` | which tool actually ran |
| `rag.input` / `rag.retrieved` / `rag.output` | the RAG question, the chunks retrieved (with the RBAC scope used), and the final answer |
| `employee_db.read.start` / `.done` | the employee-data read, before it hits the tables and after, with the resolved context |
| `turn.end` | total turn latency |

All lines from one turn share a `turn_id`. Failures are logged too
(`employee_db.read.error`, `rag.retrieve.error`, `rag.output.error`); a logging
failure is reported to stderr and never breaks a turn.

To follow a single turn:

```bash
grep <turn_id> logs/chat_turns.jsonl | python -m json.tool --json-lines
```

For real employee data, configure log access and retention carefully: turn logs
include query and response text as well as employee context.
