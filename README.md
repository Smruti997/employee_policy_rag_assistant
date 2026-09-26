# Take-Home Assignment — Mini RAG Chatbot Backend

**Time budget:** ~4–6 hours of focused work. You have 3 days to submit.

## What we're looking for

We want to see **production-quality code at small scale**, not a prototype. Clear decisions, reasonable error handling, and honest documentation of tradeoffs matter more than a long feature list.

You can use any tools at your disposal. What we care about is whether **you understand what you produced**. Do not paste output you cannot explain.

---

## The problem

Build a FastAPI backend that exposes a chatbot over WebSocket. The chatbot answers questions about company policy documents using Retrieval-Augmented Generation (RAG) and enforces role-based access control (RBAC) so users only see documents they are authorised to access.

---

## Requirements

### 1. Document ingestion

- Source documents are provided under `./docs/` organised by department:
  - `./docs/hr/` — HR policies (some accessible to all employees, some only to managers)
  - `./docs/finance/` — Finance policies
  - `./docs/exec/` — Executive-only documents (one of these is a scanned/image-based PDF)
- Write a script or endpoint that:
  - Extracts text from each PDF (handle both text-based and scanned PDFs — your choice of approach)
  - Chunks the text with a justified strategy
  - Generates embeddings — **pick anything that works**. A local model via `sentence-transformers` (e.g. `all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`) is perfectly fine for this task and costs nothing. Hosted options (OpenAI, Cohere, Voyage) are also acceptable if you prefer.
  - Stores chunks + metadata in a vector database of your choice (Qdrant or Chroma recommended)
- **Metadata on each chunk must include:**
  - `department` — one of `hr`, `finance`, `exec` (derived from the folder)
  - `access_level` — integer, see mapping below
  - `source_file` — original file name
  - `page` — page number if applicable

**Access level mapping:**

| File | department | access_level |
|------|-----------|-------------|
| `hr/leave_policy.pdf` | hr | 1 |
| `hr/code_of_conduct.pdf` | hr | 1 |
| `hr/performance_review.pdf` | hr | 2 |
| `finance/expense_policy.pdf` | finance | 1 |
| `finance/travel_reimbursement.pdf` | finance | 1 |
| `exec/compensation_committee.pdf` | exec | 3 |
| `exec/strategic_plan.pdf` | exec | 3 |

### 2. WebSocket chat endpoint

Expose `/ws/chat` with the following protocol:

**Client sends first:**
```json
{"type": "auth", "token": "<jwt>"}
```

**Server responds:**
```json
{"type": "auth_success", "user_id": "...", "department": "...", "level": ...}
```
or
```json
{"type": "auth_failed", "message": "..."}
```

**Client then sends messages:**
```json
{"type": "message", "text": "How many leave days do I get?"}
```

**Server streams back:**
```json
{"type": "stream", "text": "You are entitled to "}
{"type": "stream", "text": "22 working days..."}
{"type": "done"}
```

### 3. RBAC at retrieval time

When the user sends a message, retrieve relevant chunks **filtered at the vector DB query layer** (not by filtering Python lists after retrieval) using:

```
department == user.department  AND  access_level <= user.level
```

There is one exception: any user can access `hr` content regardless of their primary department. Think about the cleanest way to express this.

### 4. Test users

A helper script `mint_tokens.py` is provided that mints mock JWTs for 3 users. Use the signing key in `.env` (`JWT_SECRET`). The JWT payload looks like:

```json
{
  "sub": "emp-001",
  "email": "emp@test.com",
  "department": "hr",
  "level": 1,
  "iat": ...,
  "exp": ...
}
```

Three users are pre-defined:

| user_id | department | level |
|---------|-----------|-------|
| emp-001 | hr | 1 |
| mgr-002 | hr | 2 |
| exec-003 | exec | 3 |

### 5. Parallel tool call

Give the LLM one tool: `get_employee_context(user_id)`.

Internally, this tool must fetch three pieces of mock data **in parallel** (not sequentially):

- profile (mock: `asyncio.sleep(1)` then return `{"name": "Jane Doe", "grade": "Senior"}`)
- manager info (mock: `asyncio.sleep(1)` then return `{"manager": "John Smith"}`)
- team info (mock: `asyncio.sleep(1)` then return `{"team_size": 8, "team_name": "Platform"}`)

Total elapsed time for the tool call should be ~1 second, not ~3 seconds.

The LLM should be able to invoke this tool when the user asks something personal like "how much leave do I have given my grade" or "who is my manager".

### 6. LLM and embeddings — use free-tier options

We do not expect you to spend money on this assignment. Any of these are fine:

**LLMs (all have free tiers, no credit card needed for basic use):**
- **Groq** — free tier with Llama 3.3 70B, very fast streaming (`https://console.groq.com`)
- **Google Gemini** — free tier on `gemini-2.0-flash` (`https://aistudio.google.com`)
- **OpenRouter** — has several free-tier models
- **Ollama** — fully local, runs Llama / Qwen / Mistral on your machine (no API at all)
- Hosted paid providers (OpenAI, Anthropic) are also fine if you happen to have a key

**Embeddings (free, local):**
- `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5` via the `sentence-transformers` Python package. Downloads a small model once (~100MB) and runs on CPU.

Whichever combination you choose, keep it pluggable behind a thin interface so we can swap in our own provider/key without rewriting your code. State your choice in the submission README.

### 7. Error handling

At minimum:

- Invalid / expired / missing JWT → send `auth_failed` and close the connection
- LLM or tool failure → send a user-facing error, keep the connection open
- Do **not** introduce blocking calls in `async def` handlers

---

## How to test your server

A browser-based test client is provided at `test_client.html`. It speaks the exact protocol described above. To use it:

1. Copy `.env.example` to `.env` and set `JWT_SECRET` to any string you like.
2. Generate tokens: `python3 mint_tokens.py` — this prints three JWTs (one per test user). Copy one.
3. Start your server (e.g. `uvicorn app.main:app --reload`).
4. Open `test_client.html` in a browser — no build step required, just double-click the file.
5. Paste the JWT, confirm the WebSocket URL (default `ws://localhost:8000/ws/chat`), and click **Connect**.
6. Type a message and press Enter. Streamed tokens should appear progressively, not in one big chunk.

Your server must accept the protocol exactly as specified — the provided client is what we will use to evaluate your submission. You are free to also support other clients (curl/wscat/Postman) but the HTML client must work end-to-end.

---

## What to submit

Zip the whole project and email it back, or push to a GitHub repo and send the link.

Include:

- Your source code
- Your `.env.example` (do not include real keys — we will use our own)
- A `README_SUBMISSION.md` covering:
  1. How to run the ingestion script
  2. How to run the server
  3. How to test (example WebSocket exchange, or a small test HTML/script)
  4. **Chunking strategy** — chunk size, overlap, why you picked them
  5. **RBAC approach** — how you enforced filtering, why at the DB layer
  6. **Async/parallel approach** — how the 3-API tool avoids serial `await`
  7. **One thing you would improve** given more time
  8. **One tradeoff you made** and why
- Keep the LLM provider pluggable behind a small interface so we can swap it. State in the submission README which provider you picked and why.

---

## Files you are given

```
.
├── README.md                   # this file
├── docs/                       # sample PDFs to ingest
│   ├── hr/
│   │   ├── leave_policy.pdf
│   │   ├── code_of_conduct.pdf
│   │   └── performance_review.pdf      (access_level 2)
│   ├── finance/
│   │   ├── expense_policy.pdf
│   │   └── travel_reimbursement.pdf
│   └── exec/
│       ├── compensation_committee.pdf  (access_level 3)
│       └── strategic_plan.pdf          (scanned/image — OCR required)
├── mint_tokens.py              # generates JWTs for the 3 test users
├── test_client.html            # browser-based WebSocket test client
├── .env.example                # environment variables template
└── generate_pdfs.py            # ignore — used to produce the sample PDFs
```

---

## Evaluation rubric

| Area | What we are looking for |
|------|------------------------|
| WebSocket + streaming | Correct protocol, real token-by-token streaming |
| parallel APIs | Correctly parallel, ~1s total not ~3s |
| No blocking calls in `async def` |  |
| RBAC at the DB layer | Vector DB filter applied at query time, not post-filter in Python |
| Chunk strategy | Defensible choice; not a magic number copied from a tutorial |
| Scanned PDF handling | Works without manual intervention |
| Error handling | Reasonable, not over-engineered |
| Code quality | Type hints, Pydantic models, no dead code, small functions |
| README thinking | Shows judgment and tradeoffs |

### Stretch (optional, not required to score well)

- Reranker (Cohere, BGE, or cross-encoder)
- Hybrid search (BM25 + vector)
- Conversation memory across turns

---

## Ground rules

- If a requirement is unclear, make a reasonable assumption and document it in the submission README. We would rather see your judgment than be pedantic about wording.
- Show your commits. We care about how you iterate, not just the final result.
- If you get stuck on something core (e.g. WebSockets), tell us in the README what you tried and what you would do next.

Good luck.
