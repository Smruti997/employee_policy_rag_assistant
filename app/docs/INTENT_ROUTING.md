# LLM Intent Routing

`IntentRouter.route()` asynchronously asks the configured LLM provider to classify
each user message. The router requests a JSON object with this contract:

```json
{
  "route": "RAG | PERSONAL_INFO | BOTH | GREETING | CLOSURE | DIRECT_RESPONSE",
  "domain_category": "hr | finance | exec | personal | general",
  "extracted_keywords": ["short phrase from the request"]
}
```

| Route | Application action |
| --- | --- |
| `RAG` | Search Qdrant using the authenticated user's payload filter; stream an answer grounded in returned excerpts. |
| `PERSONAL_INFO` | Call `get_employee_context` with the authenticated JWT subject; stream an answer grounded in that record. |
| `BOTH` | Fetch employee context and filtered policy results concurrently; stream one answer from both contexts. |
| `GREETING` / `CLOSURE` | Return a short direct response without retrieval. |
| `DIRECT_RESPONSE` | Explain that the assistant handles company policy and employee information. |

The system instructions distinguish personal requests such as "Who is my direct
manager?" from policy wording such as "Does an expense need manager approval?".
The model never supplies an employee ID to the data tool. Only the authenticated
server-side subject is passed to it.

The response is parsed and validated against the allowed routes and categories.
If classification is unavailable or malformed, the server returns a safe
recoverable WebSocket `error`; it does not silently broaden the request into
unfiltered retrieval. Select the LLM using `LLM_PROVIDER` and set that provider's
key/model variables as described in the submission README.
