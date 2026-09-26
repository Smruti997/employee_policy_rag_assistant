# Intent Routing

The first orchestration step classifies the user's message with **one LLM call**.
The classifier routes messages to the policy, employee-data, or direct-response paths.

The router reuses the same pluggable provider as the answer path, with its own
model selected by `ROUTER_MODEL` (default `qwen/qwen3-30b-a3b-instruct-2507`)
via `app/orchestration/llm.py`:

```python
provider = provider_from_env(model=settings.router_model)
raw = await complete(provider, messages, max_tokens=settings.router_max_tokens)
decision = parse_classification(raw)
```

`complete()` collects a provider stream into a single string, so one-shot callers
reuse the streaming provider protocol rather than a second interface.

## Output contract

The model is asked for a JSON object and nothing else:

```json
{
  "route": "RAG | PERSONAL_INFO | BOTH | GREETING | CLOSURE | DIRECT_RESPONSE",
  "domain_category": "hr | finance | exec | personal | general",
  "reasoning": "one short sentence",
  "keywords": ["..."]
}
```

`parse_classification` tolerates code fences and stray prose by taking the first
`{` to the last `}`. Labels are normalised for casing and spacing, so
`"personal info"` still resolves to `PERSONAL_INFO`. `keywords` populates
`RoutingDecision.extracted_keywords`.

## Routes

| Route | Criteria | Response |
| --- | --- | --- |
| `RAG` | Company policy or one of its terms, however terse | `PolicyRAGTool` with the RBAC filter |
| `PERSONAL_INFO` | The user's own record — name, grade, manager, team, department | `get_employee_context(user_id)` |
| `BOTH` | The user's record together with a policy | Employee context, then filtered RAG |
| `GREETING` | hello, hi, hey, good morning/afternoon/evening | Time-of-day greeting (IST) |
| `CLOSURE` | bye, goodbye, thanks, that's all, no more questions | Sign-off message |
| `DIRECT_RESPONSE` | Only clearly unrelated topics: geography, sport, weather, small talk | `unable to provide answer` |

The prompt states explicitly that a bare topic phrase is `RAG`. An earlier
decision-model router sent `"what long term incentives"` and `"tell me about
strategic plan"` to `DIRECT_RESPONSE` with ~0.5 confidence, so those never
reached retrieval and the user got `unable to provide answer`. The prompt now
carries worked examples for exactly those phrasings.

`domain_category` is used for observability and for spotting out-of-scope asks;
it is never used for access control. The RBAC filter comes from the verified JWT
claims.

## Fallback and failure behaviour

Classification never raises into the request path, and a question is never
answered from a route it could not classify. When any of these happen:

- no LLM provider is configured,
- the provider call fails or times out,
- the reply contains no parsable JSON object, or
- the returned route label is unknown,

the router returns the internal `Route.ERROR` and the orchestrator replies with
the user-facing `FALLBACK_MESSAGE`:

> I am unable to generate the response at this moment, please try after sometime
> or try with switching to another LLM

The reason for the failure is logged and recorded on the decision, but only the
retry message reaches the client. `Route.ERROR` is deliberately not offered to
the model as a classifiable route.

## `ROUTER_MAX_TOKENS` matters more than it looks

The cap must stay *bounded*: OpenRouter pre-authorises `max_tokens` against the
account balance, and omitting it requests the model's full window, which a
low-balance key cannot cover — the call returns HTTP 402 instead of an answer.

For the default non-thinking model, 512 is plenty for the JSON plus its short
reasoning. A *reasoning* model (such as DeepSeek R1) spends hundreds of tokens
thinking before it emits content, so:

- if `max_tokens` is too low, the reply is truncated to **empty content with no
  error**, and the router can only treat it as unparsable (fallback message), and
- even when it is not truncated, R1 occasionally emits only its reasoning and no
  content at all — measured 1027 reasoning characters and 0 content characters on
  `"which department I belong to"`, with `finish_reason: stop`.

The non-thinking `qwen3-30b-a3b-instruct` was chosen for the router precisely to
avoid that failure mode.

## Why not a reasoning model on the critical path

Routing runs before retrieval on every message. DeepSeek R1, measured during
this build:

| Query | Router latency |
| --- | --- |
| `what long term incentives` | 24.2s |
| `tell me about strategic plan` | 14.7s |
| `who is my manager` | 8.9s |
| `hi` | 6.0s |

The same prompt on `qwen3-30b-a3b-instruct-2507`:

| Query | Router latency |
| --- | --- |
| `which department I belong to` | 1.3s |
| `who is my manager` | 1.0s |
| `what long term incentives` | 1.0s |
| `whats acme expense policy` | 0.8s |

Qwen3 30B classifies the six routes correctly, in about a second, without the
empty-content behaviour. Switching model remains a single `ROUTER_MODEL` change.

## Notes

- `IntentRouter.route` is `async` and the provider uses `httpx.AsyncClient`; a
  synchronous call here would block the WebSocket event loop.
- The call is capped by `router_max_tokens` from `app.core.config`.
