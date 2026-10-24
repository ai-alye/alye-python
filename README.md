<img src="https://www.alye.ai/brand/alye-logo.png" alt="Alye" width="120">

# alye (Python)

Records every LLM call your services make: model, tokens, latency, time to first token, errors and retries. You add two lines at startup; calls through the official OpenAI, Anthropic and Google GenAI clients are recorded automatically.

```bash
pip install alye
```

Set your API key as an environment variable (keep it out of your code):

```bash
export ALYE_API_KEY=alye_live_...      # from Setup in Alye
```

```python
import alye

alye.init(
    service="support-reply",          # one name per microservice
    env="production",
)
```

That is all most services need. Calls are sent to Alye at `https://api.alye.ai/ingest`; set `ALYE_ENDPOINT` only if you run your own server. Without an API key the SDK just prints calls to the console. Python 3.10+, no required dependencies.

## LangChain

Nothing extra to set up: after `alye.init()` every LangChain chat model call is recorded, with `invoke`, `stream`, `batch`, async calls, chains, tools, agents (LangGraph) and `with_structured_output`.

- `langchain-openai`, `langchain-anthropic`, `langchain-google-genai` and models built on the OpenAI client (DeepSeek, xAI, Together, ...) are recorded through their client, as above.
- Every other chat model (`ChatBedrockConverse`, `ChatBedrock`, `ChatMistralAI`, `ChatGroq`, `ChatOllama`, `ChatVertexAI`, ...) is recorded through a LangChain callback that Alye adds to every run. A call is never counted twice.

```python
import alye
from langchain_openai import ChatOpenAI

alye.init(service="support-reply", env="production")

llm = ChatOpenAI(model="gpt-5-mini")
llm.invoke("Summarise this ticket: ...")   # recorded with model, tokens, latency and cost
```

## What gets recorded

One span per model call, using the [OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) attribute names:

| Attribute | Meaning |
|---|---|
| `gen_ai.provider.name` | `openai`, `anthropic`, `gcp.gemini`, `gcp.vertex_ai`, `azure.ai.openai`, `deepseek`, ... |
| `gen_ai.operation.name` | `chat`, `responses`, `embeddings`, `generate_content` |
| `gen_ai.request.model` / `gen_ai.response.model` | Model asked for / model that answered |
| `gen_ai.usage.input_tokens` / `output_tokens` | Tokens, including cached input and reasoning output |
| `gen_ai.usage.cache_read.input_tokens`, `cache_creation.input_tokens`, `reasoning.output_tokens` | Token breakdown when the provider reports it |
| `gen_ai.response.finish_reasons`, `gen_ai.response.id` | Why the model stopped |
| `gen_ai.response.time_to_first_chunk` | Seconds to first streamed chunk |
| `error.type` | `rate_limited`, `timeout`, `overloaded`, `server_error`, `auth`, `invalid_request`, `connection_error`, ... |
| `alye.retries`, `alye.retry_statuses` | Retries the provider SDK made inside one call (e.g. after a 429) |
| `alye.usage_unknown` | The call returned but the provider sent no usage; it may still have been billed |
| `alye.customer_id`, `session.id`, `gen_ai.agent.name`, `alye.data_source` | From `alye.context()` |
| `gen_ai.input.messages` / `gen_ai.output.messages` | Prompt and response, only with `capture_content=True` |

Failed calls are recorded even if your code catches the exception. Cost is worked out on the server from a versioned price list, so prices are never stale in your deployed code.

Supported automatically:

- **openai**: `chat.completions.create` and `.parse` (structured outputs), `responses.create` and `.parse`, `embeddings.create`; sync and async; `stream=True`; `.with_raw_response`. Also OpenAI-compatible providers used through the same client.
- **anthropic**: `messages.create` and `beta.messages.create` (sync, async, `stream=True`, `.with_raw_response`) and the `messages.stream()` helper.
- **google-genai**: `models.generate_content`, `generate_content_stream`, `embed_content`; Gemini API and Vertex AI; sync and async.

## Tag calls (optional)

```python
with alye.context(customer_id="northwind", session_id=ticket.id, agent="reply-agent"):
    answer = run_agent(ticket)          # every call inside is tagged
```

Tags flow through `await`. For new threads, start them with `contextvars.copy_context().run(...)`. In middleware you can use `alye.set_tags(...)` instead of a `with` block. Customer tagging is only needed if your app serves several customers; without it, everything is grouped by service.

## Group steps into one trace (optional)

```python
@alye.trace()                         # or alye.trace("answer_ticket")
def answer_ticket(ticket): ...

with alye.span("retrieve_context", data_source="support-kb"):
    docs = search(query)
```

Model calls made inside become children of the span, so one agent run shows as a single trace with its steps.

## Providers that are not patched

```python
with alye.llm_call("voyage", "voyage-3", operation="embeddings") as call:
    res = vo.embed(texts, model="voyage-3")
    call.set_usage(input_tokens=res.total_tokens, output_tokens=0)
```

## Settings

Every option can be passed to `init()` or set as an environment variable.

| `init()` argument | Environment variable | Default |
|---|---|---|
| `api_key` | `ALYE_API_KEY` | none |
| `service` | `ALYE_SERVICE` or `OTEL_SERVICE_NAME` | `unknown_service` (with a warning) |
| `env` | `ALYE_ENV` | `production` |
| `endpoint` | `ALYE_ENDPOINT` | `https://api.alye.ai/ingest` when an API key is set; without a key, calls are printed to the console |
| `region` | `ALYE_REGION` | none: where calls are processed, for regional pricing (`eu`, `us`, or a region like `europe-west4`); Bedrock and Vertex regions are detected automatically |
| `capture_content` | `ALYE_CAPTURE_CONTENT` | `False` |
| `redact_pii` | `ALYE_REDACT_PII` | `True` |
| `sample_rate` | `ALYE_SAMPLE_RATE` | `1.0` (errors are always kept) |
| `exporter` | `ALYE_EXPORTER` | `otlp` (`console`, `none`, or an exporter object) |
| `instrument` | | every installed provider (e.g. `["openai"]`) |
| `debug` | `ALYE_DEBUG` | `False` |
| `disabled` | `ALYE_DISABLED` | `False` |
| `release` | `ALYE_RELEASE` | none |

Other options: `redact_patterns=[(regex, replacement)]`, `redact=callable`, `max_content_chars` (8000), `inject_stream_usage` (True: asks OpenAI streams to include usage), `flush_interval` (2 s), `max_batch` (256), `max_queue` (10,000), `timeout` (5 s), `heartbeat_interval` (60 s), `resource_attributes`.

## Privacy

- Prompts and responses are **not** sent unless you set `capture_content=True`.
- When they are, emails, phone numbers, card numbers (Luhn-checked) and API keys are replaced with `[EMAIL]`, `[PHONE]`, `[CARD]`, `[SECRET]` before anything leaves the process. Add your own patterns with `redact_patterns`.
- Content is truncated to `max_content_chars`.

## How it sends data

- Spans are queued in memory and sent in batches from a background thread, so your request path never waits on the network. If the queue fills up, new spans are dropped and counted rather than slowing you down.
- Format: OTLP/HTTP JSON, gzip, to `{endpoint}/v1/traces` with an `x-api-key` header. Retries 429 and 5xx with backoff.
- A heartbeat goes to `{endpoint}/v1/heartbeat` every 60 s with the SDK version, patched libraries and queue stats, so the backend can tell a quiet service from one that stopped reporting.
- Works with forking servers (gunicorn, uWSGI): workers restart their own sender after fork.
- In short-lived scripts, jobs and serverless handlers, call `alye.flush()` before exiting. `atexit` also flushes on normal exit.

## Local development

```bash
python -m alye.devserver           # prints every call it receives
ALYE_ENDPOINT=http://localhost:4318 python your_service.py
```

Without an API key (and no endpoint) the SDK prints one line per call to stderr, so nothing leaves your laptop.

## Tests

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

The tests run the real `openai`, `anthropic` and `google-genai` clients against fake HTTP servers, so no API keys or network are needed.

## License

Apache License 2.0. Copyright Onruyl Private Limited. See [LICENSE](LICENSE).
