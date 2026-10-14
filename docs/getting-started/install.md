# Install

Jahan is a Python engine with an optional local web interface.

## Requirements

- Python 3.12 or newer, managed with [uv](https://docs.astral.sh/uv/)
- Node.js, only for the web interface

## The engine

```bash
git clone https://github.com/fm882/jahan.git
cd jahan
uv sync                    # the engine and its test dependencies
uv run pytest              # optional: the whole suite, no network and no API keys
```

Optional extras:

| Extra | Adds | Install |
|---|---|---|
| `web` | the HTTP API the interface talks to (FastAPI, uvicorn) | `uv sync --extra web` |
| `otel` | exporting telemetry over OTLP | `uv sync --extra otel` |

## The web interface

The interface is a local, single-operator Next.js application. It holds no engine logic: every number comes from
the engine's API ([ADR 0043](../adr/0043-the-interface-is-a-single-operator-local-application.md)).

```bash
uv sync --extra web
uv run python -m jahan.web          # the engine API on http://127.0.0.1:8000

cd frontend
npm install
npm run dev                         # the interface on http://localhost:3000
```

The interface reaches the engine at `JAHAN_WEB_URL` (default `http://127.0.0.1:8000`), and that is the only thing
it is configured with.

## Building these docs

```bash
uv run --only-group docs mkdocs serve -a 127.0.0.1:8001     # live preview, beside the engine API on 8000
```

## Next

- [Your first study](first-study.md), with no API keys and no downloads
- [Running the application end to end](real-models.md), engine + LiteLLM + local TwHIN-BERT + corpus + web interface
- [How a study works](../concepts/index.md)
