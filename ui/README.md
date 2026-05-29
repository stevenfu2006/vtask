# VTASK Web UI

Dark-themed task explorer, model evaluator, and dataset exporter for the VTASK library.

## Stack

- **Backend**: FastAPI + uvicorn, imports VTASK as a library
- **Frontend**: React 18 + Vite 5 + Tailwind CSS 3, no component libraries

## Run

**Backend** (terminal 1):
```bash
cd ui/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend** (terminal 2):
```bash
cd ui/frontend
npm install
npm run dev
```

Then open http://localhost:5173

## Views

- **Explorer** — Generate individual tasks, submit answers, toggle multiple-choice mode
- **Eval** — Run Claude against a domain via Anthropic API, live SSE result stream + SVG accuracy chart
- **Export** — Generate JSONL datasets and download to disk

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/domains` | List all domains with descriptions |
| POST | `/api/generate` | Generate one task (answer not returned) |
| POST | `/api/verify` | Submit answer, reveal correct answer |
| POST | `/api/eval` | SSE stream of model eval results |

The answer is never returned from `/api/generate`. It's stored server-side keyed by `task_id` and only revealed after a `/api/verify` call.
