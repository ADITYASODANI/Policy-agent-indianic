# IndiaNIC Policy Assistant

A RAG chatbot that answers employee questions about the **IndiaNIC Infotech Limited — Code of Conduct & Ethics (Version 2, effective 01 January 2026)**. The policy document is the only source of truth: every answer cites the policy section it comes from, and anything the policy doesn't cover is referred to HR.

| Layer        | Tech                                                      |
|--------------|-----------------------------------------------------------|
| Frontend     | React 18 + Vite                                           |
| Backend      | Python 3.12, FastAPI                                      |
| RAG          | LangChain                                                 |
| LLM          | Groq API (`openai/gpt-oss-120b` by default)               |
| Embeddings   | HuggingFace `BAAI/bge-small-en-v1.5` (runs locally)       |
| Vector store | ChromaDB (persisted to `backend/chroma_db/`)              |
| Chat history | MongoDB                                                   |
| Documents    | PyPDF / python-docx (plain `.txt` is also supported)      |

## How it works

```
Ingestion (run once, or when the policy changes)
  policy PDF/DOCX/TXT → text extraction → split by numbered section → chunk
  → HuggingFace embeddings → ChromaDB (with section metadata)

Chat (per question)
  question → LLM query rewrite (spelling / Hinglish / follow-ups → English)
  → similarity search in ChromaDB (rewritten + original query)
  → top policy chunks → LangChain prompt → Groq (JSON output)
  → references validated against retrieved chunks → React UI
  → turn saved to MongoDB
```

Guardrails:
- The prompt restricts the model to the retrieved excerpts. It returns structured JSON (`kind`, `answer`, `section_numbers`, `recommended_action`).
- If the model reports "not found", the fixed message is returned: *"This information is not specified in the available company policy. Please contact HR for clarification."*
- Cited sections are kept only if they were actually retrieved, so the model cannot invent a reference. Every policy answer carries at least one section reference.
- Whistleblower answers (Section 13) always include https://nazar.indianic.biz/.
- Requests to reveal the prompt or implementation details are refused.
- Embeddings are created at ingestion time only. Each question embeds just the query.

Each chunk in ChromaDB carries this metadata:
```json
{
  "policy_name": "Code of Conduct & Ethics",
  "version": "Version 2",
  "effective_date": "01 January 2026",
  "section_number": "4",
  "section_title": "Employee Health, Safety & Workplace Security",
  "chunk_index": 0,
  "source": "code_of_conduct_v2.txt"
}
```

Each turn in MongoDB (`policy_assistant.chat_history`) looks like this. No names, emails, IPs or other employee data are stored, and `session_id` is a random browser-generated id.
```json
{
  "question": "Can I smoke inside the office?",
  "answer": "No. Smoking is strictly prohibited on Company premises.",
  "policy_section": "Section 4",
  "policy_references": [{"section_number": "4", "section_title": "Employee Health, Safety & Workplace Security"}],
  "recommended_action": null,
  "found_in_policy": true,
  "session_id": "…",
  "timestamp": "2026-10-01T10:00:00Z"
}
```

## Project structure

```
backend/
  app/
    main.py              FastAPI app, CORS, startup warm-up
    config.py            Settings loaded from .env
    schemas.py           Request/response models
    api/chat.py          /api/chat, /api/chat/history
    api/policy.py        /api/policy/ingest, /api/policy/status
    services/
      ingestion.py       load → section split → chunk → embed → Chroma
      vectorstore.py     shared embedding model + Chroma client
      rag.py             query rewrite, retrieval, Groq call, validation
      prompts.py         LangChain prompt templates
      history.py         MongoDB persistence
  scripts/ingest.py      CLI ingestion
  data/policy/           the policy document (knowledge base)
  requirements.txt
frontend/
  src/App.jsx, src/components/*, src/api.js, src/styles.css
  package.json, vite.config.js
```

## Prerequisites

- Python **3.10–3.13** (3.12 recommended. 3.14 is not yet supported by PyTorch/ChromaDB.)
- Node.js 18+
- MongoDB running locally, or a MongoDB Atlas URI. For a quick local instance:
  ```bash
  docker run -d --name policy-mongo -p 27017:27017 mongo:7
  ```
- A Groq API key from https://console.groq.com/keys

## Quick start (after setup)

```bash
./start.sh
```

This starts MongoDB (the `policy-mongo` Docker container, if nothing is already listening on :27017). It ingests the policy on first run, then starts the API (:8000) and UI (:5173). Ctrl+C stops the API and UI.

## Setup

### 1. Backend

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then set GROQ_API_KEY and MONGODB_URI
```

### 2. Ingest the policy (one-time)

```bash
# from backend/ with the venv active
python -m scripts.ingest                          # rebuild from every file in data/policy/
python -m scripts.ingest /path/to/policy.pdf      # add / refresh one file (PDF, DOCX, TXT or MD)
```

The first run downloads the embedding model (~130 MB) from HuggingFace. Re-run ingestion whenever a policy changes. Without a file it rebuilds the vectors from every policy in `data/policy/`; with a file it adds that policy, or replaces the earlier version of the same file name, and leaves the others alone.

### 3. Start the API

```bash
uvicorn app.main:app --reload --reload-dir app --port 8000
```

Interactive docs: http://localhost:8000/docs

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. In development, Vite proxies `/api` to `http://localhost:8000`. To point the frontend at a different API, set `VITE_API_BASE_URL` in `frontend/.env`.

## API

| Method | Path                    | Description |
|--------|-------------------------|-------------|
| POST   | `/api/chat`             | `{"question": "...", "session_id": "optional"}` returns answer, `policy_references[]`, `recommended_action`, `found_in_policy` |
| GET    | `/api/chat/history`     | `?session_id=...&limit=50` returns stored turns (all sessions if `session_id` is omitted) |
| DELETE | `/api/chat/history`     | `?session_id=...` clears a conversation (used by "Clear chat") |
| POST   | `/api/policy/ingest`    | Add a policy: multipart `file` (PDF/DOCX/TXT/MD) adds it or replaces the same file name, leaving other policies untouched. Without a file, everything in the policy folder is re-indexed. |
| GET    | `/api/policy/status`    | Whether the policy is ingested, and the chunk count |
| GET    | `/api/health`           | Liveness check |

Example:
```bash
curl -s localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"question":"Can I smoke inside the office?"}'
```

If MongoDB is down, chat keeps working and only history is unavailable (a 503 from the history endpoints). If the policy hasn't been ingested, `/api/chat` returns 503 with instructions.

## Configuration (`backend/.env`)

| Variable             | Default                                | Notes |
|----------------------|----------------------------------------|-------|
| `GROQ_API_KEY`       | —                                      | Required |
| `MONGODB_URI`        | `mongodb://localhost:27017`            | Required |
| `MONGODB_DB`         | `policy_assistant`                     | |
| `GROQ_MODEL`         | `openai/gpt-oss-120b`                  | Answer model. Must be available on your Groq account. |
| `GROQ_REWRITE_MODEL` | `openai/gpt-oss-20b`                   | Fast model for query normalisation |
| `EMBEDDING_MODEL`    | `BAAI/bge-small-en-v1.5`               | Re-run ingestion after changing it |
| `RETRIEVAL_K`        | `7`                                    | Chunks sent to the LLM |
| `POLICY_DIR`         | `data/policy`                          | Folder whose PDF/DOCX/TXT/MD files are all ingested; relative to `backend/` when running from there |

## Updating the policy

Put each policy as its own file in `backend/data/policy/`, then run `python -m scripts.ingest` or call `POST /api/policy/ingest`. Restart the API afterwards so it picks up the rebuilt vectors.

Sections are detected from numbered headings such as `4. EMPLOYEE HEALTH, SAFETY & WORKPLACE SECURITY`, or from markdown headings (`#### Maternity Leave`) for documents exported as markdown, where the first heading is taken as the policy name. Index and revision-history blocks are skipped. The Code of Conduct's version and effective date come from the settings defaults; other policies have none unless the text states them.
