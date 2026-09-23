# Running with Docker

## Build the image

```bash
docker build -t agentic-rag-chatbot .
```

## Run the container

```bash
docker run -p 8501:8501 --env-file .env agentic-rag-chatbot
```

Then open <http://localhost:8501> in your browser.

## Notes

- Copy `.env.example` to `.env` and fill in your API keys before running.
- `checkpoints.sqlite` and `sessions.json` are created inside the container
  and will reset if the container is removed. For persistence across
  container restarts, mount a volume:

```bash
  docker run -p 8501:8501 --env-file .env -v $(pwd)/data:/app/data agentic-rag-chatbot
```

  (This requires updating the SQLite path and sessions.json path in the code
  to point to `/app/data/` instead of the project root.)