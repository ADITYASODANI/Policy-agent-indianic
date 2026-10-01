import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import chat, policy
from app.config import get_settings
from app.services import history
from app.services.vectorstore import chunk_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("policy_assistant")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY is not set; /api/chat will fail.")
    # Warm up the embedding model and vector store once, not per request.
    count = chunk_count()
    if count == 0:
        logger.warning("No policy chunks in ChromaDB. Run `python -m scripts.ingest` or POST /api/policy/ingest.")
    else:
        logger.info("Loaded %d policy chunks from ChromaDB.", count)
    history.ensure_indexes()
    yield


app = FastAPI(title="IndiaNIC Policy Assistant API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(policy.router)


@app.exception_handler(Exception)
async def unhandled_error(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong. Please try again."})


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
