import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.schemas import IngestResponse
from app.services.ingestion import SUPPORTED_EXTENSIONS, IngestionError, ingest_all, ingest_policy
from app.services.vectorstore import chunk_count

router = APIRouter(prefix="/api/policy", tags=["policy"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile | None = File(default=None)) -> IngestResponse:
    """Upload a PDF/DOCX/TXT/MD to add it (or refresh the file of the same name), leaving the other
    policies untouched. Omit the file to rebuild everything from the policy folder."""
    try:
        if file is None:
            result = await run_in_threadpool(ingest_all)
        else:
            name = Path(file.filename or "").name
            suffix = Path(name).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise HTTPException(status_code=400, detail="Upload a PDF, DOCX, TXT or MD file.")
            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / f"policy{suffix}"
                with dest.open("wb") as out:
                    shutil.copyfileobj(file.file, out)
                result = await run_in_threadpool(ingest_policy, dest, name)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(message="Policy ingested successfully.", **result)


@router.get("/status")
def status() -> dict:
    count = chunk_count()
    return {"ingested": count > 0, "chunks": count}
