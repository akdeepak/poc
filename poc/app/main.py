from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import MAX_UPLOAD_BYTES, TOP_K, configure_models
from app.ingestion import ingest_word_document
from app.search import search


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_models()
    yield


app = FastAPI(title="RAG POC - Ingestion & Search", lifespan=lifespan)


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(TOP_K, ge=1, le=20)
    document_id: str | None = Field(None, description="Limit the search to one ingested document")


@app.post("/ingest", tags=["Ingestion"])
def ingest(file: UploadFile = File(..., description="Word document (.docx)")):
    file_name = Path(file.filename or "").name
    if not file_name.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported")

    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is too large")

    try:
        return ingest_word_document(file_name, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/search", tags=["Search"])
def search_documents(request: SearchRequest):
    try:
        return search(request.question, request.top_k, request.document_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
