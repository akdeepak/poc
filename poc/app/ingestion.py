"""Ingestion pipeline: Word document -> text -> chunks -> embeddings -> vector index."""
import io
import uuid

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from app import vector_store
from app.config import CHUNK_OVERLAP, CHUNK_SIZE


def extract_text(content: bytes) -> str:
    """Step 1 - Load: read paragraphs and tables in the order they appear in the document."""
    word_doc = docx.Document(io.BytesIO(content))
    lines: list[str] = []
    for block in word_doc.iter_inner_content():
        if isinstance(block, Paragraph) and block.text.strip():
            lines.append(block.text.strip())
        elif isinstance(block, Table):
            for row in block.rows:
                lines.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(lines)


def ingest_word_document(file_name: str, content: bytes) -> dict:
    text = extract_text(content)
    if not text.strip():
        raise ValueError("The document does not contain any readable text")

    document_id = str(uuid.uuid4())
    document = Document(
        id_=document_id,
        text=text,
        metadata={"document_id": document_id, "file_name": file_name},
        # Keep the id out of the text that gets embedded / sent to the LLM
        excluded_embed_metadata_keys=["document_id"],
        excluded_llm_metadata_keys=["document_id"],
    )

    # Step 2 - Chunk: split into overlapping, sentence-aware chunks
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents([document])

    # Step 3 - Embed + Store
    vector_store.add_nodes(nodes)

    return {
        "document_id": document_id,
        "file_name": file_name,
        "characters": len(text),
        "chunks": len(nodes),
    }
