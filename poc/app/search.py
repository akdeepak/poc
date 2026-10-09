"""Search pipeline: question -> embed -> retrieve top-k chunks -> LLM answers from those chunks."""
from llama_index.core import PromptTemplate, get_response_synthesizer
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from app import vector_store

QA_PROMPT = PromptTemplate(
    "You are an assistant that answers questions using ONLY the context below.\n"
    "If the answer is not in the context, reply: \"I could not find this in the document.\"\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer: "
)


def search(question: str, top_k: int, document_id: str | None = None) -> dict:
    index = vector_store.get_index()
    if index is None:
        raise LookupError("No documents have been ingested yet")

    filters = None
    if document_id:
        filters = MetadataFilters(filters=[ExactMatchFilter(key="document_id", value=document_id)])

    # Step 1 - Retrieve: the question is embedded and compared with the stored chunk vectors
    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
    matches = retriever.retrieve(question)
    if not matches:
        return {"answer": "I could not find this in the document.", "sources": []}

    # Step 2 - Generate: the LLM answers using only the retrieved chunks
    synthesizer = get_response_synthesizer(text_qa_template=QA_PROMPT)
    response = synthesizer.synthesize(question, nodes=matches)

    return {
        "answer": str(response),
        "sources": [
            {
                "document_id": m.node.metadata.get("document_id"),
                "file_name": m.node.metadata.get("file_name"),
                "score": round(m.score or 0.0, 4),
                "text": m.node.get_content(),
            }
            for m in matches
        ],
    }
