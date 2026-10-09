import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

load_dotenv()

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")

# LlamaIndex chunk sizes are measured in tokens, not characters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "4"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/index"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024


def configure_models() -> None:
    """Register the embedding model and LLM globally so every LlamaIndex component uses them."""
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.llm = HuggingFaceLLM(
        model_name=LLM_MODEL_NAME,
        tokenizer_name=LLM_MODEL_NAME,
        context_window=4096,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        generate_kwargs={"do_sample": False},
        device_map=DEVICE_MAP,
        # Makes LlamaIndex format prompts with the model's own chat template
        is_chat_model=True,
    )
