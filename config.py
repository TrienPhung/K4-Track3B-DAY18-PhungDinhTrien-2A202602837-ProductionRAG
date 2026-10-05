"""Shared configuration for Lab 18."""

import os
import threading
import time
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def _clean_model(name: str) -> str:
    """Chuẩn hóa tên mô hình từ .env: bỏ khoảng trắng/ngoặc kép, viết thường, bỏ tiền tố 'models/'."""
    name = name.strip().strip("\"'").strip().lower()
    return name[len("models/"):] if name.startswith("models/") else name


# --- LLM provider (ưu tiên Gemini nếu có GEMINI_API_KEY, ngược lại dùng OpenAI) ---
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
if GEMINI_API_KEY:
    LLM_PROVIDER, LLM_API_KEY, LLM_BASE_URL = "gemini", GEMINI_API_KEY, GEMINI_BASE_URL
    LLM_MODEL = _clean_model(os.getenv("LLM_MODEL", "gemini-3.5-flash-lite"))
elif OPENAI_API_KEY:
    LLM_PROVIDER, LLM_API_KEY, LLM_BASE_URL = "openai", OPENAI_API_KEY, None
    LLM_MODEL = _clean_model(os.getenv("LLM_MODEL", "gpt-4o-mini"))
else:
    LLM_PROVIDER, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL = "", "", None, ""


# --- Throttle toàn cục: giãn cách giữa các request để không vượt 15 RPM (free tier) ---
_last_call = 0.0
_lock = threading.Lock()
MIN_INTERVAL = 4.5 if LLM_PROVIDER == "gemini" else 0.0  # ~13 request/phút


def _throttle():
    global _last_call
    if MIN_INTERVAL <= 0:
        return
    with _lock:
        wait = MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()


def get_llm_client():
    """OpenAI client trỏ tới Gemini (endpoint tương thích OpenAI) hoặc OpenAI.
    max_retries=0 để mọi lần retry đều đi qua llm_chat (có throttle)."""
    from openai import OpenAI
    if LLM_BASE_URL:
        return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, max_retries=0)
    return OpenAI(api_key=LLM_API_KEY, max_retries=0)


def llm_chat(messages: list[dict], **kwargs):
    """Gọi chat completion, có throttle và tự thử lại khi bị giới hạn tốc độ (429) hoặc lỗi server."""
    client = get_llm_client()
    last = None
    for attempt in range(4):
        try:
            _throttle()
            return client.chat.completions.create(model=LLM_MODEL, messages=messages, **kwargs)
        except Exception as e:
            last = e
            code = getattr(e, "status_code", None)
            if code in (429, 500, 503) and attempt < 3:
                time.sleep(30 if code == 429 else 5 * (attempt + 1))
                continue
            raise
    raise last


# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")