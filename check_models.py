"""Thử từng mô hình Gemini 1 lần để xem mô hình nào còn hạn mức miễn phí.
Chạy: python check_models.py   (mỗi mô hình tốn 1 request)"""
import sys
from config import get_llm_client

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.5-flash",
]

client = get_llm_client()
for name in CANDIDATES:
    try:
        r = client.chat.completions.create(
            model=name, messages=[{"role": "user", "content": "hi"}], max_tokens=256)
        print(f"OK   {name}: {(r.choices[0].message.content or '').strip()[:30]!r}")
    except Exception as e:
        msg = str(e).replace("\n", " ")
        print(f"LOI  {name}: {getattr(e, 'status_code', '')} {msg[:120]}")