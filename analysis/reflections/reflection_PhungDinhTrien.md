# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Phùng Đình Triển (2A202602837)  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code bạn vừa viết trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Hierarchical chunking (Parent-Child) | M1 | `chunk_hierarchical()` | "Pipeline production tạo 104 child chunk (256 ký tự) từ 26 tài liệu, so với 57 chunk của baseline cắt theo đoạn; chunk nhỏ giúp tìm kiếm chính xác hơn và vẫn giữ `parent_id` để lấy lại ngữ cảnh." |
| Semantic / Structure-aware chunking | M1 | `chunk_semantic()`, `chunk_structure_aware()` | "Cắt theo độ tương đồng ngữ nghĩa (ngưỡng 0.85) và theo tiêu đề Markdown để không cắt đứt ý; pipeline chính dùng hierarchical." |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | "RRF cộng điểm 1/(k+rank) từ BM25 (khớp từ khóa chính xác như số tiền, tên chính sách) và Dense (khớp ngữ nghĩa), không cần chuẩn hóa thang điểm hai nguồn." |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | "Lọc top-20 ứng viên xuống top-3 bằng bge-reranker-v2-m3; tuy vậy context_precision production (0.8333) thấp hơn baseline (0.8421) một chút, cho thấy reranker chưa phân biệt tốt các chunk có từ khóa rất giống nhau (ví dụ các mốc hạn mức phê duyệt)." |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | "Production đạt faithfulness 0.9042, answer_relevancy 0.7557, context_precision 0.8333, context_recall 0.8333; cả 4 chỉ số ≥ 0.75. Một số ô bị NaN do lỗi 503 và được bỏ qua khi tính trung bình." |
| Contextual embeddings / Enrichment | M5 | `_enrich_single_call()` | "Một lệnh gọi LLM cho mỗi chunk để lấy tóm tắt, câu hỏi giả định, câu ngữ cảnh và metadata; câu ngữ cảnh được gắn vào đầu chunk trước khi embed. Kết quả được cache vào `reports/enrich_cache.json` để chạy lại không tốn quota." |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - `Error code: 429 ... Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 15, model: gemini-3.5-flash-lite`
  - `Error code: 429 ... GenerateRequestsPerDayPerProjectPerModel-FreeTier ... limit: 500` (hết hạn mức ngày)
  - `Exception raised in Job[18]: InternalServerError(Error code: 503 ... This model is currently experiencing high demand)`
  - `The window terminated unexpectedly (reason: 'oom', code: '-536870904')` (VS Code bị tràn RAM)
- **Nguyên nhân gốc rễ & Cách debug:**
  - Gemini free tier giới hạn 15 request/phút nhưng RAGAS chạy song song và client OpenAI tự retry ngầm. Cách xử lý: thêm `_throttle()` giãn cách 4.5 giây trong `config.py`, đặt `max_retries=0` cho client để mọi lần thử lại đi qua hàm có giãn cách, dùng `InMemoryRateLimiter` và giảm `max_workers` xuống 2 trong `m4_eval.py`.
  - Hết hạn mức 500 request/ngày: xem trang Rate Limit trong Google AI Studio, thấy `gemini-3.1-flash-lite` còn quota nên đổi `LLM_MODEL` trong `.env`; thêm cache cho M5 và cho `main.py` bỏ qua baseline nếu đã có report (`--force-baseline` để chạy lại).
  - Điểm NaN bị đổi thành 0.0 làm điểm trung bình thấp oan: sửa `_f()` để giữ NaN, `_avg()` bỏ qua NaN và in cảnh báo số câu bị NaN.
  - Tràn RAM do chạy đồng thời VS Code, trình duyệt và hai model lớn (bge-m3, reranker): đóng bớt ứng dụng và chạy `main.py` trong PowerShell riêng.
- **Kiến thức còn thiếu & Cách khắc phục:**
  - Chưa nắm rõ cách RAGAS xử lý job lỗi và cách `answer_relevancy` cho điểm 0 với câu trả lời né tránh; khắc phục bằng cách đọc log (`Exception raised in Job`, `No statements were generated`) và đối chiếu với báo cáo từng câu.
  - Chưa quen quản lý quota API và bảo mật khóa API (khóa nằm trong `.env`, phải thu hồi khi bị lộ và đảm bảo `.env` có trong `.gitignore`).

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

Dựa trên những kỹ thuật đã học và thực hành, lập kế hoạch cụ thể áp dụng vào project của bạn:

### Project: Chatbot hỏi đáp chính sách nội bộ công ty (nhân sự, tài chính, CNTT)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Naive RAG: cắt văn bản theo đoạn (`\n\n`) thành 57 chunk, embedding bằng bge-m3 lưu trong Qdrant, tìm kiếm chỉ bằng vector (dense), không rerank, không làm giàu chunk; LLM trả lời từ top-k đoạn tìm được.
- **Vấn đề / Bottlenecks đang gặp:** Câu hỏi nhiều vế (ví dụ ngày phép và khoảng lương) bị trả lời thiếu hoặc "không tìm thấy"; kho tài liệu có nhiều phiên bản chính sách (v2023 và v2024) nên chunk bản cũ có thể xếp đầu và gây mâu thuẫn; câu hỏi tính toán (phạt tạm ứng, hoàn trả học phí) làm LLM suy diễn ngoài context; các chunk có từ khóa giống nhau (hạn mức phê duyệt) bị xếp lẫn nên context precision chưa cao; hai tài liệu PDF scan chưa đọc được do cần OCR.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Hierarchical (Parent-Child) cho tài liệu dài, Structure-Aware cho tài liệu có tiêu đề rõ; lý do: child chunk nhỏ giúp tìm kiếm chính xác hơn và parent giữ đủ ngữ cảnh khi đưa vào LLM.
2. **Search retrieval:** Hybrid (BM25 tiếng Việt + Dense) kết hợp RRF; lý do: BM25 bắt tốt số liệu và từ khóa chính xác, Dense bắt tốt ngữ nghĩa.
3. **Reranking:** Dùng Cross-Encoder `BAAI/bge-reranker-v2-m3` lọc top-20 xuống top-3 đến top-5; thử tăng k đối với câu hỏi đa chủ đề.
4. **Evaluation:** RAGAS 4 metrics trên bộ câu hỏi riêng, dùng cùng một model giám khảo cho mọi lần so sánh, bổ sung tự chạy lại các ô NaN, theo dõi bottom-5 qua Diagnostic Tree.
5. **Enrichment:** Contextual prepend và metadata tự động bằng một lệnh gọi LLM mỗi chunk, có cache; thêm query decomposition cho câu hỏi đa bước.

#### 3. Timeline triển khai
- **Tuần 1:** Dựng bộ câu hỏi đánh giá và baseline RAGAS cho project; triển khai Hierarchical chunking và Hybrid Search (BM25 + Dense + RRF).
- **Tuần 2:** Thêm Cross-Encoder reranking, enrichment có cache và query decomposition; chạy lại RAGAS, phân tích bottom-5 và tối ưu prompt (`temperature=0`, yêu cầu trích nguồn).