# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Phùng Đình Triển (2A202602837)  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8235 | 0.9042 | +0.0806 |
| Answer Relevancy | 0.6762 | 0.7557 | +0.0795 |
| Context Precision | 0.8421 | 0.8333 | -0.0088 |
| Context Recall | 0.8250 | 0.8333 | +0.0083 |

> Ghi chú: baseline chấm bằng `gemini-3.5-flash-lite`, production chấm bằng `gemini-3.1-flash-lite` (đổi do hết hạn mức ngày). Một số job RAGAS bị lỗi 503 nên thành NaN (production: answer_relevancy 2/20, context_precision 3/20, context_recall 1/20); NaN được bỏ qua khi tính trung bình. Phần "Got" bên dưới lấy từ lần chạy lại 5 câu hỏi bằng `get_answers.py`, có thể khác đôi chút so với lần chạy đã được RAGAS chấm.

## Bottom-5 Failures

### #1
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** "Một nhân viên có 9 năm thâm niên được hưởng 18 ngày phép năm. Thông tin về khoảng lương không được tìm thấy."
- **Worst metric:** answer_relevancy = 0.0
- **Error Tree:** Output sai → Context đúng? Chỉ một phần: chunk đầu tiên chứa đúng ví dụ "9 năm thâm niên được 18 ngày phép (15 + 3)", nhưng không có chunk về khoảng lương Senior → Query OK? Chưa, câu hỏi gồm hai vế (ngày phép và lương) nhưng chỉ dùng một truy vấn → Fix ở bước query/retrieval
- **Root cause:** Câu hỏi đa bước (multi-hop). Vế ngày phép đúng, vế lương không truy xuất được vào top-3 nên LLM trả lời "không tìm thấy". Câu trả lời né tránh một vế nên answer_relevancy bị chấm 0.
- **Suggested fix:** Query decomposition (tách thành "ngày phép của nhân viên 9 năm thâm niên" và "khoảng lương Senior"), gộp context của hai truy vấn rồi mới sinh câu trả lời; có thể tăng `RERANK_TOP_K` từ 3 lên 4-5.

### #2
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Cần xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất, và đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Vế CNTT đúng (cần xác nhận cấu hình kỹ thuật). Vế phê duyệt: "Văn bản không quy định cụ thể người phê duyệt cho đơn hàng 30 triệu đồng (chỉ nêu thẩm quyền cho đơn hàng dưới 5 triệu đồng)." Không nhắc đến yêu cầu 3 báo giá.
- **Worst metric:** answer_relevancy = 0.0
- **Error Tree:** Output sai → Context đúng? Chỉ một phần: chunk đầu chứa quy định xác nhận của phòng CNTT, nhưng chunk bảng thẩm quyền phê duyệt theo mức giá (5-50 triệu) không lọt vào top-3 → Query OK? Chưa, câu hỏi hai vế → Fix ở bước query/retrieval
- **Root cause:** Cùng dạng multi-hop như #1: thông tin nằm ở hai chunk khác nhau (quy trình mua sắm theo mức giá và quy định thiết bị CNTT), top-3 chỉ phủ được một.
- **Suggested fix:** Query decomposition, tăng số chunk đưa vào LLM, và yêu cầu trong prompt trả lời đủ tất cả các vế của câu hỏi.

### #3
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Theo chính sách v2024 hiện hành: từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm.
- **Got:** Trả lời cả hai con số: "từ 5 năm trở lên" (quy định cũ) và "từ 3 năm trở lên" (phiên bản 2024), không kết luận phiên bản nào áp dụng.
- **Worst metric:** answer_relevancy = 0.0
- **Error Tree:** Output sai → Context đúng? Có cả hai phiên bản, nhưng chunk đứng đầu là bản cũ (v2023, "5 năm trở lên... 1 ngày phép cho mỗi 5 năm") → Query OK? Có → Fix ở bước retrieval/metadata và prompt
- **Root cause:** Kho tài liệu chứa hai phiên bản chính sách mâu thuẫn nhau (v2023 và v2024) mà chunk không có metadata phiên bản/ngày hiệu lực; reranker xếp bản cũ lên đầu. LLM đưa cả hai đáp án nên không đưa ra câu trả lời dứt khoát. (Đã sửa so với giả thuyết ban đầu là điểm NaN.)
- **Suggested fix:** Trích xuất metadata `version`/`effective_date` ở bước enrichment (M5) và lọc hoặc ưu tiên bản mới nhất khi truy xuất; thêm vào prompt "nếu có nhiều phiên bản thì dùng bản mới nhất và nêu rõ".

### #4
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** "Số tiền bị phạt là 2% trên số tiền chưa hoàn ứng (15.000.000 VNĐ) cho 1 tháng. Cụ thể: 15.000.000 x 2% = 300.000 VNĐ."
- **Worst metric:** faithfulness = 0.33
- **Error Tree:** Output sai → Context đúng? Có chunk "Phạt quá hạn" ở vị trí đầu, nhưng câu trả lời không xác định thời hạn 15 ngày hay số ngày quá hạn → Query OK? Có → Fix ở bước sinh câu trả lời (prompt)
- **Root cause:** Câu hỏi cần kết hợp hai quy định (hạn 15 ngày và phí 2%/tháng) và tính theo số ngày quá hạn. LLM bỏ qua việc tính theo ngày và tự áp dụng phí cả một tháng, nên phần kết luận không được context hỗ trợ (faithfulness thấp).
- **Suggested fix:** Siết system prompt (suy luận từng bước: xác định hạn, số ngày quá hạn, rồi mới tính phí; chỉ dùng thông tin trong context), `temperature=0`, yêu cầu trích dẫn đoạn nguồn.

### #5
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** "Tổng Giám đốc (CEO)" (đúng).
- **Worst metric:** context_precision = 0.0
- **Error Tree:** Output đúng → Context đúng? Câu trả lời đúng nhưng chunk đứng đầu là quy định xác nhận cấu hình của phòng CNTT, không phải bảng thẩm quyền phê duyệt, nên các chunk liên quan bị xếp sau → Query OK? Có → Fix ở bước rerank/metadata
- **Root cause:** Các chunk về mua sắm thiết bị CNTT có từ khóa và câu ngữ cảnh enrichment rất giống nhau nên chunk xác nhận cấu hình bị xếp trên chunk bảng phê duyệt. Điểm 0.0 phản ánh thứ hạng xấu chứ không phải câu trả lời sai, và cũng có thể do job chấm lỗi 503.
- **Suggested fix:** Tăng độ phân biệt giữa các chunk bằng metadata theo loại quy định (xác nhận kỹ thuật so với thẩm quyền phê duyệt), thử reranker với danh sách ứng viên lớn hơn.

## Case Study (cho presentation)

**Question chọn phân tích:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?

**Error Tree walkthrough:**
1. Output đúng? → Không dứt khoát: câu trả lời nêu cả "5 năm" (v2023) và "3 năm" (v2024) mà không kết luận, trong khi đáp án đúng là 3 năm theo bản hiện hành.
2. Context đúng? → Có cả hai phiên bản, nhưng chunk đứng đầu là bản cũ (5 năm), vì chunk không mang thông tin phiên bản hay ngày hiệu lực.
3. Query rewrite OK? → Có, câu hỏi đơn giản và rõ ràng.
4. Fix ở bước: Enrichment (M5) thêm metadata `version`/`effective_date`, kết hợp lọc theo metadata ở M2 và prompt ưu tiên phiên bản mới nhất.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm metadata phiên bản/ngày hiệu lực và lọc hoặc ưu tiên bản mới nhất khi truy xuất.
- Thêm query decomposition cho câu hỏi đa bước (#1, #2) và gộp context từ các truy vấn con.
- Siết prompt sinh câu trả lời (`temperature=0`, suy luận từng bước cho câu tính toán, trả lời đủ các vế, nêu rõ khi có nhiều phiên bản).
- Thêm bước tự chạy lại các ô NaN của RAGAS để điểm đánh giá ổn định, và OCR hai file PDF scan (`BCTC.pdf`, Nghị định 13/2023) đang bị bỏ qua.