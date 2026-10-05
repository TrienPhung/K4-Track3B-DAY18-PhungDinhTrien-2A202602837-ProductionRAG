import json
from src.pipeline import build_pipeline, run_query

QS = [
    "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?",
    "Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?",
    "Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?",
    "Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?",
    "Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?",
]
gt = {x["question"]: x["ground_truth"] for x in json.load(open("test_set.json", encoding="utf-8"))}
search, reranker = build_pipeline()
for q in QS:
    ans, ctx = run_query(q, search, reranker)
    print("\nQ:", q, "\nEXPECTED:", gt.get(q), "\nGOT:", ans, "\nCONTEXT[0]:", ctx[0][:200])