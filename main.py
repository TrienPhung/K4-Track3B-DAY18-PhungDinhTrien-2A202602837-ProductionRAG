"""
Lab 18: Production RAG Pipeline — Main Entry Point
===================================================
Chạy toàn bộ pipeline: naive baseline → production → so sánh → report.

Usage:
    python main.py                    # bỏ qua baseline nếu đã có report hợp lệ
    python main.py --force-baseline   # ép chạy lại baseline
"""

import argparse
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-baseline", action="store_true",
                        help="Chạy lại baseline dù đã có report")
    args = parser.parse_args()

    print("=" * 60)
    print("LAB 18: PRODUCTION RAG PIPELINE")
    print("=" * 60)
    start = time.time()

    os.makedirs("reports", exist_ok=True)
    naive_path = "reports/naive_baseline_report.json"
    prod_path = "reports/ragas_report.json"

    # Step 1: Basic Baseline (bỏ qua nếu đã có kết quả hợp lệ)
    baseline_ok = False
    if os.path.exists(naive_path) and not args.force_baseline:
        with open(naive_path, encoding="utf-8") as f:
            agg = json.load(f).get("aggregate", {})
        baseline_ok = any(isinstance(v, (int, float)) and v > 0 for v in agg.values())

    print("\n📌 STEP 1: Running Basic RAG Baseline...")
    print("-" * 40)
    if baseline_ok:
        print("  ✓ Đã có reports/naive_baseline_report.json, bỏ qua "
              "(dùng --force-baseline để chạy lại).")
    else:
        from naive_baseline import main as run_baseline
        run_baseline()
        if os.path.exists("naive_baseline_report.json"):
            os.replace("naive_baseline_report.json", naive_path)
        print("  ⏳ Nghỉ 60s để quota Gemini hồi lại...")
        time.sleep(60)

    # Step 2: Production Pipeline
    print("\n📌 STEP 2: Running Production Pipeline...")
    print("-" * 40)
    from src.pipeline import build_pipeline, evaluate_pipeline
    search, reranker = build_pipeline()
    prod_results = evaluate_pipeline(search, reranker)

    # Ensure reports are located in reports/
    for f in ["ragas_report.json", "naive_baseline_report.json"]:
        if os.path.exists(f):
            os.replace(f, f"reports/{f}")

    if not any(prod_results.get(m, 0) > 0 for m in METRICS):
        print("\n❌ Toàn bộ điểm production = 0: RAGAS đã lỗi (xem dòng "
              "'RAGAS evaluation failed' phía trên). Đừng dùng report này để nộp.")

    # Step 3: Comparison
    print("\n📌 STEP 3: Comparison")
    print("-" * 40)

    if os.path.exists(naive_path) and os.path.exists(prod_path):
        with open(naive_path, encoding="utf-8") as f:
            naive = json.load(f)
        with open(prod_path, encoding="utf-8") as f:
            prod = json.load(f)

        print(f"\n{'Metric':<25} {'Basic':>8} {'Production':>12} {'Δ':>8}")
        print("-" * 55)
        for m in METRICS:
            n = naive.get("aggregate", {}).get(m, 0)
            p = prod.get("aggregate", {}).get(m, 0)
            d = p - n
            status = "✓" if p >= 0.75 else " "
            print(f"{status} {m:<23} {n:>8.4f} {p:>12.4f} {d:>+8.4f}")

    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    print("\n📋 Next steps:")
    print("  1. Điền analysis/failure_analysis.md")
    print("  2. Viết analysis/reflections/reflection_[HọTên].md")
    print("  3. Chạy: python check_lab.py")


if __name__ == "__main__":
    main()