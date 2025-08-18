#!/usr/bin/env python3
import argparse, concurrent.futures, random, time, statistics, sys, traceback
from typing import List, Dict, Any

# Import pipeline kamu
# Pastikan modul import sesuai struktur proyekmu
# (di app streamlit kamu: `from search import RAG_pipeline`)
from search import RAG_pipeline  # sesuaikan jika path berbeda

# Kumpulan query contoh — ganti dengan datasetmu
QUERIES = [
    "Apa saja fasilitas di Departemen Ilmu Komputer?",
    "Jelaskan KBK/Penjurusan di Ilkom UPI",
    "Visi dan Misi Pendidikan Ilmu Komputer",
    "Bagaimana proses pendaftaran mahasiswa baru?",
    "Daftar mata kuliah inti semester awal",
    "Apa itu Keluarga Mahasiswa Komputer?",
    "Informasi beasiswa untuk mahasiswa Ilkom",
    "Profil lulusan dan capaian pembelajaran",
]


def bench_one(query: str, timeout: float) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ttft = None
    total_tokens = 0
    ok = True
    err = None
    try:
        # RAG_pipeline di streamlit mengembalikan generator streaming
        stream = RAG_pipeline(query=query, chat_history=[], streaming=True)
        last_yield = t0
        for chunk in stream:
            delta = getattr(chunk, "content", None)
            if not delta:
                continue
            total_tokens += len(delta)
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            last_yield = now
            # timeout proteksi saat ngegantung di tengah stream
            if timeout and (now - t0) > timeout:
                ok = False
                err = f"timeout>{timeout}s"
                break
        t1 = time.perf_counter()
        total = t1 - t0
    except Exception as e:
        ok = False
        err = f"{type(e).__name__}: {e}"
        total = time.perf_counter() - t0
        ttft = ttft
    return {
        "ok": ok,
        "ttft": ttft if ttft is not None else float("inf"),
        "total": total,
        "tokens": total_tokens,
        "error": err,
        "query": query,
    }


def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p/100.0) * (len(s)-1)))))
    return s[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--timeout", type=float, default=90.0, help="timeout per request (detik)")
    args = ap.parse_args()

    print(f"Running bench: concurrency={args.concurrency} requests={args.requests}")

    results = []
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = []
        for i in range(args.requests):
            q = random.choice(QUERIES)
            futs.append(ex.submit(bench_one, q, args.timeout))
        for fu in concurrent.futures.as_completed(futs):
            try:
                results.append(fu.result())
            except Exception:
                traceback.print_exc()
    duration = time.perf_counter() - start

    ok = [r for r in results if r["ok"]]
    errs = [r for r in results if not r["ok"]]
    ttfts = [r["ttft"] for r in ok]
    totals = [r["total"] for r in ok]

    def fmt(x):
        return f"{x*1000:.0f} ms"

    print("\n==== Summary (Streamlit-only / RAG_pipeline) ====")
    print({
        "requests": len(results),
        "success": len(ok),
        "errors": len(errs),
        "duration_s": round(duration, 2),
        "rps": round(len(results)/duration, 2) if duration > 0 else None,
    })
    if ok:
        print("TTFT:", {"p50": fmt(percentile(ttfts,50)), "p95": fmt(percentile(ttfts,95)), "max": fmt(max(ttfts))})
        print("Total:", {"p50": fmt(percentile(totals,50)), "p95": fmt(percentile(totals,95)), "max": fmt(max(totals))})
    if errs[:5]:
        print("Sample errors:")
        for e in errs[:5]:
            print(" -", e["error"])

if __name__ == "__main__":
    main()