#!/usr/bin/env python3
"""
rank.py — Redrob Intelligent Candidate Discovery & Ranking
==========================================================
Team: Team_unique_2026

Single-command entrypoint that turns the 100K candidate pool into a top-100
ranked CSV, offline, CPU-only, well within the 5-minute budget.

    python rank.py --candidates ./candidates.jsonl --out ./Team_unique_2026.csv

Pipeline (two stages: cheap recall, then careful rank):

    1. STREAM the JSONL once. For every candidate:
         - extract interpretable features         (features.extract)
         - flag structural honeypots              (features.detect_honeypot)
         - keep a compact record + profile text    (no full-JSON retained)
       This bounds memory: we never hold 100K raw records at once.

    2. Fit a TF-IDF semantic model over the corpus and score every candidate
       (scoring.score_candidate) = interpretable additive fit
         * behavioural availability multiplier
         * trap gates
         + small semantic tilt.
       Honeypots are forced to the bottom (score 0) so they cannot enter the
       top 100 — directly protecting the >10% honeypot-rate disqualifier.

    3. Rank, break ties by candidate_id ascending (per the spec), enforce a
       strictly non-increasing score column, and write the CSV.

Everything is deterministic: same input file -> byte-identical output.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

# allow running as `python rank.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from features import detect_honeypot, extract          # noqa: E402
from jd_profile import JDProfile                        # noqa: E402
from scoring import build_reasoning, score_candidate    # noqa: E402
from semantic import TfidfSemantic, try_embedding_semantic  # noqa: E402


def _iter_candidates(path: Path):
    """Stream a .jsonl (or .jsonl.gz) file one record at a time."""
    if path.suffix == ".gz":
        import gzip
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = lambda: open(path, "r", encoding="utf-8")
    with opener() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="Redrob candidate ranker")
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl(.gz)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--top", type=int, default=100, help="How many to output")
    ap.add_argument("--asof", default=None, help="Reference date YYYY-MM-DD (default: dataset max last_active)")
    ap.add_argument("--embeddings", default=None,
                    help="Optional local sentence-transformers model dir for dense semantic re-rank")
    args = ap.parse_args()

    t0 = time.time()
    jd = JDProfile()
    cpath = Path(args.candidates)

    # A stable "today" so scoring is reproducible regardless of wall-clock. We
    # anchor to a fixed reference just after the dataset's activity window.
    today = datetime.strptime(args.asof, "%Y-%m-%d").date() if args.asof else date(2026, 6, 1)

    records = []      # compact feature dicts
    texts = []        # profile text for the semantic layer
    honey = 0
    total = 0

    print(f"[1/3] Streaming + featurising {cpath} ...", file=sys.stderr)
    for cand in _iter_candidates(cpath):
        total += 1
        f = extract(cand, jd, today)
        is_hp, hp_reasons = detect_honeypot(cand, today)
        f["is_honeypot"] = is_hp
        f["hp_reasons"] = hp_reasons
        if is_hp:
            honey += 1
        texts.append(f.pop("_text"))
        records.append(f)
        if total % 20000 == 0:
            print(f"      ...{total} processed", file=sys.stderr)
    print(f"      {total} candidates, {honey} structural honeypots flagged "
          f"({time.time()-t0:.1f}s)", file=sys.stderr)

    # ---- semantic layer ---------------------------------------------------- #
    print("[2/3] Semantic similarity (TF-IDF) + scoring ...", file=sys.stderr)
    sem = TfidfSemantic(jd.jd_query_text())
    sem.fit(texts)
    sims = sem.similarity(texts)

    if args.embeddings:
        dense = try_embedding_semantic(jd.jd_query_text(), texts, args.embeddings)
        if dense is not None:
            sims = 0.5 * sims + 0.5 * dense
            print("      dense embedding re-rank blended in.", file=sys.stderr)
        else:
            print("      (embedding model unavailable; TF-IDF only)", file=sys.stderr)

    del texts  # free memory before scoring

    for i, f in enumerate(records):
        if f["is_honeypot"]:
            f["score"] = 0.0
            f["comps"] = {}
            f["flags"] = ["honeypot: " + (f["hp_reasons"][0] if f["hp_reasons"] else "impossible profile")]
            continue
        s, comps, flags = score_candidate(f, jd, semantic_sim=float(sims[i]))
        f["score"] = s
        f["comps"] = comps
        f["flags"] = flags

    # ---- rank + tie-break -------------------------------------------------- #
    print("[3/3] Ranking + writing CSV ...", file=sys.stderr)
    # Round to the displayed precision FIRST, then sort by (display score desc,
    # candidate_id asc). This guarantees the spec's tie-break rule: whenever two
    # displayed scores are equal, candidate_id is ascending.
    scored = [(round(max(r["score"], 0.0), 4), r["candidate_id"], r) for r in records]
    scored.sort(key=lambda t: (-t[0], t[1]))
    top = scored[: args.top]

    out_rows = []
    for rank, (disp, cid, r) in enumerate(top, start=1):
        reasoning = build_reasoning(r, r["comps"], r["flags"])
        out_rows.append((cid, rank, f"{disp:.4f}", reasoning))

    outp = Path(args.out)
    with open(outp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in out_rows:
            w.writerow(row)

    print(f"Done. Wrote {len(out_rows)} rows to {outp} in {time.time()-t0:.1f}s.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
