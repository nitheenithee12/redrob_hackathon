"""
app.py — Redrob Ranker sandbox demo
===================================
A minimal hosted-sandbox app that satisfies the submission_spec Section 10.5
requirement: accept a small candidate sample (<=100), run the ranking system
end-to-end on CPU within the compute budget, and show the ranked output.

Deploy free on Streamlit Cloud or HuggingFace Spaces:
    streamlit run app.py

Upload a .jsonl / .json file of candidate records (schema per candidate_schema.json)
or click "Use bundled sample" to run on sample_candidates.json.
"""

import io
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from features import detect_honeypot, extract          # noqa: E402
from jd_profile import JDProfile                        # noqa: E402
from scoring import build_reasoning, score_candidate    # noqa: E402
from semantic import TfidfSemantic                      # noqa: E402

st.set_page_config(page_title="Redrob Ranker — Team_unique_2026", layout="wide")
st.title("Redrob Intelligent Candidate Ranker")
st.caption("Team_unique_2026 · CPU-only · offline · interpretable")

jd = JDProfile()
with st.expander("What the ranker looks for (JD understanding)"):
    st.markdown(
        "- **Role fit**: is the trajectory actually an AI/ML/search/recsys engineer? "
        "(a Marketing Manager with 9 AI skills is *not* a fit)\n"
        "- **Shipped work**: evidence of retrieval / ranking / recommendation / "
        "semantic-search systems built in production\n"
        "- **Retrieval stack + evaluation**: vector DBs, embeddings, NDCG/MRR/A-B testing\n"
        "- **Experience** ~5–9 yrs, **product** (not services-only) companies, "
        "**preferred cities** or willing to relocate\n"
        "- **Availability modifier**: dormant / non-responsive profiles are down-weighted\n"
        "- **Traps excluded**: keyword stuffers, honeypots (impossible profiles), "
        "title-chasers, CV/speech-without-NLP, LangChain-only"
    )


def load_records(raw: bytes):
    text = raw.decode("utf-8")
    recs = []
    text_stripped = text.lstrip()
    if text_stripped.startswith("["):
        recs = json.loads(text)
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs[:100]


col1, col2 = st.columns([2, 1])
uploaded = col1.file_uploader("Upload candidate sample (.json / .jsonl, <=100)", type=["json", "jsonl"])
use_sample = col2.button("Use bundled sample")

records = None
if uploaded is not None:
    records = load_records(uploaded.read())
elif use_sample:
    sample = Path(__file__).resolve().parent / "sample_candidates.json"
    if sample.exists():
        records = load_records(sample.read_bytes())
    else:
        st.warning("sample_candidates.json not found in repo root.")

if records:
    today = date(2026, 6, 1)
    feats, texts = [], []
    for c in records:
        f = extract(c, jd, today)
        f["is_honeypot"], f["hp_reasons"] = detect_honeypot(c, today)
        texts.append(f.pop("_text"))
        feats.append(f)

    sem = TfidfSemantic(jd.jd_query_text())
    sem.fit(texts if len(texts) > 3 else texts + [jd.jd_query_text()])
    sims = sem.similarity(texts)

    rows = []
    for i, f in enumerate(feats):
        if f["is_honeypot"]:
            s, comps, flags = 0.0, {}, ["honeypot: " + (f["hp_reasons"][0] if f["hp_reasons"] else "impossible")]
        else:
            s, comps, flags = score_candidate(f, jd, semantic_sim=float(sims[i]))
        rows.append((round(s, 4), f["candidate_id"], f, comps, flags))

    rows.sort(key=lambda t: (-t[0], t[1]))
    out = []
    for rank, (score, cid, f, comps, flags) in enumerate(rows, start=1):
        out.append({
            "rank": rank, "candidate_id": cid, "score": f"{score:.4f}",
            "title": f["current_title"], "yoe": f["yoe"],
            "reasoning": build_reasoning(f, comps, flags),
        })
    df = pd.DataFrame(out)
    st.subheader(f"Ranked {len(df)} candidates")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download ranked CSV",
                       df[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False),
                       file_name="ranked_sample.csv", mime="text/csv")
else:
    st.info("Upload a sample or click **Use bundled sample** to run the ranker.")
