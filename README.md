# Redrob Intelligent Candidate Discovery & Ranking — `Team_unique_2026`

> Rank candidates the way a great recruiter would — by understanding **who fits the role**, not by counting keywords.

Given the released JD (*Senior AI Engineer — Founding Team @ Redrob AI*) and a
pool of **100,000 candidates**, this system produces a trustworthy **top-100
shortlist** with a specific, honest one-line justification for every pick.

It runs **CPU-only, offline, in ~2 minutes** on the 100K pool (well inside the
5-min / 16 GB / no-network budget), and every ranking decision is **auditable** —
no opaque embedding blob decides who gets hired.

---

## TL;DR — reproduce the submission

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./Team_unique_2026.csv
python validate_submission.py ./Team_unique_2026.csv   # -> "Submission is valid."
```

`candidates.jsonl` (or `candidates.jsonl.gz`) is the pool from the hackathon
bundle. Output is a spec-compliant CSV: `candidate_id,rank,score,reasoning`.

---

## Why not "just embed everything and sort by cosine"?

Because that is precisely the trap the dataset is built to punish. Pure
embedding similarity happily ranks:

- **keyword stuffers** — a *Marketing Manager* whose skills list contains 9 AI
  terms looks great to a bag-of-words model;
- **honeypots** — profiles with *"expert"* skills used **0 months**, or tenure
  that predates the company's existence;
- **unreachable** perfect-on-paper candidates who last logged in 8 months ago.

A great recruiter reads the **gap between what the JD says and what it means**.
So do we — explicitly.

---

## Architecture

```
candidates.jsonl ──► [1] stream + featurise ──► [2] score ──► [3] rank ──► CSV
                          │                          │
                     features.py                scoring.py
                     jd_profile.py              semantic.py (TF-IDF)
```

**1 · JD understanding (`src/jd_profile.py`).**
The JD is encoded *once* as a structured recruiter mental-model: the target
roles, the explicit disqualifiers (pure research, consulting-only, title-chasers,
CV/speech-without-NLP, LangChain-only), the experience band, the preferred
cities, and evidence lexicons for *work described* (retrieval, ranking, recsys,
vector DBs, evaluation metrics). This is the auditable "what good looks like".

**2 · Candidate understanding (`src/features.py`).**
Each profile is parsed into interpretable features from the fields a recruiter
actually reads (headline, summary, every role description, skills, signals).
Two things happen here beyond parsing:
- **Honeypot detection** — *structural* impossibility (expert@0-months, dates
  that can't sum, future start dates). Honeypots are forced to the bottom.
- **Trap flags** — off-target current title, services-only career, title-chaser
  hopping pattern, off-domain dominance.

**3 · Scoring (`src/scoring.py`).** An **interpretable additive model** over 8
components, then a **multiplicative** availability modifier and trap gates:

```
final = ( Σ wᵢ·componentᵢ ) · availability · trap_gates  + small semantic tilt
```

| component | weight | what it rewards |
|---|---|---|
| role_fit | 0.26 | trajectory *is* an AI/ML/search/recsys engineer |
| core_work | 0.24 | shipped retrieval / ranking / recsys / search |
| retrieval_stack | 0.12 | vector DBs + embedding tooling |
| evaluation | 0.08 | NDCG / MRR / MAP / A-B maturity (hard req) |
| experience | 0.10 | ~5–9 yrs band fit (soft) |
| skill_trust | 0.10 | skills discounted by endorsements + duration + assessment |
| product_company | 0.06 | product vs services-only history |
| location | 0.04 | preferred city / willing to relocate |

- The **availability multiplier** (0.4–1.08) down-weights dormant, low-response,
  not-open-to-work profiles — the JD is explicit that an unreachable candidate is
  *not actually available*.
- The **skill-trust** discount defeats keyword stuffing: an unendorsed skill
  used 0 months barely counts; an endorsed skill used for years, with a matching
  Redrob assessment score, counts fully.
- **`semantic.py`** adds an 8% tilt from **TF-IDF cosine** (scikit-learn, offline)
  so paraphrased-but-relevant profiles surface — without letting lexical
  similarity override the role logic. An optional local sentence-transformer
  re-rank (`--embeddings <dir>`) is supported but disabled by default so the
  reproducible path never depends on downloaded weights.

**Reasoning** is generated from the same features that drove the score, so it is
**specific, varied, and cannot hallucinate** — a phrase only appears if the
profile actually contains it, and the tone matches the rank.

---

## Results on the released 100K pool

- **0 honeypots** in the top 100 (disqualifier threshold is >10%).
- **100%** of the top 100 are genuine AI/ML/Search/Recsys/NLP engineers — zero
  HR Managers, Marketing Managers, or Accountants (the traps that top the
  provided `sample_submission.csv`).
- Average **6.5 yrs** experience (squarely in the JD's ideal band) and average
  **0.69** recruiter-response rate (reachable, engaged candidates).
- Runtime **~2 min**, peak RSS **~1.5 GB**, single-threaded, no network.

---

## Repo layout

```
rank.py                     # single-command entrypoint (reproduce command)
app.py                      # Streamlit sandbox demo (Section 10.5)
requirements.txt
submission_metadata.yaml
validate_submission.py      # official format validator (copy)
src/
  jd_profile.py             # structured JD understanding + weights
  features.py               # feature extraction + honeypot/trap detection
  scoring.py                # interpretable scoring + reasoning generation
  semantic.py               # TF-IDF semantic layer (+ optional embedding path)
tests/
  test_ranker.py            # determinism, honeypot, on-target > off-target
```

## Sandbox / demo

`app.py` is a self-contained Streamlit app that accepts a small sample
(≤100 candidates), runs the full ranker on CPU, and returns a ranked CSV — deploy
free on Streamlit Cloud or HuggingFace Spaces (`streamlit run app.py`).

## Reproducibility notes

- Deterministic: same input file → byte-identical CSV.
- No GPU, no network, no hosted-LLM calls during ranking.
- A fixed reference date (`--asof`, default `2026-06-01`) makes the
  activity-based signals reproducible regardless of wall-clock time.

## AI tools

Claude was used for architecture discussion, docs, and code review. All ranking
logic was designed by the team; no candidate data was sent to any hosted LLM,
and the ranker makes zero LLM/network calls. See `submission_metadata.yaml`.
