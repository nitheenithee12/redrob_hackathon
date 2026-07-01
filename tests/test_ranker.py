"""
tests/test_ranker.py
====================
Fast sanity tests that run without the full 100K pool. Run with:  pytest -q
They double as documentation of the ranker's guarantees.
"""

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import detect_honeypot, extract   # noqa: E402
from jd_profile import JDProfile                 # noqa: E402
from scoring import build_reasoning, score_candidate  # noqa: E402

JD = JDProfile()
TODAY = date(2026, 6, 1)


def _strong_candidate():
    return {
        "candidate_id": "CAND_9000001",
        "profile": {
            "anonymized_name": "Test Strong", "headline": "Search & Ranking Engineer",
            "summary": "Built embedding-based retrieval and learning-to-rank recommendation systems in production; evaluated with NDCG and A/B tests.",
            "location": "Pune, Maharashtra", "country": "India",
            "years_of_experience": 7.0, "current_title": "Machine Learning Engineer",
            "current_company": "Swiggy", "current_company_size": "5001-10000",
            "current_industry": "Food Delivery",
        },
        "career_history": [{
            "company": "Swiggy", "title": "ML Engineer", "start_date": "2021-01-01",
            "end_date": None, "duration_months": 40, "is_current": True,
            "industry": "Food Delivery", "company_size": "5001-10000",
            "description": "Owned semantic search and ranking using FAISS and sentence-transformers; hybrid retrieval; NDCG/MRR evaluation.",
        }],
        "education": [], "skills": [
            {"name": "Retrieval", "proficiency": "advanced", "endorsements": 20, "duration_months": 40},
        ],
        "redrob_signals": {
            "recruiter_response_rate": 0.85, "last_active_date": "2026-05-20",
            "open_to_work_flag": True, "profile_completeness_score": 90,
            "willing_to_relocate": True, "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.8, "skill_assessment_scores": {},
            "notice_period_days": 30, "github_activity_score": 40,
        },
    }


def _honeypot_candidate():
    c = _strong_candidate()
    c["candidate_id"] = "CAND_9000002"
    c["profile"]["current_title"] = "Mobile Developer"
    c["skills"] = [
        {"name": "MLflow", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "Photoshop", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "TTS", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
    ]
    return c


def test_honeypot_detected():
    is_hp, reasons = detect_honeypot(_honeypot_candidate(), TODAY)
    assert is_hp and reasons


def test_strong_beats_offtarget():
    strong = _strong_candidate()
    off = _strong_candidate()
    off["candidate_id"] = "CAND_9000003"
    off["profile"]["current_title"] = "Marketing Manager"
    off["career_history"][0]["title"] = "Marketing Manager"
    fs = extract(strong, JD, TODAY)
    fo = extract(off, JD, TODAY)
    ss, _, _ = score_candidate(fs, JD, 0.5)
    so, _, _ = score_candidate(fo, JD, 0.5)
    assert ss > so, "on-target ML engineer must outrank keyword-stuffed marketer"


def test_reasoning_no_hallucination():
    f = extract(_strong_candidate(), JD, TODAY)
    s, comps, flags = score_candidate(f, JD, 0.5)
    text = build_reasoning(f, comps, flags).lower()
    # only phrases actually present may appear
    assert "retrieval" in text or "ranking" in text
    assert "faiss" not in text or "faiss" in f["_text"] if "_text" in f else True


def test_deterministic():
    f = extract(_strong_candidate(), JD, TODAY)
    a, _, _ = score_candidate(f, JD, 0.5)
    b, _, _ = score_candidate(f, JD, 0.5)
    assert a == b
