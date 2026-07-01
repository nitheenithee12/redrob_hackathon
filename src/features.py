"""
features.py
===========
Turns one raw candidate record into an interpretable feature dict plus a list of
human-readable evidence strings, and detects honeypots / trap profiles.

Design principles
-----------------
1. Everything is derived from fields that actually exist in the profile, so the
   reasoning we later emit can never hallucinate — a claim is only made if the
   feature that backs it is non-zero.
2. Honeypot detection is *structural*, not a blocklist. We look for internal
   impossibility (expert skill with 0 months used; tenure that predates the
   company's plausible existence; experience that cannot sum up). The challenge
   says a good ranker should avoid honeypots "naturally"; we make the structural
   contradictions explicit so they fall out of scoring rather than being
   special-cased into a lookup table.
3. Text scanning is done once over a lowercased concatenation of the fields a
   recruiter actually reads: headline, summary, and every role description.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, List, Tuple

from jd_profile import JDProfile


def _to_date(s: Any) -> date | None:
    if not s or not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _count_phrases(text: str, phrases: List[str]) -> Tuple[int, List[str]]:
    """Return (#distinct phrases present, the matched phrases)."""
    hits = [p for p in phrases if p in text]
    return len(hits), hits


def profile_text(cand: Dict) -> str:
    """The free text a recruiter reads, lowercased and concatenated."""
    p = cand.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", "")]
    for job in cand.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    for sk in cand.get("skills", []):
        parts.append(sk.get("name", ""))
    return " ".join(parts).lower()


def detect_honeypot(cand: Dict, today: date) -> Tuple[bool, List[str]]:
    """Structural impossibility detection.

    Returns (is_honeypot, reasons). We require the contradiction to be
    unambiguous so we do not wrongly kill real strong candidates.
    """
    reasons: List[str] = []
    p = cand.get("profile", {})
    yoe = float(p.get("years_of_experience", 0) or 0)

    # (a) "expert" proficiency in a skill used for 0 months — impossible.
    expert_zero = [
        s.get("name", "?")
        for s in cand.get("skills", [])
        if s.get("proficiency") == "expert" and (s.get("duration_months", 0) or 0) == 0
    ]
    if len(expert_zero) >= 2:
        reasons.append(
            f"{len(expert_zero)} 'expert' skills with 0 months of use "
            f"(e.g. {', '.join(expert_zero[:3])})"
        )

    # (b) tenure at a company that exceeds the time since the company plausibly
    #     could have employed them, i.e. a role duration that predates a
    #     start_date/end_date span by a wide margin. We approximate with:
    #     duration_months grossly larger than the actual start->end/today span.
    for job in cand.get("career_history", []):
        start = _to_date(job.get("start_date"))
        end = _to_date(job.get("end_date")) or today
        dur = int(job.get("duration_months", 0) or 0)
        if start:
            span = (end.year - start.year) * 12 + (end.month - start.month)
            if dur - span >= 24:  # claims 2+ yrs more than the dates allow
                reasons.append(
                    f"role at {job.get('company','?')} claims {dur}mo but dates "
                    f"span only ~{max(span,0)}mo"
                )
                break

    # (c) total tenure impossibly exceeds stated years of experience.
    total_months = sum(int(j.get("duration_months", 0) or 0) for j in cand.get("career_history", []))
    if yoe > 0 and total_months > yoe * 12 + 60:
        reasons.append(
            f"career history sums to {total_months}mo but only {yoe:.1f} yrs "
            f"experience stated"
        )

    # (d) a role that starts in the future.
    for job in cand.get("career_history", []):
        start = _to_date(job.get("start_date"))
        if start and start > today:
            reasons.append(f"role at {job.get('company','?')} starts in the future")
            break

    return (len(reasons) > 0, reasons)


def extract(cand: Dict, jd: JDProfile, today: date) -> Dict[str, Any]:
    """Compute the full interpretable feature dict for one candidate."""
    p = cand.get("profile", {})
    sig = cand.get("redrob_signals", {}) or {}
    text = profile_text(cand)

    cur_title = (p.get("current_title", "") or "").lower()
    all_titles = " ".join(
        [cur_title] + [(j.get("title", "") or "").lower() for j in cand.get("career_history", [])]
    )

    # ---- title / role targeting -------------------------------------------- #
    on_target = any(tok in all_titles for tok in jd.on_target_titles)
    cur_on_target = any(tok in cur_title for tok in jd.on_target_titles)
    adjacent = any(tok in all_titles for tok in jd.adjacent_titles)
    off_target_current = any(tok in cur_title for tok in jd.off_target_titles) and not cur_on_target

    # ---- evidence counts (beyond keyword: work described, not tags) -------- #
    n_core, core_hits = _count_phrases(text, jd.core_work)
    n_vec, vec_hits = _count_phrases(text, jd.vector_infra)
    n_emb, emb_hits = _count_phrases(text, jd.embedding_tools)
    n_eval, eval_hits = _count_phrases(text, jd.eval_phrases)
    n_llm, llm_hits = _count_phrases(text, jd.llm_depth)
    n_ml, _ = _count_phrases(text, jd.ml_context)
    n_off_dom, off_dom_hits = _count_phrases(text, jd.off_domain)
    n_hype, hype_hits = _count_phrases(text, jd.framework_hype)

    # ---- employer history: product vs services ---------------------------- #
    companies = [(j.get("company", "") or "").lower() for j in cand.get("career_history", [])]
    cur_company = (p.get("current_company", "") or "").lower()
    consulting_hits = [c for c in jd.consulting if any(c in comp for comp in companies + [cur_company])]
    n_jobs = len(companies)
    consulting_only = n_jobs > 0 and all(
        any(c in comp for c in jd.consulting) for comp in companies if comp
    )

    # ---- research-only -----------------------------------------------------#
    n_research, research_hits = _count_phrases(text, jd.research_only)
    production_signal = (n_core + n_vec + n_emb) > 0 or "production" in text or "shipped" in text
    research_only = n_research >= 2 and not production_signal

    # ---- skills, with a trust discount ------------------------------------ #
    # Keyword stuffers list many skills with low endorsements and 0 duration.
    # We reward skills that are *substantiated* by endorsements or real usage,
    # and by matching Redrob assessment scores.
    prof_weight = {"beginner": 0.25, "intermediate": 0.55, "advanced": 0.8, "expert": 1.0}
    assess = sig.get("skill_assessment_scores", {}) or {}
    trusted_skill_score = 0.0
    n_skills = len(cand.get("skills", []))
    for s in cand.get("skills", []):
        name = s.get("name", "")
        prof = prof_weight.get(s.get("proficiency", "beginner"), 0.25)
        endorse = min((s.get("endorsements", 0) or 0) / 30.0, 1.0)
        dur = min((s.get("duration_months", 0) or 0) / 36.0, 1.0)
        # substantiation: a claim backed by usage + endorsement + (if present)
        # an assessment score is trusted; a bare claim is heavily discounted.
        substantiation = 0.15 + 0.45 * dur + 0.4 * endorse
        assess_boost = 1.0
        if name in assess:
            assess_boost = 0.6 + 0.8 * (assess[name] / 100.0)  # 0.6..1.4
        trusted_skill_score += prof * substantiation * assess_boost
    # normalise to roughly 0..1 (a strong genuine profile lands ~0.6-1.0)
    trusted_skill_score = min(trusted_skill_score / 8.0, 1.0)

    # ---- experience -------------------------------------------------------- #
    yoe = float(p.get("years_of_experience", 0) or 0)

    # ---- location ---------------------------------------------------------- #
    loc = (p.get("location", "") or "").lower()
    country = (p.get("country", "") or "").lower()
    in_preferred_city = any(city in loc for city in jd.preferred_cities)
    in_india = "india" in country
    willing_relocate = bool(sig.get("willing_to_relocate", False))

    # ---- title-chaser (job hopping for title bumps) ------------------------ #
    durations = [int(j.get("duration_months", 0) or 0) for j in cand.get("career_history", [])]
    short_stints = sum(1 for d in durations if 0 < d < 18)
    title_chaser = n_jobs >= 4 and short_stints >= 3

    # ---- behavioural / availability signals -------------------------------- #
    last_active = _to_date(sig.get("last_active_date"))
    days_inactive = (today - last_active).days if last_active else 999
    resp_rate = float(sig.get("recruiter_response_rate", 0) or 0)
    open_flag = bool(sig.get("open_to_work_flag", False))
    completeness = float(sig.get("profile_completeness_score", 0) or 0)
    interview_rate = float(sig.get("interview_completion_rate", 0) or 0)
    saved = int(sig.get("saved_by_recruiters_30d", 0) or 0)
    github = float(sig.get("github_activity_score", -1) or -1)
    notice = int(sig.get("notice_period_days", 90) or 90)

    return {
        "candidate_id": cand.get("candidate_id"),
        "name": p.get("anonymized_name", ""),
        "current_title": p.get("current_title", ""),
        "current_company": p.get("current_company", ""),
        "yoe": yoe,
        "location": p.get("location", ""),
        "country": p.get("country", ""),
        # role
        "on_target": on_target,
        "cur_on_target": cur_on_target,
        "adjacent": adjacent,
        "off_target_current": off_target_current,
        "title_chaser": title_chaser,
        # work evidence
        "n_core": n_core, "core_hits": core_hits,
        "n_vec": n_vec, "vec_hits": vec_hits,
        "n_emb": n_emb, "emb_hits": emb_hits,
        "n_eval": n_eval, "eval_hits": eval_hits,
        "n_llm": n_llm, "llm_hits": llm_hits,
        "n_ml": n_ml,
        "n_off_dom": n_off_dom, "off_dom_hits": off_dom_hits,
        "n_hype": n_hype, "hype_hits": hype_hits,
        # employer / research
        "consulting_only": consulting_only,
        "consulting_hits": consulting_hits,
        "research_only": research_only,
        "n_jobs": n_jobs,
        # skills
        "trusted_skill_score": trusted_skill_score,
        "n_skills": n_skills,
        # location
        "in_preferred_city": in_preferred_city,
        "in_india": in_india,
        "willing_relocate": willing_relocate,
        # behavioural
        "days_inactive": days_inactive,
        "resp_rate": resp_rate,
        "open_flag": open_flag,
        "completeness": completeness,
        "interview_rate": interview_rate,
        "saved": saved,
        "github": github,
        "notice": notice,
        # raw text kept only transiently for the semantic layer
        "_text": text,
    }
