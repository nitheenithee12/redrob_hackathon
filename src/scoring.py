"""
scoring.py
==========
Maps candidate features (features.py) onto the JD understanding (jd_profile.py)
to produce (a) a final score in [0, 1], and (b) a specific, non-hallucinated,
1-2 sentence reasoning string.

The model is an *interpretable additive score* over eight components, followed
by a multiplicative behavioural-availability modifier and a set of soft gates
for the traps the JD names. Additive-then-multiplicative is deliberate:

    final = (sum_i w_i * component_i) * availability * trap_gates + tiny semantic tilt

- The additive core answers "is this the right person on paper?"
- The availability multiplier answers "can we actually hire them?" — the JD is
  explicit that an unreachable perfect profile should be down-weighted.
- The trap gates answer "is this a keyword-stuffer / off-target / honeypot?"
  They multiply the score down rather than hard-dropping, so a borderline real
  candidate is not accidentally deleted; a blatant trap collapses toward zero.

Nothing here calls a network or an LLM. It is arithmetic over parsed fields,
which is exactly why it fits the 5-min / CPU-only / offline budget and why it
can be defended line-by-line in an interview.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from jd_profile import JDProfile, SCORE_WEIGHTS


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _role_fit(f: Dict, jd: JDProfile) -> float:
    """Is the candidate's trajectory actually this role?"""
    if f["cur_on_target"]:
        s = 1.0
    elif f["on_target"]:
        s = 0.8            # was on-target historically; currently adjacent
    elif f["adjacent"]:
        # adjacent title only counts if the described work is ML/retrieval
        s = 0.5 if (f["n_core"] + f["n_ml"]) > 0 else 0.25
    else:
        s = 0.1
    if f["off_target_current"]:
        # canonical trap: off-target current title. Cap hard regardless of skills.
        s = min(s, 0.15)
    return _clip(s)


def _core_work(f: Dict) -> float:
    """Evidence of shipped retrieval / ranking / recsys / search work."""
    # diminishing returns; 3+ distinct core phrases is already strong
    core = min(f["n_core"] / 3.0, 1.0)
    return _clip(0.85 * core + 0.15 * min(f["n_llm"] / 3.0, 1.0))


def _retrieval_stack(f: Dict) -> float:
    vec = min(f["n_vec"] / 2.0, 1.0)
    emb = min(f["n_emb"] / 2.0, 1.0)
    return _clip(0.6 * vec + 0.4 * emb)


def _evaluation(f: Dict) -> float:
    return _clip(min(f["n_eval"] / 2.0, 1.0))


def _experience(f: Dict, jd: JDProfile) -> float:
    y = f["yoe"]
    if jd.exp_ideal_low <= y <= jd.exp_ideal_high:
        return 1.0
    if jd.exp_soft_low <= y <= jd.exp_soft_high:
        # linear falloff toward the soft edges
        if y < jd.exp_ideal_low:
            return _clip(0.6 + 0.4 * (y - jd.exp_soft_low) / (jd.exp_ideal_low - jd.exp_soft_low))
        return _clip(0.6 + 0.4 * (jd.exp_soft_high - y) / (jd.exp_soft_high - jd.exp_ideal_high))
    # outside soft band — JD still "seriously considers" strong signals
    return 0.3


def _skill_trust(f: Dict) -> float:
    return _clip(f["trusted_skill_score"])


def _product_company(f: Dict) -> float:
    if f["consulting_only"]:
        return 0.05
    if f["consulting_hits"]:
        return 0.55         # some services history but not exclusively
    return 0.9


def _location(f: Dict, jd: JDProfile) -> float:
    if f["in_preferred_city"]:
        return 1.0
    if f["in_india"] and f["willing_relocate"]:
        return 0.8
    if f["in_india"]:
        return 0.6
    if f["willing_relocate"]:
        return 0.45
    return 0.2              # outside India, won't relocate; JD: no visa sponsor


def _availability_multiplier(f: Dict, jd: JDProfile) -> float:
    """A 0.4..1.08 multiplier for 'can we actually hire this person?'.

    Never sends a strong candidate to zero on behaviour alone, but a dormant,
    non-responsive profile is materially discounted, exactly as the JD asks.
    """
    m = 1.0
    if f["days_inactive"] > jd.inactive_days_threshold:
        m *= 0.6
    elif f["days_inactive"] > 60:
        m *= 0.85
    if f["resp_rate"] < jd.low_response_rate:
        m *= 0.7
    elif f["resp_rate"] < 0.35:
        m *= 0.9
    if not f["open_flag"]:
        m *= 0.9
    if f["completeness"] < 50:
        m *= 0.9
    # small positive lift for verified recruiter demand / reliability
    if f["saved"] >= 3:
        m *= 1.04
    if f["interview_rate"] >= 0.7:
        m *= 1.04
    return _clip(m, 0.4, 1.08)


def _trap_gate(f: Dict) -> Tuple[float, List[str]]:
    """Multiplicative penalty for the JD's named anti-patterns. Returns
    (multiplier in (0,1], list of triggered flags)."""
    g = 1.0
    flags: List[str] = []

    # off-domain (CV/speech/robotics) dominance with no NLP/IR/retrieval anchor
    if f["n_off_dom"] >= 2 and (f["n_core"] + f["n_vec"] + f["n_emb"]) == 0:
        g *= 0.4
        flags.append("off-domain (CV/speech) without NLP/IR")

    # framework-hype / LangChain-only
    if f["n_hype"] >= 1 and f["n_core"] == 0 and f["n_vec"] == 0:
        g *= 0.6
        flags.append("framework-hype without systems evidence")

    # title-chaser
    if f["title_chaser"]:
        g *= 0.75
        flags.append("frequent short stints (title-chaser pattern)")

    # research-only, no production
    if f["research_only"]:
        g *= 0.5
        flags.append("research-only, no production deployment")

    return g, flags


def score_candidate(
    f: Dict, jd: JDProfile, semantic_sim: float = 0.0
) -> Tuple[float, Dict[str, float], List[str]]:
    """Return (final_score, component_breakdown, trap_flags)."""
    comps = {
        "role_fit":        _role_fit(f, jd),
        "core_work":       _core_work(f),
        "retrieval_stack": _retrieval_stack(f),
        "evaluation":      _evaluation(f),
        "experience":      _experience(f, jd),
        "skill_trust":     _skill_trust(f),
        "product_company": _product_company(f),
        "location":        _location(f, jd),
    }
    base = sum(SCORE_WEIGHTS[k] * comps[k] for k in comps)

    # semantic tilt: a small nudge from TF-IDF cosine so paraphrased-but-relevant
    # profiles surface, without letting lexical similarity override role logic.
    base = 0.92 * base + 0.08 * _clip(semantic_sim)

    avail = _availability_multiplier(f, jd)
    gate, flags = _trap_gate(f)

    final = _clip(base * avail * gate)
    comps["_availability"] = avail
    comps["_trap_gate"] = gate
    comps["_semantic"] = semantic_sim
    return final, comps, flags


# --------------------------------------------------------------------------- #
# Reasoning generation. Specific, honest, varied — never templated name-inserts,
# never a skill the profile doesn't contain. Every clause is guarded by a
# feature being present, so it cannot hallucinate.
# --------------------------------------------------------------------------- #

def _phrase_list(hits: List[str], limit: int = 3) -> str:
    """De-dupe near-duplicate phrases (e.g. 'learning to rank' / 'learning-to-rank')
    and join the most informative few into readable prose."""
    seen: List[str] = []
    norm_seen = set()
    for h in hits:
        norm = h.replace("-", " ").replace("_", " ")
        if norm in norm_seen:
            continue
        norm_seen.add(norm)
        seen.append(h)
        if len(seen) >= limit:
            break
    return ", ".join(seen)


def build_reasoning(f: Dict, comps: Dict[str, float], flags: List[str]) -> str:
    parts: List[str] = []

    title = f["current_title"] or "Candidate"
    parts.append(f"{title}, {f['yoe']:.1f} yrs")

    # strongest positive evidence — only phrases actually found in the profile
    ev: List[str] = []
    if f["core_hits"]:
        ev.append("shipped " + _phrase_list(f["core_hits"], 3))
    if f["vec_hits"]:
        ev.append(_phrase_list(f["vec_hits"], 2))
    if f["emb_hits"] and not f["vec_hits"]:
        ev.append(_phrase_list(f["emb_hits"], 2))
    if f["eval_hits"]:
        ev.append("evaluated via " + _phrase_list(f["eval_hits"], 2))
    if ev:
        parts.append("; ".join(ev))
    elif f["adjacent"] and f["n_ml"] > 0:
        parts.append("adjacent eng background with ML exposure")

    # role / employer honesty
    if f["consulting_only"]:
        parts.append("career is services-only (JD flags this)")
    elif f["off_target_current"]:
        parts.append("current title is off-target for the AI-eng seat")
    elif f["consulting_hits"]:
        parts.append("some services-firm history")

    # location fit
    if f["in_preferred_city"]:
        city = f["location"].split(",")[0].strip()
        parts.append(f"{city}-based (preferred)")
    elif f["in_india"] and f["willing_relocate"]:
        parts.append("in India, will relocate")
    elif not f["in_india"] and not f["willing_relocate"]:
        parts.append("outside India, no relocate")

    # availability — concern OR positive, never both
    concerns: List[str] = []
    if f["days_inactive"] > 120:
        concerns.append(f"dormant ~{f['days_inactive']}d")
    if f["resp_rate"] < 0.2:
        concerns.append(f"{f['resp_rate']:.0%} recruiter response")
    if not f["open_flag"]:
        concerns.append("not open-to-work")
    if flags:
        concerns.append(flags[0])
    if concerns:
        parts.append("concern: " + ", ".join(concerns[:2]))
    elif f["resp_rate"] >= 0.5 and f["days_inactive"] <= 60:
        parts.append(f"engaged ({f['resp_rate']:.0%} response, active)")

    text = ". ".join([parts[0], "; ".join(parts[1:])]) if len(parts) > 1 else parts[0]
    return (text.rstrip(". ") + ".")[:300]
