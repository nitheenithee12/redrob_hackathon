"""
jd_profile.py
=============
A *structured understanding* of the released Job Description
(Senior AI Engineer — Founding Team @ Redrob AI).

The whole point of this challenge is to rank the way a great recruiter would,
not to keyword-match. A recruiter does not read a JD as a bag of words; they
read it as a set of *intents*:

    - what the role actually is (retrieval / ranking / search / recsys ownership)
    - who is explicitly disqualified (pure research, consulting-only, title
      chasers, CV/speech-without-NLP, LangChain-only "AI experience")
    - what is genuinely required vs. nice-to-have
    - what the "ideal candidate" looks like between the lines

We encode that reading here, once, as explicit data. Every downstream scoring
decision refers back to this object, so the ranker's behaviour is auditable and
defensible in an interview — nothing is hidden inside an opaque embedding.

This is deliberately hand-authored from the JD text. It is the JD-understanding
layer; `features.py` is the candidate-understanding layer; `scoring.py` maps
one onto the other.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# --------------------------------------------------------------------------- #
# Lexicons. These are *evidence phrases*, not skill tags. We look for them in
# free-text (summaries, job descriptions) because the JD is explicit that the
# right candidate may never write the word "RAG" but will describe having built
# retrieval. Matching on described work is how we get "beyond keyword".
# --------------------------------------------------------------------------- #

# Core intelligence-layer work the role owns. Strongest positive evidence.
CORE_WORK_PHRASES: List[str] = [
    "recommendation system", "recommender", "recsys", "ranking model",
    "learning to rank", "learning-to-rank", "ltr", "retrieval",
    "semantic search", "vector search", "embedding", "embeddings",
    "dense retrieval", "hybrid search", "hybrid retrieval", "reranking",
    "re-ranking", "search relevance", "search engine", "information retrieval",
    "nearest neighbor", "ann index", "faiss", "candidate ranking",
    "personalization", "discovery feed", "matching system", "query understanding",
]

# Retrieval / vector infra the JD names as required operational experience.
VECTOR_INFRA_PHRASES: List[str] = [
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "vespa", "vector database", "vector db",
    "bm25", "inverted index", "annoy", "hnsw", "scann",
]

# Embedding-model tooling the JD names (model-agnostic).
EMBEDDING_TOOL_PHRASES: List[str] = [
    "sentence-transformers", "sentence transformer", "sbert",
    "openai embedding", "bge", "e5", "cohere embed", "text-embedding",
    "word2vec", "fasttext", "glove", "transformer encoder",
]

# Evaluation-of-ranking maturity — explicitly a hard requirement.
EVAL_PHRASES: List[str] = [
    "ndcg", "mrr", "map@", "mean average precision", "precision@",
    "recall@", "a/b test", "ab test", "offline evaluation", "online evaluation",
    "eval framework", "evaluation framework", "offline-to-online",
    "relevance judgment", "ranking metric", "click-through", "ctr uplift",
]

# LLM / modern ML depth (nice-to-have but a positive depth signal).
LLM_DEPTH_PHRASES: List[str] = [
    "fine-tune", "fine-tuning", "lora", "qlora", "peft", "rag",
    "retrieval augmented", "retrieval-augmented", "llm", "large language model",
    "prompt engineering", "distillation", "quantization", "xgboost",
    "lightgbm", "gradient boost",
]

# Generic ML/NLP context that supports fit but is not itself decisive.
ML_CONTEXT_PHRASES: List[str] = [
    "machine learning", "deep learning", "nlp", "natural language",
    "feature engineering", "model deployment", "mlops", "inference",
    "pytorch", "tensorflow", "scikit-learn", "production ml", "ml pipeline",
    "data pipeline", "spark", "airflow",
]

# ---- NEGATIVE / TRAP evidence ---------------------------------------------- #

# "Things we explicitly do NOT want": pure computer-vision / speech / robotics
# WITHOUT NLP/IR exposure. Presence alone is not fatal; dominance without any
# NLP/IR/retrieval evidence is heavily penalised in scoring.py.
OFF_DOMAIN_PHRASES: List[str] = [
    "computer vision", "image classification", "object detection",
    "image segmentation", "speech recognition", "text-to-speech", "tts",
    "asr", "robotics", "slam", "point cloud", "autonomous driving",
    "video analytics", "ocr", "pose estimation",
]

# LangChain-tutorial / framework-enthusiast trap.
FRAMEWORK_HYPE_PHRASES: List[str] = [
    "langchain", "llamaindex", "llama-index", "autogpt", "auto-gpt",
    "crewai", "flowise", "no-code", "wrapper around openai",
    "calling openai", "chatgpt wrapper",
]

# Consulting / pure-services employers the JD calls out by name.
CONSULTING_EMPLOYERS: List[str] = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl technologies", "tech mahindra",
    "mindtree", "mphasis", "ltimindtree", "l&t infotech", "deloitte",
    "persistent systems", "hexaware", "birlasoft", "coforge",
]

# Pure-research contexts the JD disqualifies when there is no production signal.
RESEARCH_ONLY_PHRASES: List[str] = [
    "research scientist", "phd student", "postdoc", "post-doctoral",
    "academic lab", "research assistant", "research fellow", "publications",
    "published papers", "research intern",
]


# --------------------------------------------------------------------------- #
# Role targeting. A recruiter reads the *title trajectory*, not the skill list.
# A Marketing Manager with 9 AI skills is not an AI engineer.
# --------------------------------------------------------------------------- #

# Titles that ARE the role (or a direct feeder into it). Matched case-insensitive
# as substrings against current + historical titles.
ON_TARGET_TITLE_TOKENS: List[str] = [
    "ml engineer", "machine learning engineer", "ai engineer",
    "applied ml", "applied scientist", "applied ai", "nlp engineer",
    "search engineer", "recommendation", "recsys", "relevance engineer",
    "data scientist", "research engineer", "ranking engineer",
]

# Adjacent titles: plausible feeders (software/data/backend) — count if the
# described work is ML/retrieval, otherwise weak.
ADJACENT_TITLE_TOKENS: List[str] = [
    "software engineer", "backend engineer", "data engineer",
    "analytics engineer", "full stack", "platform engineer",
    "senior software engineer",
]

# Titles that are clearly NOT this role. A great skill list under one of these
# titles is the JD's canonical keyword-stuffer trap.
OFF_TARGET_TITLE_TOKENS: List[str] = [
    "hr manager", "marketing manager", "sales executive", "accountant",
    "content writer", "graphic designer", "civil engineer",
    "mechanical engineer", "operations manager", "customer support",
    "business analyst", "project manager", "recruiter", "designer",
]

# Cities the JD prefers (Pune/Noida primary; Hyderabad/Mumbai/Delhi NCR welcome).
PREFERRED_CITIES: List[str] = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon",
    "gurugram", "ncr", "new delhi",
]


@dataclass
class JDProfile:
    """The recruiter's mental model of the role, as data."""

    title: str = "Senior AI Engineer — Founding Team"
    company: str = "Redrob AI (Series A, AI-native talent intelligence)"

    # Experience: 5-9 is a guide, not a gate. Ideal ~6-8; hard tails penalised
    # gently (JD: "we'll seriously consider candidates outside the band").
    exp_ideal_low: float = 6.0
    exp_ideal_high: float = 8.0
    exp_soft_low: float = 4.0
    exp_soft_high: float = 12.0

    # Behavioural availability: a perfect-on-paper but unreachable candidate is,
    # for hiring purposes, not available. These set where the multiplier bites.
    inactive_days_threshold: int = 120     # ~4 months since last active
    low_response_rate: float = 0.15        # JD's own "5% response" example region

    core_work: List[str] = field(default_factory=lambda: CORE_WORK_PHRASES)
    vector_infra: List[str] = field(default_factory=lambda: VECTOR_INFRA_PHRASES)
    embedding_tools: List[str] = field(default_factory=lambda: EMBEDDING_TOOL_PHRASES)
    eval_phrases: List[str] = field(default_factory=lambda: EVAL_PHRASES)
    llm_depth: List[str] = field(default_factory=lambda: LLM_DEPTH_PHRASES)
    ml_context: List[str] = field(default_factory=lambda: ML_CONTEXT_PHRASES)

    off_domain: List[str] = field(default_factory=lambda: OFF_DOMAIN_PHRASES)
    framework_hype: List[str] = field(default_factory=lambda: FRAMEWORK_HYPE_PHRASES)
    consulting: List[str] = field(default_factory=lambda: CONSULTING_EMPLOYERS)
    research_only: List[str] = field(default_factory=lambda: RESEARCH_ONLY_PHRASES)

    on_target_titles: List[str] = field(default_factory=lambda: ON_TARGET_TITLE_TOKENS)
    adjacent_titles: List[str] = field(default_factory=lambda: ADJACENT_TITLE_TOKENS)
    off_target_titles: List[str] = field(default_factory=lambda: OFF_TARGET_TITLE_TOKENS)
    preferred_cities: List[str] = field(default_factory=lambda: PREFERRED_CITIES)

    def jd_query_text(self) -> str:
        """A dense natural-language query used for the semantic (TF-IDF) layer.

        This is the JD compressed to the phrases a recruiter would actually
        search on. It is intentionally about *work done*, not job perks.
        """
        return (
            "senior ai engineer owning the intelligence layer: ranking, retrieval "
            "and matching systems in production. embeddings based retrieval, "
            "hybrid search, vector databases such as faiss pinecone weaviate qdrant "
            "elasticsearch opensearch. learning to rank, recommendation systems, "
            "semantic search, reranking, personalization, discovery feed. rigorous "
            "evaluation of ranking with ndcg mrr map precision recall, offline and "
            "online a/b testing. strong python and production ml deployment at a "
            "product company, not pure research and not pure services consulting. "
            "llm re-ranking, fine-tuning lora, rag, xgboost lightgbm. shipped an "
            "end to end search or recommendation system to real users at scale. "
            "based in or willing to relocate to pune noida hyderabad mumbai delhi."
        )


# Component weights for the interpretable scoring model (sum to 1.0 before the
# behavioural multiplier and gates are applied). Tuned to the JD's emphasis:
# role/work fit dominates; keywords alone never can.
SCORE_WEIGHTS: Dict[str, float] = {
    "role_fit":        0.26,  # is the trajectory actually this role?
    "core_work":       0.24,  # evidence of shipped retrieval/ranking/recsys
    "retrieval_stack": 0.12,  # vector infra + embedding tooling
    "evaluation":      0.08,  # ranking-eval maturity (hard requirement)
    "experience":      0.10,  # years, band-fit
    "skill_trust":     0.10,  # skills, discounted by endorsement/duration trust
    "product_company": 0.06,  # product vs services employer history
    "location":        0.04,  # preferred-city / relocate fit
}
