import os
import re
from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ragas import evaluate
from ragas.metrics import Faithfulness, ContextRelevance, ResponseRelevancy, ResponseGroundedness

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from ragas.llms import LangchainLLMWrapper

# ---------------------------
# CONFIG
# ---------------------------
DATA_PATH = r"C:\Users\mattp\Downloads\Ragas\Batch_run.json"
LLM_MODEL = "qwen2.5:7b"
HF_EMB_MODEL = "BAAI/bge-base-en-v1.5"

# How many contexts to keep per question (smaller often improves relevancy/groundedness)
KEEP_TOP_K_CONTEXTS = None

# Filter out rows with IDK / errors / empty contexts
FILTER_BAD_ROWS = True

# Strip inline citations like [1] from answers before evaluation (helps embedding similarity)
STRIP_INLINE_CITATIONS = True

# Embeddings device ("cuda" if you have GPU, else "cpu")
EMB_DEVICE = "cpu"

OUT_DIR = os.path.dirname(DATA_PATH)
OUT_CSV = os.path.join(OUT_DIR, "ragas_results_1.csv")
OUT_PNG = os.path.join(OUT_DIR, "ragas_metrics_bar.png")

IDK_PHRASE = "i don't know based on the provided pdfs."
CIT_RE = re.compile(r"\[\d+\]")

# ---------------------------
# HELPERS
# ---------------------------
def trim_k_only(ex, k=KEEP_TOP_K_CONTEXTS):
    ctx = ex.get("contexts") or []
    
    # If k is None → keep everything
    if k is None:
        ex["contexts"] = ctx
    else:
        ex["contexts"] = ctx[:k]
    
    return ex

def ensure_contexts_list(ex):
    """
    RAGAS expects: contexts = List[str]
    Your JSON may contain Contexts as a string or list; normalize it.
    """
    c = ex.get("contexts", [])
    if c is None:
        ex["contexts"] = []
    elif isinstance(c, str):
        ex["contexts"] = [c]
    elif isinstance(c, list):
        ex["contexts"] = [str(x) for x in c if x is not None]
    else:
        ex["contexts"] = [str(c)]
    return ex

def strip_inline_citations(ex):
    ans = ex.get("answer", "")
    ex["answer"] = CIT_RE.sub("", ans or "").strip()
    return ex

def filter_bad_rows_fn(ex):
    ans = (ex.get("answer") or "").strip().lower()
    err = ex.get("error")

    # Drop if error present
    if err not in (None, "", "null"):
        return False

    # Drop if empty answer
    if not ans:
        return False

    # Drop if IDK
    if IDK_PHRASE in ans:
        return False

    # Drop if no contexts
    ctx = ex.get("contexts") or []
    if not isinstance(ctx, list) or len(ctx) == 0:
        return False

    return True

def _pick_col_case_insensitive(df: pd.DataFrame, name: str):
    m = {c.lower(): c for c in df.columns}
    return m.get(name.lower())

# ---------------------------
# MAIN
# ---------------------------
def main():
    # ---------------------------
    # LOAD JSON DATASET
    # ---------------------------
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # ---------------------------
    # RENAME YOUR COLUMNS -> RAGAS STANDARD
    # ---------------------------
    rename_map = {
        "Question": "question",
        "Answer": "answer",
        "Contexts": "contexts",
    }

    existing = set(dataset.column_names)
    to_rename = {k: v for k, v in rename_map.items() if k in existing}
    dataset = dataset.rename_columns(to_rename)

    # ---------------------------
    # NORMALIZE TYPES
    # ---------------------------
    dataset = dataset.map(ensure_contexts_list)

    # Trim contexts
    dataset = dataset.map(lambda ex: trim_k_only(ex, k=KEEP_TOP_K_CONTEXTS))

    # Optionally strip inline citations from answer text
    if STRIP_INLINE_CITATIONS:
        dataset = dataset.map(strip_inline_citations)

    # Optionally filter bad rows
    if FILTER_BAD_ROWS:
        dataset = dataset.filter(filter_bad_rows_fn)

    # ---------------------------
    # VALIDATE REQUIRED COLUMNS
    # ---------------------------
    required = {"question", "answer", "contexts"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {dataset.column_names}")

    # ---------------------------
    # LLM (Ollama/Qwen) wrapped for RAGAS
    # ---------------------------
    judge_llm = ChatOllama(model=LLM_MODEL, temperature=0)
    ragas_llm = LangchainLLMWrapper(judge_llm)

    # ---------------------------
    # EMBEDDINGS (BGE — normalize + device)
    # ---------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name=HF_EMB_MODEL,
        model_kwargs={"device": EMB_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )

    # ---------------------------
    # METRICS
    # ---------------------------
    metrics = [
        Faithfulness(),
        ContextRelevance(),
        ResponseRelevancy(),
        ResponseGroundedness()
    ]

    # ---------------------------
    # EVALUATE
    # ---------------------------
    results = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=embeddings,
        batch_size=1
    )

    print(results)

    # Save per-row results
    df = results.to_pandas()
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved per-row results to:\n{OUT_CSV}")

    # ---------------------------
    # BAR GRAPH: mean of 4 metrics (single PNG)
    # ---------------------------
    col_faith = _pick_col_case_insensitive(df, "faithfulness")

    col_ctx = (
        _pick_col_case_insensitive(df, "context_relevance")
        or _pick_col_case_insensitive(df, "context_relevancy")
        or _pick_col_case_insensitive(df, "nv_context_relevance")
    )

    col_resp_rel = (
        _pick_col_case_insensitive(df, "response_relevancy")
        or _pick_col_case_insensitive(df, "response_relevance")
        or _pick_col_case_insensitive(df, "answer_relevancy")
        or _pick_col_case_insensitive(df, "answer_relevance")
    )

    col_ground = (
        _pick_col_case_insensitive(df, "response_groundedness")
        or _pick_col_case_insensitive(df, "nv_response_groundedness")
        or _pick_col_case_insensitive(df, "groundedness")
        or _pick_col_case_insensitive(df, "response_grounding")
    )

    missing_cols = []
    if not col_faith: missing_cols.append("faithfulness")
    if not col_ctx: missing_cols.append("context_relevance / nv_context_relevance")
    if not col_resp_rel: missing_cols.append("response_relevancy / answer_relevancy")
    if not col_ground: missing_cols.append("response_groundedness / nv_response_groundedness")

    if missing_cols:
        raise ValueError(
            "Could not find one or more metric columns in the RAGAS output.\n"
            f"Missing: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    vals = {
        "Faithfulness": pd.to_numeric(df[col_faith], errors="coerce"),
        "ResponseGroundedness": pd.to_numeric(df[col_ground], errors="coerce"),
        "ContextRelevance": pd.to_numeric(df[col_ctx], errors="coerce"),
        "AnswerRelevancy": pd.to_numeric(df[col_resp_rel], errors="coerce"),
    }

    means = {k: float(v.replace([np.inf, -np.inf], np.nan).dropna().mean()) for k, v in vals.items()}

    plt.figure(figsize=(9, 5))
    names = list(means.keys())
    scores = list(means.values())

    plt.bar(names, scores)
    plt.ylim(0, 1)
    plt.ylabel("Average Score (0 to 1)")
    plt.title("RAGAS — Average of 4 Metrics")

    for i, v in enumerate(scores):
        if np.isfinite(v):
            plt.text(i, min(v + 0.02, 0.98), f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved bar graph PNG to:\n{OUT_PNG}")

if __name__ == "__main__":
    main()