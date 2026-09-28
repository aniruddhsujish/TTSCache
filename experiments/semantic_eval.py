"""Grade similarity + NLI against hand labels, at several thresholds.

Usage: python -m experiments.semantic_eval
"""

import json
import statistics
import argparse
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
PAIRS_FILE = "experiments/semantic_pairs.jsonl"
THRESHOLDS = [0.80, 0.85, 0.90, 0.95]

embedder = SentenceTransformer(EMBED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)


def similarity(a: str, b: str) -> float:
    vectors = embedder.encode([a, b], convert_to_tensor=True)
    return util.cos_sim(vectors[0], vectors[1]).item()


def nli_probs(premise: str, hypothesis: str) -> dict[str, float]:
    """Probability of each label: entailment, neutral, contradiction."""
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        probs = nli(**inputs).logits.softmax(dim=-1)[0]
    return {nli.config.id2label[i]: p.item() for i, p in enumerate(probs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=PAIRS_FILE)
    args = parser.parse_args()
    lines = Path(args.pairs).read_text(encoding="utf-8").splitlines()

    pairs = [json.loads(line) for line in lines if line.strip()]

    # warm up both models so one-time setup isn't counted in the timings
    similarity("warm up", "warm up")
    nli_probs("warm up", "warm up")

    scored, sim_ms, nli_ms = [], [], []
    for p in pairs:
        t0 = time.perf_counter()
        sim = similarity(p["a"], p["b"])
        t1 = time.perf_counter()
        forward = nli_probs(p["a"], p["b"])
        backward = nli_probs(p["b"], p["a"])
        t2 = time.perf_counter()

        same_by_nli = (
            max(forward, key=forward.get) == "entailment"
            and max(backward, key=backward.get) == "entailment"
        )
        nli_conf = min(
            forward["entailment"], backward["entailment"]
        )  # the weaker direction decides

        sim_ms.append((t1 - t0) * 1000)
        nli_ms.append((t2 - t1) * 1000)
        scored.append({**p, "sim": sim, "nli_same": same_by_nli, "nli_conf": nli_conf})

    n_same = sum(p["same"] for p in pairs)
    print(
        f"{len(pairs)} pairs  ({n_same} same meaning, {len(pairs) - n_same} different)\n"
    )

    print(
        f"{'threshold':>9}  {'guard':>5}  {'safe hits':>9}  {'wrong hits':>10}  {'missed paraphrases':>18}"
    )
    for threshold in THRESHOLDS:
        for use_nli in (False, True):
            hits = [
                s
                for s in scored
                if s["sim"] >= threshold and (s["nli_same"] or not use_nli)
            ]
            safe = sum(s["same"] for s in hits)
            wrong = sum(not s["same"] for s in hits)
            missed = n_same - safe
            print(
                f"{threshold:>9.2f}  {'NLI' if use_nli else 'none':>5}  {safe:>9}  {wrong:>10}  {missed:>18}"
            )

    print("\nNLI mistakes (verdict disagrees with label):")
    for s in sorted(scored, key=lambda s: s["sim"], reverse=True):
        if s["nli_same"] != s["same"]:
            label = "same" if s["same"] else "DIFF"
            print(
                f"  {s['sim']:.3f}  label={label}  {s['category']:<24} {s['a']} | {s['b']}"
            )

    print("\nPer category @0.85 with NLI guard (matched / total):")
    for cat in sorted({s["category"] for s in scored}):
        items = [s for s in scored if s["category"] == cat]
        hits = sum(1 for s in items if s["sim"] >= 0.85 and s["nli_same"])
        print(f"  {cat:<26} {hits:>3} / {len(items)}")
    p95 = sorted(nli_ms)[max(0, int(0.95 * len(nli_ms)) - 1)]

    print("\nSimilarity ≥ 0.85, varying NLI confidence (both directions):")
    for cutoff in [0.5, 0.7, 0.9, 0.95, 0.99]:
        hits = [s for s in scored if s["sim"] >= 0.85 and s["nli_conf"] >= cutoff]
        safe = sum(s["same"] for s in hits)
        wrong = sum(not s["same"] for s in hits)
        print(f"  conf ≥ {cutoff:.2f}: safe {safe:>3}  wrong {wrong:>2}")

    print(
        f"\nLatency per pair (ms): similarity median {statistics.median(sim_ms):.0f}, "
        f"NLI both directions median {statistics.median(nli_ms):.0f}, p95 {p95:.0f}"
    )

    print("\nDifferent-meaning pairs with similarity ≥ 0.85, by NLI confidence:")
    risky = [s for s in scored if not s["same"] and s["sim"] >= 0.85]
    for s in sorted(risky, key=lambda s: s["nli_conf"], reverse=True)[:10]:
        print(
            f"  conf {s['nli_conf']:.3f}  sim {s['sim']:.3f}  {s['category']:<18} {s['a']} | {s['b']}"
        )


if __name__ == "__main__":
    main()
