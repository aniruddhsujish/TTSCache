"""Run every strategy over the same traffic and compare TTS savings.

Usage:
    python -m harness.compare
    python -m harness.compare --data data/traffic.jsonl
"""

import argparse
import asyncio
import json
import tempfile
import time
from collections import defaultdict
from functools import cache, partial
from pathlib import Path

from harness.generate_dataset import DEFAULT_OUTPUT, Request, load
from tts_cache.normalize import normalize
from tts_cache.splitter import split_sentences
from tts_cache.strategies.template import find_variables, make_template
from tts_cache.coalescing import Coalescer
from tts_cache.counter import RequestCounter
from tts_cache.keys import VoiceProfile
from tts_cache.metrics import Metrics, Outcome
from tts_cache.semantic import SemanticMatcher
from tts_cache.storage import FileTier, MemoryTier, TieredStorage
from tts_cache.strategies.baseline import BaselineStrategy
from tts_cache.strategies.segment import SegmentStrategy
from tts_cache.strategies.semantic import SemanticTemplateStrategy
from tts_cache.strategies.template import TemplateStrategy
from tts_cache.tts.fake import FakeTTS

THRESHOLDS = [1, 2, 5, 10, 20]
HEADLINE_THRESHOLD = 5

@cache
def profile(language: str) -> VoiceProfile:
    """One voice profile per language, for any language in the traffic (configured or not)."""
    return VoiceProfile(
        language=language, voice="v1", model="m1", output_format="pcm_16000"
    )


def build_strategies() -> dict:
    """Every strategy the harness compares. Semantic is added only if its models are installed."""
    strategies = {
        "baseline": BaselineStrategy,
        "segment": SegmentStrategy,
        "template": TemplateStrategy,
    }
    try:
        matcher = SemanticMatcher(
            sim_threshold=0.85, conf_threshold=0.95
        )  # loads both models once
        strategies["semantic"] = partial(SemanticTemplateStrategy, matcher=matcher)
        print(f"(semantic models running on {matcher.device})")
    except ImportError:
        print(
            "(semantic models not installed: pip install -r requirements-semantic.txt; skipping)"
        )
    return strategies


async def run_one_strategy(
    name: str,
    make_strategy,
    requests: list[Request],
    threshold: int,
    meanings: dict[str, set[str]],
) -> dict:
    """Run one strategy over all requests with a fresh, empty cache."""
    with tempfile.TemporaryDirectory() as folder:
        tts = FakeTTS(delay=0)
        metrics = Metrics()
        strategy = make_strategy(
            tts,
            TieredStorage(MemoryTier(), FileTier(root=folder)),
            RequestCounter(secret=b"harness", threshold=threshold),
            Coalescer(),
            metrics,
        )
        for r in requests:
            await strategy.get_audio(r.text, profile(r.language), r.user_id)

    row = {
        "strategy": name,
        "threshold": threshold,
        "tts_calls": tts.calls,
        "chars_requested": metrics.chars_requested,
        "chars_saved": metrics.chars_saved,
        "savings_pct": round(100 * metrics.savings_ratio(), 1),
        "outcomes": {o.value: metrics.outcomes[o] for o in Outcome},
    }
    if hasattr(strategy, "semantic_matches"):
        row["semantic_matches"] = [
            {"query": q, "matched": m, "count": c, "correct": grade(q, m, meanings)}
            for (q, m), c in strategy.semantic_matches.most_common()
        ]
        row["semantic_wrong"] = sum(
            m["count"] for m in row["semantic_matches"] if m["correct"] is False
        )
    return row


def meaning_key(sentence: str) -> str:
    """What a sentence is cached and semantically matched as: its template if it has slots, else itself."""
    words = sentence.split()
    positions = find_variables(words)
    return make_template(words, positions)[0] if positions else sentence


def meaning_index(requests: list[Request]) -> dict[str, set[str]]:
    """Cache unit (sentence or template) → the meaning labels it was generated with."""
    index = defaultdict(set)
    for r in requests:
        if r.meanings:
            sentences = split_sentences(normalize(r.text, r.language), r.language)
            for sentence, meaning in zip(sentences, r.meanings):
                index[meaning_key(sentence)].add(meaning)
    return index


def grade(query: str, matched: str, meanings: dict[str, set[str]]) -> bool | None:
    """Was serving `matched` in place of `query` correct? None if either is unlabeled."""
    if not meanings.get(query) or not meanings.get(matched):
        return None
    return bool(meanings[query] & meanings[matched])


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'strategy':<10}{'threshold':>10}{'tts calls':>11}"
        f"{'chars req':>11}{'chars saved':>13}{'savings':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['strategy']:<10}{r['threshold']:>10}{r['tts_calls']:>11}"
            f"{r['chars_requested']:>11}{r['chars_saved']:>13}{r['savings_pct']:>8}%"
        )


def print_semantic_review(rows: list[dict]) -> None:
    """Every distinct 'served X instead of Y' decision, for a human to check."""
    for r in rows:
        if r["strategy"] == "semantic" and r["threshold"] == HEADLINE_THRESHOLD:
            matches = r["semantic_matches"]
            print(
                f"\nSemantic matches to review @ threshold {HEADLINE_THRESHOLD} "
                f"({len(matches)} distinct, {r['outcomes']['semantic_hit']} hits, "
                f"{r['semantic_wrong']} wrong by label):"
            )
            marks = {True: "ok", False: "WRONG", None: "?"}
            for m in matches:
                print(
                    f"  {marks[m['correct']]:>5} {m['count']:>4}×  {m['query']}  →  {m['matched']}"
                )


def draw_chart(rows: list[dict], names: list[str], path: Path) -> None:
    """Savings (%) against admission threshold, one line per strategy."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping chart)")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in names:
        points = [r for r in rows if r["strategy"] == name]
        ax.plot(
            [p["threshold"] for p in points],
            [p["savings_pct"] for p in points],
            marker="o",
            label=name,
        )
    ax.axvline(HEADLINE_THRESHOLD, linestyle="--", color="grey", alpha=0.6)
    ax.set_xlabel("Admission threshold (distinct users)")
    ax.set_ylabel("TTS characters saved (%)")
    ax.set_title("TTS savings by strategy and admission threshold")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to {path}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Compare caching strategies.")
    parser.add_argument("--data", default=DEFAULT_OUTPUT, help="traffic JSONL file")
    args = parser.parse_args()

    requests = load(args.data)
    print(f"Loaded {len(requests)} requests from {args.data}\n")

    meanings = meaning_index(requests)
    strategies = build_strategies()
    rows = []
    for name, make_strategy in strategies.items():
        for threshold in THRESHOLDS:
            started = time.perf_counter()
            print(f"  running {name} @ threshold {threshold} ...", end=" ", flush=True)
            rows.append(
                await run_one_strategy(
                    name, make_strategy, requests, threshold, meanings
                )
            )
            print(f"{time.perf_counter() - started:.0f}s", flush=True)
    print()

    print(f"Headline (threshold = {HEADLINE_THRESHOLD}):")
    print_table([r for r in rows if r["threshold"] == HEADLINE_THRESHOLD])

    print("\nThreshold sweep:")
    print_table(rows)

    print_semantic_review(rows)

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    draw_chart(rows, list(strategies), out / "savings_vs_threshold.png")


if __name__ == "__main__":
    asyncio.run(main())
