"""Run every strategy over the same traffic and compare TTS savings

Usage:
    python -m harness.compare
    python -m harness.compare --data data/traffic.jsonl
"""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from harness.generate_dataset import DEFAULT_OUTPUT, Request, load
from tts_cache.coalescing import Coalescer
from tts_cache.counter import RequestCounter
from tts_cache.keys import VoiceProfile
from tts_cache.metrics import Metrics, Outcome
from tts_cache.storage import FileTier, MemoryTier, TieredStorage
from tts_cache.strategies.baseline import BaselineStrategy
from tts_cache.strategies.segment import SegmentStrategy
from tts_cache.tts.fake import FakeTTS

STRATEGIES = {"baseline": BaselineStrategy, "segment": SegmentStrategy}
THRESHOLDS = [1, 2, 5, 10, 20]
HEADLINE_THRESHOLD = 5

PROFILES = {
    "en": VoiceProfile(
        language="en", voice="v1", model="m1", output_format="pcm_16000"
    ),
    "hi": VoiceProfile(
        language="hi", voice="v1", model="m1", output_format="pcm_16000"
    ),
}


async def run_one_strategy(name: str, requests: list[Request], threshold: int) -> dict:
    """Run one strategy over the dataset with a fresh cache."""
    with tempfile.TemporaryDirectory() as folder:
        tts = FakeTTS(delay=0)
        metrics = Metrics()
        strategy = STRATEGIES[name](
            tts,
            TieredStorage(MemoryTier(), FileTier(root=folder)),
            RequestCounter(secret=b"harness", threshold=threshold),
            Coalescer(),
            metrics,
        )
        for r in requests:
            await strategy.get_audio(r.text, PROFILES[r.language], r.user_id)

    return {
        "strategy": name,
        "threshold": threshold,
        "tts_calls": tts.calls,
        "chars_requested": metrics.chars_requested,
        "chars_saved": metrics.chars_saved,
        "savings_pct": round(100 * metrics.savings_ratio(), 1),
        "outcomes": {o.value: metrics.outcomes[o] for o in Outcome},
    }


def print_table(rows: list[dict]) -> None:
    header = f"{'strategy':<10}{'threshold':>10}{'tts calls':>11}{'chars req':>11}{'chars saved':>13}{'savings':>9}"

    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['strategy']:<10}{r['threshold']:>10}{r['tts_calls']:>11}"
            f"{r['chars_requested']:>11}{r['chars_saved']:>13}{r['savings_pct']:>8}%"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Compare caching strategies.")
    parser.add_argument("--data", default=DEFAULT_OUTPUT, help="dataset file path")
    args = parser.parse_args()

    requests = load(args.data)
    print(f"Loaded {len(requests)} requests from {args.data}\n")

    rows = []
    for name in STRATEGIES:
        for threshold in THRESHOLDS:
            rows.append(await run_one_strategy(name, requests, threshold))

    print(f"Headline (threshold = {HEADLINE_THRESHOLD})")
    print_table([r for r in rows if r["threshold"] == HEADLINE_THRESHOLD])

    print("\nThreshold sweep:")
    print_table(rows)

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
