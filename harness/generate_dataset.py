"""Generate a synthetic TTS traffic dataset and save it as JSON Lines.

Usage:
    python -m harness.generate_dataset
    python -m harness.generate_dataset --output data/my_traffic.jsonl

Assumptions (state these in the doc):
- Responses = optional greeting (60%) + one body sentence + optional closing (60%)
- Body: 40% fixed FAQ-style, 35% with a variable (order number / amount / days), 25% long tail
- Fixed and variable bodies include paraphrases (same meaning, different words) and
  near-misses (similar words, different meaning) so semantic strategies can be evaluated
- One variable body has two variables, to exercise template caching's fallback
- 30% Hindi, 5,000 requests from 800 users
- Popularity is Zipf-skewed (s = 1.1): a few phrases are very common, most are rare
- Fixed seed, so every run produces identical traffic

This dataset measures savings. It cannot grade whether a semantic hit was correct;
that needs a separate hand-labeled set of sentence pairs.
"""

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_OUTPUT = "data/traffic.jsonl"


@dataclass
class Request:
    user_id: str
    language: str
    text: str


OPENINGS = {
    "en": [
        "Hello, thanks for calling.",
        "Hi! How can I help you today?",
        "Welcome back.",
    ],
    "hi": [
        "नमस्ते, कॉल करने के लिए धन्यवाद।",
        "मैं आपकी क्या मदद कर सकता हूँ?",
        "आपका फिर से स्वागत है।",
    ],
}

STATIC_BODIES = {
    "en": [
        "Please hold while I check that for you.",
        "Your request has been received.",
        "You can track your order in the app.",
        "The payment was successful.",
        "Our support team is available on weekdays.",
        # paraphrases of the lines above (same meaning, different words)
        "Please wait while I look into that for you.",
        "We have received your request.",
        # near-misses (similar words, different meaning)
        "The payment was not successful.",
        "Your request has been rejected.",
    ],
    "hi": [
        "कृपया प्रतीक्षा करें, मैं जाँच कर रहा हूँ।",
        "आपका अनुरोध प्राप्त हो गया है।",
        "आप ऐप में अपना ऑर्डर ट्रैक कर सकते हैं।",
        "भुगतान सफल रहा।",
        # paraphrase
        "हमें आपका अनुरोध मिल गया है।",
        # near-miss
        "भुगतान सफल नहीं रहा।",
    ],
}

VARIABLE_BODIES = {
    "en": [
        "Your order {n} has been shipped.",
        "Your refund of Rs. {amt} has been processed.",
        "Your ticket number is {n}.",
        "Your bill of ₹{amt} is due on Friday.",
        # paraphrase
        "We have shipped your order {n}.",
        # near-misses
        "Your order {n} will arrive today.",
        "Your order {n} will arrive tomorrow.",
        # two variables → template caching should fall back to segment-level
        "Your order {n} will arrive in {d} days.",
        "Your appointment is on {date}.",
        "The technician will visit at {time}.",
    ],
    "hi": [
        "आपका ऑर्डर {n} भेज दिया गया है।",
        "आपका ₹{amt} का रिफंड प्रोसेस हो गया है।",
        "आपका टिकट नंबर {n} है।",
        "आपकी अपॉइंटमेंट {date} को है।",
        # paraphrase
        "हमने आपका ऑर्डर {n} भेज दिया है।",
        # near-misses
        "आपका ऑर्डर {n} आज पहुँचेगा।",
        "आपका ऑर्डर {n} कल पहुँचेगा।",
    ],
}

CLOSINGS = {
    "en": [
        "Is there anything else I can help you with?",
        "Thank you, have a great day!",
    ],
    "hi": ["क्या मैं आपकी और कोई मदद कर सकता हूँ?", "धन्यवाद, आपका दिन शुभ हो।"],
}

# Long tail: combinations of these rarely repeat, like free-form LLM answers.
TAIL_PARTS = {
    "en": (
        [
            "your recent delivery",
            "the charge on your card",
            "your address change",
            "a damaged item",
            "your subscription plan",
            "a missing refund",
        ],
        [
            "from last week",
            "on your latest statement",
            "you mentioned earlier",
            "for the second time",
            "that you flagged today",
            "from your previous call",
        ],
        [
            "I will escalate this",
            "our team will review it",
            "I have logged a note",
            "a specialist will call you back",
            "I have updated your case",
        ],
    ),
    "hi": (
        [
            "आपकी हाल की डिलीवरी",
            "आपके कार्ड पर शुल्क",
            "आपके पते में बदलाव",
            "एक खराब सामान",
        ],
        ["पिछले हफ्ते की", "जिसका आपने पहले ज़िक्र किया", "आज आपने जो बताया"],
        [
            "मैं इसे आगे भेजूँगा",
            "हमारी टीम इसकी जाँच करेगी",
            "मैंने नोट दर्ज कर लिया है",
        ],
    ),
}

AMOUNTS = ["199", "499", "1,250", "45,230.50"]


def zipf_choice(rng: random.Random, items: list[str], s: float = 1.1) -> str:
    """Pick an item where earlier items are much more popular than later ones."""
    weights = [1 / (rank + 1) ** s for rank in range(len(items))]
    return rng.choices(items, weights=weights)[0]


def tail_sentence(rng: random.Random, language: str) -> str:
    topic, detail, action = (rng.choice(part) for part in TAIL_PARTS[language])
    if language == "hi":
        return f"{detail} {topic} के बारे में, {action}।"
    return f"Regarding {topic} {detail}, {action}."


def generate(
    n_requests: int = 5000, n_users: int = 800, hindi_share: float = 0.3, seed: int = 7
) -> list[Request]:
    rng = random.Random(seed)
    requests = []

    for _ in range(n_requests):
        language = "hi" if rng.random() < hindi_share else "en"
        user_id = f"user{rng.randrange(n_users)}"
        parts = []

        if rng.random() < 0.6:
            parts.append(zipf_choice(rng, OPENINGS[language]))

        kind = rng.random()
        if kind < 0.40:
            parts.append(zipf_choice(rng, STATIC_BODIES[language]))
        elif kind < 0.75:
            template = zipf_choice(rng, VARIABLE_BODIES[language])
            parts.append(
                template.format(
                    n=rng.randint(1000, 9999),
                    amt=rng.choice(AMOUNTS),
                    d=rng.randint(2, 7),
                    date=f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2026",
                    time=f"{rng.randint(9, 18)}:{rng.choice(['00', '15', '30', '45'])}",
                )
            )
        else:
            parts.append(tail_sentence(rng, language))

        if rng.random() < 0.6:
            parts.append(zipf_choice(rng, CLOSINGS[language]))

        requests.append(
            Request(user_id=user_id, language=language, text=" ".join(parts))
        )

    return requests


def save(requests: list[Request], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def load(path: str) -> list[Request]:
    with open(path, encoding="utf-8") as f:
        return [Request(**json.loads(line)) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic TTS traffic.")
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="where to write the JSONL file"
    )
    args = parser.parse_args()

    requests = generate()
    save(requests, args.output)
    print(f"Wrote {len(requests)} requests to {args.output}")


if __name__ == "__main__":
    main()
