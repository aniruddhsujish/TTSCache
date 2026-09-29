import re
import unicodedata
import json
import functools
from pathlib import Path

LANG_DIR = Path(__file__).parent / "lang"


def normalize_basic(text: str) -> str:
    """Removes whitespaces and normalizes to NFC format"""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@functools.lru_cache
def load_lang_rules(language: str) -> dict | None:
    """Loads the rules for a particular language from its config file"""
    path = LANG_DIR / f"{language}.json"
    if path.exists():
        rules = json.loads(path.read_text(encoding="utf-8"))
        return rules
    return None


def build_currency_pattern(rules: dict) -> str:
    """Builds a regex pattern for identifying currency patterns"""
    variants = rules["currency"]["variants"]
    variants = sorted(variants, key=len, reverse=True)
    escaped = [re.escape(v) for v in variants]
    alternatives = "|".join(escaped)
    return rf"(?<!\w)(?:{alternatives})\s*([\d,]+(?:\.\d+)?)"


def apply_lang_rules(text: str, rules: dict) -> str:
    """Applies language specific rules (normalizing currency format)

    Potentially could do more on a case by case basis for languages
    """
    pattern = build_currency_pattern(rules)
    canonical_currency_word = rules["currency"]["canonical"]
    replacement = rf"\1 {canonical_currency_word}"
    return re.sub(pattern, replacement, text)


def normalize(text: str, language: str) -> str:
    """Applies basic normalization and language specific normalization"""
    text = normalize_basic(text)
    rules = load_lang_rules(language)
    if rules:
        text = apply_lang_rules(text, rules)
    return text
