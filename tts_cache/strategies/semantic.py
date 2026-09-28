"""Semantic template caching: on a would-be miss, reuse a stored phrase that means the same thing.

Order: exact sentence -> template -> semantic match -> synthesize.

A semantic hit plays the matched (canonical) wording with THIS request's numbers/ dates/ times.
"""

from collections import Counter, defaultdict

from tts_cache.keys import VoiceProfile, build_key
from tts_cache.metrics import Outcome
from tts_cache.strategies.template import (
    TRAILING_PUNCTUATION,
    TemplateStrategy,
    find_variables,
    interleave,
    join,
    make_template,
    slot_punctuation,
    slot_type,
    slots,
)
from tts_cache.tts.base import TTSResult


class SemanticTemplateStrategy(TemplateStrategy):

    def __init__(self, *args, matcher, **kwargs):
        super().__init__(*args, **kwargs)
        self.matcher = matcher
        self.index: dict[str, list[str]] = defaultdict(
            list
        )  # profile → stored sentences/templates
        self.semantic_matches: Counter = (
            Counter()
        )  # (query, matched) → count, for review

    def _profile_id(self, profile: VoiceProfile) -> str:
        return build_key("[profile]", profile)

    def _on_stored(self, text: str, profile: VoiceProfile) -> None:
        # index only number-free sentences and templates; never bare values like "4321"
        if (
            find_variables(text.split()) is None
            and text not in self.index[self._profile_id(profile)]
        ):
            self.index[self._profile_id(profile)].append(text)

    async def _resolve_sentence(
        self, sentence: str, profile: VoiceProfile, user_id: str
    ) -> TTSResult:
        words = sentence.split()
        positions = find_variables(words)
        too_many_numbers = positions is None and any(slot_type(w) for w in words)

        if not too_many_numbers:
            query, values = (
                (sentence, []) if positions is None else make_template(words, positions)
            )
            exact_hit = self.storage.get(build_key(sentence, profile)) is not None
            template_hit = (
                bool(values)
                and self._load_parts(
                    build_key("[template] " + query, profile), len(values) + 1
                )
                is not None
            )
            if not (exact_hit or template_hit):
                served = await self._serve_semantic(
                    query, values, sentence, profile, user_id
                )
                if served is not None:
                    return served
        return await super()._resolve_sentence(sentence, profile, user_id)

    async def _serve_semantic(
        self, query, values, sentence, profile, user_id
    ) -> TTSResult | None:
        # Values are placed by position. With two slots of the same type, a paraphrase can
        # swap their roles ("balance {NUM}, due {NUM}" vs "due {NUM}, balance {NUM}") and NLI
        # can't tell, since it sees identical placeholders. Never risk a swapped number.
        query_slots = slots(query)
        if len(query_slots) != len(set(query_slots)):
            return None

        candidates = [
            t
            for t in self.index[self._profile_id(profile)]
            if t != query and slots(t) == slots(query)
        ]
        match = self.matcher.best_match(query, candidates)
        if match is None:
            return None
        matched = match[0]

        if values:
            fixed = self._load_parts(
                build_key("[template] " + matched, profile), len(values) + 1
            )
            if fixed is None:
                return None
            self.metrics.record(
                Outcome.SEMANTIC_HIT, len(sentence) - sum(len(v) for v in values)
            )
            # the slot may sit elsewhere in the matched wording: speak each value with the
            # punctuation of the slot it lands in, not the one it had in the query
            spoken = [
                v.rstrip(TRAILING_PUNCTUATION) + p
                for v, p in zip(values, slot_punctuation(matched))
            ]
            variables = [
                await self._get_or_synthesize(v, profile, user_id) for v in spoken
            ]
            result = join(interleave(fixed, variables))
        else:
            result = self.storage.get(build_key(matched, profile))
            if result is None:
                return None
            self.metrics.record(Outcome.SEMANTIC_HIT, len(sentence))

        self.semantic_matches[(query, matched)] += 1
        return result
