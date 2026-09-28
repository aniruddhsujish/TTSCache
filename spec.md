# Spec — TTS Response Caching PoC

## 1. Scope

The PoC implements the caching decision logic between an agent's text output and a TTS call, and a harness that
compares caching strategies on the same synthetic traffic. TTS sits behind one interface; the default backend is a
stub. Not built (described in `ARCHITECTURE.md`): streaming, a real provider integration, the offline
alias-map pipeline, shadow mode, and the per-language template switch.

### Assumptions
- Target provider is ElevenLabs; the design is provider-agnostic.
- Agent text is fixed input: the cache cannot change wording or get variable slots marked by the agent.
- "Standard audio caching" = exact match of the whole response text → stored audio (the baseline).

### Guiding principle
Never play words the agent didn't say. A wrong match is worse than a missed one; when unsure, synthesize.
The only exception is the gated semantic strategy (§5.3), which may play a same-meaning paraphrase.

## 2. Strategies

Each strategy is a class; each differs from the one before it in one clearly named step.

| Strategy | Class | Differs by | Role |
|---|---|---|---|
| Baseline | `BaselineStrategy` | `_split_into_cache_units` → whole response | Standard audio caching |
| Segment-level | `SegmentStrategy` | (defaults) sentence = cache unit | Main technique |
| Template | `TemplateStrategy` | `_resolve_sentence` → exact, then template | Reuse fixed parts; values as slots |
| Semantic template | `SemanticTemplateStrategy` | `_resolve_sentence` → adds a gated semantic match before synthesis | Evaluated, gated; optional install |

Admission (distinct-user counter), request coalescing, the quality gate, tiered storage and metrics are shared by
all strategies.

## 3. Modules

```
tts_cache/
  normalize.py            layer 1 (NFC, whitespace) + layer 2 (per-language spoken-form rules); rules cached
  splitter.py             word-based sentence splitting from per-language terminators/abbreviations
  lang/en.json, hi.json   per-language rules: currency, terminators, abbreviations
  keys.py                 VoiceProfile (frozen) + build_key
  counter.py              RequestCounter: distinct HMAC'd users per HMAC'd key, 1-week window, capped, sweep
  storage.py              CacheEntry, MemoryTier (LRU + TTL), FileTier (PCM + JSON, TTL, sweep), TieredStorage
  coalescing.py           Coalescer (in-flight futures, timeout, LeaderCancelled)
  quality.py              passes_quality (non-empty, plausible duration)
  metrics.py              Outcome enum, Metrics (outcomes, chars requested/saved)
  semantic.py             SemanticMatcher (embedding + NLI; lazy imports; optional install)
  pipeline.py             CachePipeline: entry point with fallback
  tts/base.py             TTSBackend interface, TTSResult, WordTimestamp, TTSError
  tts/fake.py             FakeTTS: fake audio + timestamps, delay, failure rate, cancellable
  strategies/baseline.py, segment.py, template.py, semantic.py
harness/
  generate_dataset.py     synthetic traffic with a meaning label per sentence → data/traffic.jsonl
  compare.py              runs every strategy × threshold, grades semantic matches → tables, results.json, chart
experiments/
  semantic_eval.py        grades similarity + NLI against labeled pairs
  semantic_pairs.jsonl    416 labeled sentence pairs
data/traffic.jsonl        committed dataset used for all results: 13,000 requests (en, hi, kn), meaning-labeled
tests/
```

## 4. Shared components

### 4.1 Normalization
1. Layer 1 (every language): Unicode NFC; collapse runs of whitespace to one space; trim.
2. Layer 2 (only if `lang/<code>.json` exists): currency variants (`₹`, `Rs.`, `Rs`, `रु`, `INR`, …) followed by a
   number → `<number> <canonical word>`; variants matched literally (escaped), longest first.
3. Never changed: letter case, punctuation, zero-width joiners.
4. Normalization runs on the whole response **before** splitting.

### 4.2 Sentence splitting
- Split on whitespace; a word ending in a terminator ends a sentence unless it is a listed abbreviation.
- Defaults for languages without rules: terminators `. ? ! ।`, no abbreviations.
- Leftover words without a terminator form a final sentence; empty text → no sentences.

### 4.3 Cache key
- `build_key(text, profile)` = SHA-256 hex of `json.dumps({text, namespace, language, voice, model, output_format,
  settings}, sort_keys=True, ensure_ascii=False)`.
- `VoiceProfile(language, voice, model, output_format, settings={}, namespace="shared")`, frozen.
- The TTS interface is `synthesize(text, profile)`: the key and the TTS call use the same inputs.
- Template entries use `build_key("[template] " + template_text, profile)` so they never collide with sentences.

### 4.4 Admission (counter)
- `record(key, user_id)` on arrival of a unit, before lookup; `should_admit(key)` after synthesis.
  Not recorded: an exact hit on a sentence with slots (§5.2 step 2) and a semantic hit (§5.3); both are served
  without being counted toward their own admission.
- Admit when **≥ 5 distinct users** (configurable) were seen within **7 days** (configurable).
- Stores only `HMAC-SHA256(secret, key)` and `HMAC-SHA256(secret, user_id)`.
- Per key, at most `threshold` users are tracked; old visits are pruned on access; `prune_all()` sweeps all keys.

### 4.5 Storage
- `TieredStorage.get(key) -> TTSResult | None`, `put(key, TTSResult)`. Callers never see `CacheEntry`.
- Memory tier: `OrderedDict`, LRU eviction at `max_items` (default 500).
- File tier: `<key>.pcm` (raw audio) + `<key>.json` (sample rate, word timestamps, `created_at`); audio written
  first, metadata last; an entry without metadata is a miss.
- TTL: **expire-after-write, 30 days**, enforced on both tiers; expired file entries are deleted on read;
  `prune_expired()` sweeps files never read again.
- A file-tier hit is promoted to memory **with its original `created_at`**.
- One clock stamps each entry once (`TieredStorage.put`); tiers take an injectable clock.
- Any storage exception → logged, treated as a miss (get) or ignored (put).

### 4.6 Quality gate
- Store only if audio is non-empty and `2 ≤ characters / seconds ≤ 40`.
- Gates storage only; the current request's audio is returned either way.

### 4.7 Coalescing
- `Coalescer.run(key, make_call)`: the first caller for a key is the leader and registers a future; later callers wait
  on `shield(future)` with `wait_timeout` (default 1 s).
- Waiter timeout or `LeaderCancelled` → the waiter calls `make_call()` itself.
- Leader failure with any other exception (e.g. `TTSError`) → the same exception reaches waiters.
- The in-flight entry is always removed (`finally`).

### 4.8 Metrics
- Per unit, one `Outcome`: `hit`, `semantic_hit`, `miss_stored`, `miss_below_threshold`, `miss_quality_rejected`,
  with its character count.
- `savings_ratio = chars_saved / chars_requested`; `hit` and `semantic_hit` count as saved.
- Template and semantic hits record only the fixed characters; each slot value records its own outcome, so
  characters requested equal the sentence length. Sentence-level strategies therefore request the same total;
  the baseline requests ~1.3% more, because a whole response also includes the spaces between its sentences.

### 4.9 Fallback (pipeline)
- `CachePipeline.speak(text, profile, user_id)` calls the strategy.
- `TTSError` → re-raised (not hidden, not retried).
- Any other exception → log, then plain synthesis of the original text.

## 5. Request paths

### 5.1 Segment-level (and baseline)
1. Normalize the full response; split into units (sentences; baseline: the whole response).
2. Per unit, `_get_or_synthesize(unit)`:
   1. Build the key; `counter.record(key, user_id)`.
   2. `storage.get(key)` → hit: record `hit`, return.
   3. Miss: `coalescer.run(key, lambda: tts.synthesize(unit, profile))`.
   4. Not admitted → `miss_below_threshold`; quality fails → `miss_quality_rejected`;
      else `storage.put`, call the `_on_stored(unit, profile)` hook, record `miss_stored`.
3. Return results in the original order.
4. A cancelled synthesis raises before `storage.put`, so incomplete audio is never stored.

### 5.2 Template
Per sentence:
1. **Detect slots** word by word (trailing punctuation ignored): `DATE` (`d/m/y`, `y-m-d`, `d.m.y`),
   `TIME` (`h:mm`), `NUM` (`4521`, `45,230.50`), checked in that order.
   None, or more than `MAX_VARIABLES` (2) → segment-level path.
2. **Exact sentence cached** → return it (one clip, no seams); record `hit`.
3. **Template text** = words with slots replaced by `{TYPE}` + the slot's trailing punctuation; values keep their
   punctuation. `counter.record(template_key, user_id)`.
4. **Template hit** (all `len(slots)+1` fixed parts present — all or nothing): record `hit` for fixed characters;
   get each value through `_get_or_synthesize`; join `fixed0, var0, fixed1, …` shifting timestamps.
5. **Template miss**: `_get_or_synthesize(sentence)`; if the template is admitted and the audio passes quality,
   slice the fixed parts with word timestamps (`byte = round(seconds × rate) × 2`) and store them as
   `<template_key>:<i>`; call `_on_stored(template_text, profile)`.

### 5.3 Semantic template
Before falling through to §5.2, when the exact sentence and its template would both miss:
1. Skip if the sentence has more than `MAX_VARIABLES` slot words, or the same slot type twice (values are placed
   by position, and NLI cannot tell two identical placeholders apart, so their roles could swap).
2. Query = template text (or the sentence if it has no slots).
3. Candidates = indexed texts for this profile with the same slot sequence, excluding the query itself.
   The index holds only texts stored via `_on_stored` that contain no raw slot values (never bare values like `4521.`).
4. `matcher.best_match`: top candidate by cosine similarity; accept if similarity ≥ **0.85** and NLI entailment
   probability ≥ **0.95 in both directions**. The second direction is checked only if the first passes (same
   decision, half the NLI passes). Models run on a GPU (CUDA or Apple MPS) when available, else CPU.
5. Accepted: serve the matched sentence, or the matched template's fixed parts + this request's values; record
   `semantic_hit`; log `(query, matched)` for review. Missing parts (expired/evicted) → normal path.
6. Each value is spoken with the trailing punctuation of the slot it fills in the **matched** wording, not the
   query's (a sentence-final `7788.` placed mid-sentence becomes `7788`).

## 6. Semantic evaluation (experiment)
- Pairs file: `{a, b, same, category, language}`; pairs and labels AI-generated, then reviewed by me.
- Report, for similarity thresholds 0.80–0.95, with and without NLI: safe hits, wrong hits, missed paraphrases;
  per-category matches at 0.85; NLI confidence sweep; wrong pairs by confidence; latency.
- Decision rule: a setting qualifies only with **0 wrong hits and meaningfully > 0 safe hits**.
- Result on the pairs: qualifies at confidence ≥ 0.95 (57/175 safe, 0/241 wrong). Cutoff chosen on the same set.
- Result on the traffic (§7, threshold 5): 642 semantic hits from 17 distinct pairs; 2 pairs were wrong by label,
  **17 wrong hits** ("delivered" served as "shipped" in Hindi; "on {DATE}" served as "by {DATE}"), for +0.1 points
  of savings. The pairs-set result did not hold on unseen traffic, so semantic matching is not recommended live.

## 7. Harness
- Dataset (`python -m harness.generate_dataset`, fixed seed): 13,000 requests from 2,000 users. 30% are canned
  replies sent word for word (whole-response repeats); the rest are composed. Several wordings
  per intent, near-miss intents side by side, alphanumeric order IDs, varied currency and date formats, customer
  names, free-form sentences. ~59% English, ~35% Hindi, ~6% Kannada (no rules file, to exercise the
  language-agnostic defaults). Each request carries one meaning label per sentence.
- Loads `data/traffic.jsonl`; for each strategy and threshold `[1, 2, 5, 10, 20]`, runs all requests through a fresh
  cache with `FakeTTS(delay=0)`; headline at threshold 5. A voice profile is built for any language in the traffic.
- Grading: every semantic `(query, matched)` pair is checked against the meaning labels (`ok` / `WRONG` / `?` if
  unlabeled); `semantic_wrong` counts wrong hits per run.
- Outputs: progress per run, tables, `results/results.json` (including graded semantic matches),
  `results/savings_vs_threshold.png`.
- The semantic strategy runs only if its models are installed (`requirements-semantic.txt`); one matcher is shared
  across runs.

## 8. Acceptance criteria

### Normalization
1. `"Your  order ."` → spaces collapsed.
2. `"क़"` and `"क़"` → identical output.
3. `"US"` unchanged; `"."` vs `"!"` stay different.
4. A zero-width joiner survives normalization.
5. Hindi: `"₹500"`, `"Rs. 500"`, `"रु 500"` → `"500 रुपये"`; `"500 रुपये"` unchanged.
6. A language without a rules file → layer 1 only, no error.
7. `"Pay Rs. 500."` → `"Pay 500 रुपये."` (the sentence-final dot survives).

### Splitting
8. Two sentences split correctly; `"Dr. Sharma. He is free."` splits only after `Sharma.`.
9. `।` ends a Hindi sentence; a space before `।` is handled.
10. No terminator → one sentence; empty text → `[]`; a language without rules uses defaults.

### Keys
11. Same text + profile → same key; different text → different key.
12. Changing any one of language, voice, model, output format, settings, namespace → different key.
13. Hindi vs Marathi, same text → different keys; settings order does not matter.

### TTS stub
14. Longer text → longer audio; timestamps per word; different text → different audio.
15. `calls` counts calls; `failure_rate=1.0` raises `TTSError`; a running call can be cancelled.

### Counter / admission
16. Admits after 5 distinct users, not before; one user repeating 10 times is never enough.
17. Visits older than the window don't count; `prune_all()` removes forgotten keys.
18. Raw keys and user IDs never appear in the counter's state.

### Storage
19. Put/get round trip (including Hindi word timestamps); unknown key → `None`.
20. Memory tier evicts the least recently used entry.
21. Entries expire after 30 days on both tiers; expired files are deleted; `prune_expired()` removes unread files.
22. After a simulated restart, a file-tier entry is served and promoted; a promoted entry keeps its original age
    (written day 0, promoted day 25, expired after day 30).
23. A failing tier → `get` returns `None`, `put` does not raise.

### Quality
24. Normal audio passes; empty, cut-off (too fast) and runaway (too slow) audio fail.
25. With admission allowing it, empty audio is still never cached.

### Coalescing
26. 50 concurrent misses → 1 TTS call; everyone receives the same audio; in-flight map empty afterwards.
27. A slow leader → waiters time out and synthesize themselves.
28. A TTS failure reaches every waiter without retries.
29. A cancelled leader → the waiter still gets audio.

### Segment / baseline / pipeline / metrics
30. The second request for a sentence (threshold 1) is a hit with no TTS call.
31. A response with a cached and a new sentence → only the new one is synthesized; order is preserved.
32. With threshold 5, the sixth distinct user gets a hit.
33. Two requests for one sentence (threshold 1) → one `miss_stored`, one `hit`, savings ratio 0.5.
34. When one sentence of a two-sentence response changes, the baseline saves 0% and segment-level saves > 0%.
35. A cache-path error → plain synthesis of the original text; a `TTSError` is re-raised with no extra TTS call.

### Template
36. Slot detection: one or two slots found at the right positions; three → `None`; no numbers → `None`; Hindi works.
37. Templates: `{NUM}`, `{DATE}`, `{TIME}` placeholders keep trailing punctuation; values keep theirs;
    `"12 March"` → `{NUM} March` (known limitation).
38. Slicing keeps only fixed words, with timestamps shifted to start at 0.
39. Slice then join with the original value reproduces the original audio exactly.
40. A new number → only the number is synthesized; a repeated number → no TTS call at all.
41. A sentence with three numbers never uses a template.

### Semantic (fake matcher, no models)
42. A paraphrase of a cached sentence is served with no TTS call and recorded as `semantic_hit`.
43. A semantic template match plays the matched wording with this request's number.
44. An exact hit never consults the matcher.
45. Bare numeric values are never added to the index.
46. A value takes the punctuation of its slot in the matched wording (mid-sentence → none; sentence-final → `.`);
    characters requested still equal the sentence length.
47. A sentence with the same slot type twice never gets a semantic hit, even when the matcher would accept it.
