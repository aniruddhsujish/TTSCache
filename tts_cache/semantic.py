"""Semantic equivalence check: embedding similarity finds a candidate, NLI confitms it.

Models are imported lazily to prevent big imports (torch) till we actually need it
"""

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


class SemanticMatcher:

    def __init__(self, sim_threshold: float = 0.85, conf_threshold: float = 0.95):
        import torch
        from sentence_transformers import SentenceTransformer, util
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch, self._util = torch, util
        self.sim_threshold = sim_threshold
        self.conf_threshold = conf_threshold
        # a GPU makes NLI ~15x faster than CPU (Apple MPS: ~12 ms vs ~200 ms per pass)
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.embedder = SentenceTransformer(EMBED_MODEL, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(
            self.device
        )
        self.nli.eval()
        self._vectors: dict[str, object] = {}  # text → embedding
        self._entailment: dict[tuple[str, str], float] = (
            {}
        )  # (premise, hypothesis) → probability
        self.nli_calls = 0

    def _embed(self, text: str):
        if text not in self._vectors:
            self._vectors[text] = self.embedder.encode(text, convert_to_tensor=True)
        return self._vectors[text]

    def _entail(self, premise: str, hypothesis: str) -> float:
        if (premise, hypothesis) not in self._entailment:
            self.nli_calls += 1
            inputs = self.tokenizer(
                premise, hypothesis, return_tensors="pt", truncation=True
            ).to(self.device)
            with self._torch.no_grad():
                probs = self.nli(**inputs).logits.softmax(dim=-1)[0]
            self._entailment[(premise, hypothesis)] = probs[
                self.nli.config.label2id["entailment"]
            ].item()
        return self._entailment[(premise, hypothesis)]

    def confidence(self, a: str, b: str) -> float:
        """Checks if it's the same meaning, both directions. Weaker direction is prio"""
        return min(self._entail(a, b), self._entail(b, a))

    def best_match(
        self, query: str, candidates: list[str]
    ) -> tuple[str, float, float] | None:
        """The most similar candidate, if it passes both similarity and NLI confidence."""
        if not candidates:
            return None
        sims = self._util.cos_sim(
            self._embed(query), self._torch.stack([self._embed(c) for c in candidates])
        )[0]
        best = int(sims.argmax())
        sim = sims[best].item()
        if sim < self.sim_threshold:
            return None
        # both directions must pass; most candidates fail the first, so skip the second NLI pass
        match = candidates[best]
        if self._entail(query, match) < self.conf_threshold:
            return None
        conf = self.confidence(query, match)  # the first direction is cached
        return (match, sim, conf) if conf >= self.conf_threshold else None
