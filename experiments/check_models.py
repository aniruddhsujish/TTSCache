"""Load both models once and run one example through each."""

import time

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

embedder = SentenceTransformer(EMBED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)


def similarity(a: str, b: str) -> float:
    vectors = embedder.encode([a, b], convert_to_tensor=True)
    return util.cos_sim(vectors[0], vectors[1]).item()


def nli_label(premise: str, hypothesis: str) -> str:
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = nli(**inputs).logits
    return nli.config.id2label[logits.argmax().item()]


pairs = [
    ("Your request has been received.", "We have received your request."),
    ("The payment was successful.", "The payment was not successful."),
    ("भुगतान सफल रहा।", "भुगतान सफल नहीं रहा।"),
]

for a, b in pairs:
    t0 = time.perf_counter()
    sim = similarity(a, b)
    forward, backward = nli_label(a, b), nli_label(b, a)
    ms = (time.perf_counter() - t0) * 1000
    print(f"{sim:.3f}  {forward:<13} {backward:<13} {ms:6.1f} ms   {a} | {b}")
