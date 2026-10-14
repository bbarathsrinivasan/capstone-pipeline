"""
Train a small from-scratch tokenizer, fully offline.

Why from-scratch at all: this sandbox's network policy allows PyPI/npm/
crates/the Go proxy, but not huggingface.co or the PyTorch CDN (confirmed
via the proxy status endpoint -- both return 403). So no pretrained model
or tokenizer can be downloaded here. `tokenizers` (a pip package, no
network needed to run) lets us train a real BPE tokenizer directly on our
own corpus instead.

This is a hard infrastructure constraint, not a data problem -- flagged
clearly in results/findings.md. It means the model below is NOT a
pretrained LLM and the run is a mechanics demo, not a quality benchmark.
See that doc for what to do differently with real GPU/HF access.
"""
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "tokenizer" / "corpus.txt"
TOKENIZER_DIR = ROOT / "tokenizer"
VOCAB_SIZE = 4000

# A modest amount of generic descriptive language, on top of the actual
# prompt corpus (see build_corpus below), so the vocabulary isn't ONLY the
# literal numbers/feature names in our prompts -- gives the model at least
# a chance at descriptive words when generating summaries.
SUPPLEMENTARY_TEXT = """
This person's sleep was irregular and shorter than usual this week.
Sleep duration was consistent and within a healthy range most nights.
Screen time increased noticeably in the evening hours, especially late at night.
Phone activity dropped sharply, suggesting reduced social engagement.
Time spent at home increased substantially compared to a typical week.
Activity and mobility patterns appear stable with no notable disruption.
Several nights show missing data, so the picture for this week is incomplete.
Bedtime shifted later across the week, with more wake after sleep onset.
Overall the behavioral pattern suggests elevated risk and warrants follow-up.
Overall the behavioral pattern looks unremarkable with low apparent risk.
The data this week is sparse, limiting confidence in any assessment.
Nighttime phone use was elevated, which can disrupt sleep continuity.
Social contact, approximated by nearby devices, was lower than typical.
Sleep efficiency remained high despite a shorter overall duration.
This week shows a clear decline relative to prior patterns.
This week shows no meaningful change from a typical baseline week.
""".strip()


def build_corpus():
    import sys
    sys.path.insert(0, str(ROOT))
    from prompts import build_examples

    examples = build_examples()
    lines = [e["prompt"] for e in examples]
    lines.append(SUPPLEMENTARY_TEXT)
    CORPUS_PATH.write_text("\n".join(lines))
    print(f"Corpus: {len(lines)} prompt(s)+supplement -> {CORPUS_PATH} "
          f"({CORPUS_PATH.stat().st_size / 1024:.0f} KB)")
    return CORPUS_PATH


def train_tokenizer():
    corpus_path = build_corpus()
    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[str(corpus_path)],
        vocab_size=VOCAB_SIZE,
        min_frequency=1,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )
    TOKENIZER_DIR.mkdir(exist_ok=True)
    tok.save_model(str(TOKENIZER_DIR))
    print(f"Trained byte-level BPE tokenizer (vocab_size={VOCAB_SIZE}) -> {TOKENIZER_DIR}")
    return tok


if __name__ == "__main__":
    train_tokenizer()
