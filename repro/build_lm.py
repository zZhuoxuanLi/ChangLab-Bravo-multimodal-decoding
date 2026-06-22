"""
Build the n-gram language models the SAME WAY the paper did.

From the paper Methods ("Beam-search algorithm" / "Sentence sets"):

  "We used a custom-trained 5-gram language model with Kneser-Ney smoothing.
   We used the KenLM software package to train the 5-gram language model on the
   full 18,284 sentences that were eligible to be in the 1024-word-General set
   before any pruning ... we first extracted sentences from the nltk Twitter
   corpus and the Cornell film corpus ... sentences ... composed entirely from
   the [vocabulary] ... 4 to 8 words in length."

So the language model is NOT trained on the participant's trial sentences; it is
trained on the broad Twitter + Cornell-film corpus, filtered to:
  * sentences whose every word is in the decoding vocabulary, and
  * sentences 4-8 words long.
The 249 realtime test sentences are drawn from this same pool, so they are
naturally present in the LM corpus (exactly as in the paper).

This script reconstructs that corpus and trains a 3-gram and 5-gram KenLM model
(the notebook uses the 3-gram for fast validation during training and the
5-gram for the final evaluation). torchaudio's ctc_decoder reads the .arpa
output directly, so building the .binary is optional.

Caveat vs. the paper: the paper filtered to a 1,152-word pre-pruning vocabulary
that we do not have, so by default we filter to the 1,024-word vocabulary we can
derive from the released labels. Pass --vocab_file if you obtain the original
1,152-word list. Corpus size will therefore differ slightly from 18,284.

Requirements (install on the login node `pia`, which has internet):
    pip install nltk
    conda install -c conda-forge kenlm      # provides lmplz / build_binary
    python -c "import nltk; nltk.download('twitter_samples')"

Usage:
    python repro/build_lm.py \
        --data_dir /userdata/dmoses/b3_features/zenodo \
        --out_dir  text/custom_lms \
        --cornell_path /userdata/zli/cornell_movie_dialogs_corpus/movie_lines.txt
"""
import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

import pandas as pd

CORNELL_URL = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
_clean_re = re.compile(r"[^a-z ]+")


def clean(text):
    """Lowercase, drop apostrophes, keep only a-z, collapse whitespace ->
    list of word tokens. Matches the simple style of the released labels."""
    text = text.lower().replace("'", "")
    text = _clean_re.sub(" ", text)
    return text.split()


def get_vocab(data_dir, vocab_file):
    if vocab_file:
        with open(vocab_file) as f:
            words = {w.strip().lower() for w in f if w.strip()}
        print(f"vocab: {len(words)} words from {vocab_file}", flush=True)
        return words
    words = set()
    for name in ["training_labels.h5", "realtime_test_labels.h5"]:
        df = pd.read_hdf(os.path.join(data_dir, name))
        for s in df["txt_label"].values:
            words.update(str(s).lower().split())
    print(f"vocab: {len(words)} words derived from the released labels", flush=True)
    return words


def get_twitter_sentences():
    try:
        import nltk
        try:
            from nltk.corpus import twitter_samples
            twitter_samples.fileids()
        except LookupError:
            print("downloading nltk twitter_samples ...", flush=True)
            nltk.download("twitter_samples", quiet=True)
            from nltk.corpus import twitter_samples
        sents = list(twitter_samples.strings())
        print(f"twitter: {len(sents)} raw tweets", flush=True)
        return sents
    except Exception as e:
        print(f"!! could not load nltk twitter corpus: {e!r}", flush=True)
        return []


def get_cornell_sentences(cornell_path, out_dir):
    """Return raw utterance strings from the Cornell Movie-Dialogs corpus.
    cornell_path may point to movie_lines.txt or its directory; if missing we
    try to download the corpus zip into out_dir."""
    candidates = []
    if cornell_path:
        if os.path.isdir(cornell_path):
            candidates.append(os.path.join(cornell_path, "movie_lines.txt"))
        else:
            candidates.append(cornell_path)
    candidates.append(os.path.join(out_dir, "cornell movie-dialogs corpus", "movie_lines.txt"))

    lines_file = next((c for c in candidates if os.path.isfile(c)), None)
    if lines_file is None:
        try:
            print(f"downloading Cornell corpus from {CORNELL_URL} ...", flush=True)
            with urllib.request.urlopen(CORNELL_URL, timeout=120) as resp:
                data = resp.read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                z.extractall(out_dir)
            cand = os.path.join(out_dir, "cornell movie-dialogs corpus", "movie_lines.txt")
            lines_file = cand if os.path.isfile(cand) else None
        except Exception as e:
            print(f"!! could not download Cornell corpus: {e!r}", flush=True)
            return []
    if lines_file is None:
        print("!! Cornell movie_lines.txt not found; skipping.", flush=True)
        return []

    sents = []
    with open(lines_file, encoding="latin-1") as f:
        for line in f:
            parts = line.split(" +++$+++ ")
            if parts:
                sents.append(parts[-1])
    print(f"cornell: {len(sents)} raw utterances from {lines_file}", flush=True)
    return sents


def filter_corpus(raw_sentences, vocab, min_w=4, max_w=8):
    kept = []
    for s in raw_sentences:
        toks = clean(s)
        if min_w <= len(toks) <= max_w and all(t in vocab for t in toks):
            kept.append(" ".join(toks))
    return kept


def build_arpa(corpus_path, arpa_path, order):
    if shutil.which("lmplz") is None:
        sys.exit("ERROR: 'lmplz' not found on PATH. Install KenLM "
                 "(conda install -c conda-forge kenlm).")
    # KenLM lmplz uses modified Kneser-Ney by default (matches the paper).
    # --discount_fallback is needed for small corpora at higher orders.
    cmd = ["lmplz", "-o", str(order), "--discount_fallback",
           "--text", corpus_path, "--arpa", arpa_path]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print("wrote", arpa_path, flush=True)


def build_binary(arpa_path, binary_path):
    if shutil.which("build_binary") is None:
        print("note: 'build_binary' not found; skipping .binary "
              "(the .arpa works directly with torchaudio).", flush=True)
        return
    cmd = ["build_binary", arpa_path, binary_path]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print("wrote", binary_path, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/userdata/dmoses/b3_features/zenodo")
    p.add_argument("--out_dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "text", "custom_lms"))
    p.add_argument("--vocab_file", default=None,
                   help="optional word-per-line vocab (e.g. the original 1,152-word list)")
    p.add_argument("--cornell_path", default=None,
                   help="path to movie_lines.txt or the Cornell corpus dir (auto-downloads if omitted)")
    p.add_argument("--orders", default="3,5", help="comma-separated n-gram orders")
    p.add_argument("--min_words", type=int, default=4)
    p.add_argument("--max_words", type=int, default=8)
    p.add_argument("--add_data_sentences", action="store_true", default=True,
                   help="also include the released trial sentences (they are part of the eligible pool)")
    p.add_argument("--no_add_data_sentences", dest="add_data_sentences", action="store_false")
    p.add_argument("--make_binary", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vocab = get_vocab(args.data_dir, args.vocab_file)

    raw = []
    raw += get_twitter_sentences()
    raw += get_cornell_sentences(args.cornell_path, args.out_dir)
    corpus = filter_corpus(raw, vocab, args.min_words, args.max_words)
    print(f"in-vocab {args.min_words}-{args.max_words}-word sentences from corpora: {len(corpus)}", flush=True)

    if args.add_data_sentences:
        for name in ["training_labels.h5", "realtime_test_labels.h5"]:
            df = pd.read_hdf(os.path.join(args.data_dir, name))
            for s in df["txt_label"].values:
                toks = clean(str(s))
                if args.min_words <= len(toks) <= args.max_words and all(t in vocab for t in toks):
                    corpus.append(" ".join(toks))

    # de-duplicate while preserving order
    seen, deduped = set(), []
    for s in corpus:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    corpus = deduped
    print(f"final corpus: {len(corpus)} unique sentences "
          f"(paper used 18,284 eligible sentences)", flush=True)

    if len(corpus) < 1000:
        print("!! WARNING: corpus is very small. Check that the Twitter/Cornell "
              "corpora loaded and that the vocabulary is correct.", flush=True)

    corpus_path = os.path.join(args.out_dir, "lm_corpus.txt")
    with open(corpus_path, "w") as f:
        f.write("\n".join(corpus) + "\n")
    print("wrote", corpus_path, flush=True)

    for order in [int(o) for o in args.orders.split(",")]:
        arpa_path = os.path.join(args.out_dir, f"corpus_lm_{order}gram.arpa")
        build_arpa(corpus_path, arpa_path, order)
        if args.make_binary:
            build_binary(arpa_path, os.path.join(args.out_dir, f"corpus_lm_{order}gram.binary"))

    print("\nDone. Point the trainer at these, e.g.:", flush=True)
    print(f"  --lm_3gram {os.path.join(args.out_dir, 'corpus_lm_3gram.arpa')}", flush=True)
    print(f"  --lm_5gram {os.path.join(args.out_dir, 'corpus_lm_5gram.arpa')}", flush=True)


if __name__ == "__main__":
    main()
