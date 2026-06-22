"""
Inspection script for the text-decoding reproduction.

Run this FIRST on the server. It does NOT train anything and is lightweight
(uses memory-mapped numpy so it will not load the big arrays into RAM).

It reports:
  - python / torch / torchaudio versions + CUDA availability
  - the contents of the data directory and its README
  - shapes / dtypes of the neural arrays (train + realtime test)
  - the label dataframes (columns, dtypes, a few example rows)
  - the provided csv files (heads)
  - whether the KenLM language-model binaries exist anywhere
  - whether the packages we need for training import cleanly

Usage:
    python repro/inspect_data.py \
        --data_dir /userdata/dmoses/b3_features/zenodo \
        --repo_dir /home/zli/b3paper
"""
import argparse
import os
import sys
import glob


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/userdata/dmoses/b3_features/zenodo",
                   help="folder with train_data.npy etc.")
    p.add_argument("--repo_dir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   help="root of the cloned repo (the one that contains the text/ folder)")
    args = p.parse_args()

    hr("ENVIRONMENT")
    print("python     :", sys.version.replace("\n", " "))
    print("executable :", sys.executable)
    try:
        import numpy as np
        print("numpy      :", np.__version__)
    except Exception as e:
        print("numpy import FAILED:", e)
        np = None
    try:
        import torch
        print("torch      :", torch.__version__)
        print("cuda avail :", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("cuda dev   :", torch.cuda.get_device_name(0))
            print("cuda ver   :", torch.version.cuda)
    except Exception as e:
        print("torch import FAILED:", e)
    try:
        import torchaudio
        print("torchaudio :", torchaudio.__version__)
        from torchaudio.models.decoder import ctc_decoder  # noqa: F401
        print("ctc_decoder: importable (flashlight text available)")
    except Exception as e:
        print("torchaudio / ctc_decoder import issue:", repr(e))

    hr("DATA DIRECTORY LISTING: " + args.data_dir)
    if os.path.isdir(args.data_dir):
        for f in sorted(os.listdir(args.data_dir)):
            full = os.path.join(args.data_dir, f)
            try:
                size = os.path.getsize(full) / 1e6
                print(f"  {f:35s} {size:12.2f} MB")
            except OSError:
                print(f"  {f}")
    else:
        print("  !! data_dir does not exist")

    readme = os.path.join(args.data_dir, "README.md")
    if os.path.isfile(readme):
        hr("DATA README.md (first 200 lines)")
        with open(readme, "r", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= 200:
                    print("  ... (truncated)")
                    break
                print(line.rstrip())

    hr("NEURAL ARRAYS (memory-mapped, shapes only)")
    for name in ["train_data.npy", "realtime_test_data.npy"]:
        path = os.path.join(args.data_dir, name)
        if os.path.isfile(path) and np is not None:
            arr = np.load(path, mmap_mode="r")
            print(f"  {name}: shape={arr.shape} dtype={arr.dtype}")
        else:
            print(f"  {name}: MISSING")

    hr("LABEL DATAFRAMES")
    try:
        import pandas as pd
        for name in ["training_labels.h5", "realtime_test_labels.h5"]:
            path = os.path.join(args.data_dir, name)
            if not os.path.isfile(path):
                print(f"  {name}: MISSING")
                continue
            df = pd.read_hdf(path)
            print(f"\n  --- {name} ---")
            print("  n_rows  :", len(df))
            print("  columns :", list(df.columns))
            print("  dtypes  :")
            for c in df.columns:
                print(f"      {c}: {df[c].dtype}")
            # show a couple example rows in a robust way
            for idx in range(min(2, len(df))):
                print(f"  example row {idx}:")
                for c in df.columns:
                    val = df.iloc[idx][c]
                    sval = str(val)
                    if len(sval) > 200:
                        sval = sval[:200] + " ...(truncated)"
                    print(f"      {c} = {sval}")
            if "length" in df.columns:
                lens = df["length"].values
                print("  length stats: min=%s max=%s mean=%.1f" % (lens.min(), lens.max(), float(lens.mean())))
            if "txt_label" in df.columns:
                vocab = set()
                for s in df["txt_label"].values:
                    vocab.update(str(s).split(" "))
                print("  unique words in this split:", len(vocab))
    except Exception as e:
        print("  label inspection FAILED:", repr(e))

    hr("CSV FILES (head)")
    try:
        import pandas as pd
        for name in ["tm1k_blocks_and_splits.csv", "results_and_error_rates.csv"]:
            path = os.path.join(args.data_dir, name)
            if not os.path.isfile(path):
                print(f"  {name}: MISSING")
                continue
            df = pd.read_csv(path)
            print(f"\n  --- {name} ---  shape={df.shape}")
            print("  columns:", list(df.columns))
            with pd.option_context("display.max_columns", None, "display.width", 200):
                print(df.head(5).to_string())
    except Exception as e:
        print("  csv inspection FAILED:", repr(e))

    hr("LANGUAGE MODEL SEARCH (.binary / .arpa)")
    search_roots = [args.data_dir, args.repo_dir, os.path.expanduser("~")]
    found = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for ext in ("*.binary", "*.arpa", "*.bin"):
            found.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    found = sorted(set(found))
    if found:
        for f in found:
            print("  found:", f)
    else:
        print("  No .binary/.arpa language models found.")
        print("  --> The notebook expects:")
        print("        text/custom_lms/full_corpus_lm_3_abs_slm.binary  (training beam search)")
        print("        text/custom_lms/full_corpus_lm_5_abs_slm.binary  (final eval beam search)")
        print("  These are NOT in the repo or data. We will need to build them (build_lm.py).")

    hr("OPTIONAL TRAINING DEPENDENCIES")
    for mod in ["speechbrain", "kenlm", "wandb", "h5py", "tables", "sklearn"]:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  {mod:12s}: OK ({ver})")
        except Exception as e:
            print(f"  {mod:12s}: NOT available ({e.__class__.__name__})")

    hr("DONE")


if __name__ == "__main__":
    main()
