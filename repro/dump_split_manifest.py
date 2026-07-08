"""
Generate (or compare) a lightweight train/val/test split manifest.

The manifest lists which trials are in each split (index / block /
within-block index / ground truth) WITHOUT copying any neural data, so you can
verify that different runs used the same split and compare across experiments.
It only loads the small label files, so it runs in seconds (no 11.5 GB load).

Generate a reference manifest for a given fold:
    python repro/dump_split_manifest.py --eval_set 1 \
        --out repro/runs/split_manifest_evalset1.json

Compare the manifests written by several runs (checks split membership is
identical across them):
    python repro/dump_split_manifest.py --compare \
        repro/runs/bs_32/split_manifest.json \
        repro/runs/bs_512/split_manifest.json
"""
import argparse
import json
import os

import pandas as pd

# reuse the exact split logic + manifest writer from the training script
import train_text_decoder as T


def load_split_index_sets(path):
    with open(path) as f:
        m = json.load(f)
    return {k: [row["index"] for row in m[k]] for k in ("train", "val", "test")}


def compare(paths):
    ref, ref_path, ok = None, None, True
    for p in paths:
        s = load_split_index_sets(p)
        if ref is None:
            ref, ref_path = s, p
            print(f"reference: {p}")
            for k, v in s.items():
                print(f"  {k}: n={len(v)}")
            continue
        same = all(s[k] == ref[k] for k in ("train", "val", "test"))
        ok = ok and same
        print(f"{'MATCH  ' if same else 'DIFFERS'}: {p}")
        if not same:
            for k in ("train", "val", "test"):
                a, b = set(ref[k]), set(s[k])
                if a != b:
                    print(f"    {k}: ref_n={len(a)} this_n={len(b)} "
                          f"only_in_ref={len(a - b)} only_in_this={len(b - a)}")
    print("\nALL IDENTICAL" if ok else "\nSPLITS DIFFER")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/userdata/dmoses/b3_features/zenodo")
    p.add_argument("--out", default=None)
    p.add_argument("--eval_set", type=int, default=1)
    p.add_argument("--train_amt", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compare", nargs="+", default=None,
                   help="compare existing manifest json files instead of generating")
    args = p.parse_args()

    if args.compare:
        compare(args.compare)
        return

    labels = pd.read_hdf(os.path.join(args.data_dir, "training_labels.h5"))
    labels_te = pd.read_hdf(os.path.join(args.data_dir, "realtime_test_labels.h5"))
    n_test = len(labels_te)
    labels = pd.concat((labels, labels_te), ignore_index=True)
    n = len(labels)

    train, val, te, test_day = T.compute_splits(
        n, n_test, args.eval_set, train_amt=args.train_amt, seed=args.seed)
    meta = {
        "split_seed": args.seed,
        "eval_set": args.eval_set,
        "train_amt": args.train_amt,
        "n_total": n,
        "n_test": n_test,
        "data_dir": args.data_dir,
        "note": "test = realtime held-out set (last n_test rows); "
                "fold-internal test is unused for the reported metrics",
    }
    out = args.out or f"repro/runs/split_manifest_evalset{args.eval_set}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    T.write_split_manifest(out, labels, train, val, test_day, meta=meta)
    print("wrote", out)
    print(f"counts  train={len(train)}  val={len(val)}  test={len(test_day)}")


if __name__ == "__main__":
    main()
