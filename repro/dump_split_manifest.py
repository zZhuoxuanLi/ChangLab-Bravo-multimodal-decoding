"""
Generate (or compare) a lightweight train/val/test split manifest.

The manifest lists which trials are in each split (index / block /
within-block index / ground truth) WITHOUT copying any neural data, so you can
verify that different runs used the same split and compare across experiments.
It only loads the small label files, so it runs in seconds (no 11.5 GB load).

Generate the reference manifest (3 files) for a given fold into a directory:
    python repro/dump_split_manifest.py --eval_set 1 \
        --out_dir repro/runs/split_manifest_evalset1

Compare the manifests written by several runs (checks split membership is
identical across them). Pass each run's directory (the one holding
split_train.json / split_val.json / split_test.json):
    python repro/dump_split_manifest.py --compare \
        repro/runs/bs_32 repro/runs/bs_512
"""
import argparse
import json
import os

import pandas as pd

# reuse the exact split logic + manifest writer from the training script
import train_text_decoder as T


def load_split_index_sets(run_dir, prefix="split"):
    out = {}
    for name in ("train", "val", "test"):
        with open(os.path.join(run_dir, f"{prefix}_{name}.json")) as f:
            out[name] = [row["index"] for row in json.load(f)]
    return out


def compare(run_dirs):
    ref, ok = None, True
    for d in run_dirs:
        s = load_split_index_sets(d)
        if ref is None:
            ref = s
            print(f"reference: {d}")
            for k, v in s.items():
                print(f"  {k}: n={len(v)}")
            continue
        same = all(s[k] == ref[k] for k in ("train", "val", "test"))
        ok = ok and same
        print(f"{'MATCH  ' if same else 'DIFFERS'}: {d}")
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
    p.add_argument("--out_dir", default=None,
                   help="directory to write split_train/val/test.json into")
    p.add_argument("--eval_set", type=int, default=1)
    p.add_argument("--train_amt", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compare", nargs="+", default=None,
                   help="compare existing run directories instead of generating")
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
    out_dir = args.out_dir or f"repro/runs/split_manifest_evalset{args.eval_set}"
    paths = T.write_split_manifest(out_dir, labels, train, val, test_day)
    print("wrote:")
    for name, pth in paths.items():
        print(f"  {name}: {pth}")
    print(f"counts  train={len(train)}  val={len(val)}  test={len(test_day)}")


if __name__ == "__main__":
    main()
