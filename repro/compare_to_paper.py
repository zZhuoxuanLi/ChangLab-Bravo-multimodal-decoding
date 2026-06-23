"""
Compare our reproduction to the paper's published real-time results.

Loads:
  * the paper's per-sentence results (results_and_error_rates.csv from the
    dataset), and
  * our run's realtime_predictions.csv,
then reports mean / median / pseudo-blocked-median WER, CER and PER for each
side by side, using the same pseudo-blocking used in the paper notebook
(blocks of 10 sentences, word-count weighted).

Lightweight (CPU only). Example:
    python repro/compare_to_paper.py \
        --paper_csv /userdata/dmoses/b3_features/zenodo/results_and_error_rates.csv \
        --our_csv   repro/runs/run1/realtime_predictions.csv
"""
import argparse
import os

import numpy as np
import pandas as pd


def parceled_metric(vals, lengths=None, parcelation=10):
    dist, cur_list, cur_lengths = [], [], []
    if lengths is None:
        lengths = np.ones(len(vals))
    for k, x in enumerate(vals):
        cur_list.append(x)
        cur_lengths.append(np.nan if pd.isna(x) else lengths[k])
        if (k + 1) % parcelation == 0:
            num = np.nansum(np.array(cur_list) * np.array(cur_lengths))
            dist.append(num / np.nansum(np.array(cur_lengths)))
            cur_list, cur_lengths = [], []
    if len(cur_list) > parcelation / 2:
        dist.append(np.nansum(np.array(cur_list) * np.array(cur_lengths))
                    / np.nansum(np.array(cur_lengths)))
    return np.array(dist)


def summarize(df, gt_col, wer_col, cer_col, per_col):
    gts = df[gt_col].astype(str).values
    wc = np.array([len(g.split()) for g in gts])
    cc = np.array([len(g) for g in gts])
    out = {}
    out["n"] = len(df)
    out["mean_wer"] = float(np.nanmean(df[wer_col].values))
    out["median_wer"] = float(np.nanmedian(df[wer_col].values))
    out["pb_median_wer"] = float(np.nanmedian(parceled_metric(df[wer_col].values, wc)))
    out["pb_median_cer"] = float(np.nanmedian(parceled_metric(df[cer_col].values, cc)))
    out["pb_median_per"] = float(np.nanmedian(parceled_metric(df[per_col].values, wc)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paper_csv",
                   default="/userdata/dmoses/b3_features/zenodo/results_and_error_rates.csv")
    p.add_argument("--our_csv",
                   default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "repro", "runs", "run1", "realtime_predictions.csv"))
    args = p.parse_args()

    paper = pd.read_csv(args.paper_csv)
    ours = pd.read_csv(args.our_csv)

    paper_s = summarize(paper, "Ground Truth", "WER", "CER", "PER")
    ours_s = summarize(ours, "gt", "wer", "cer", "per")

    rows = [
        ("n sentences", "n", "%d"),
        ("mean WER", "mean_wer", "%.4f"),
        ("median WER (per-sentence)", "median_wer", "%.4f"),
        ("pseudo-blocked median WER", "pb_median_wer", "%.4f"),
        ("pseudo-blocked median CER", "pb_median_cer", "%.4f"),
        ("pseudo-blocked median PER", "pb_median_per", "%.4f"),
    ]
    print(f"{'metric':32s} {'paper':>12s} {'ours':>12s}")
    print("-" * 58)
    for label, key, fmt in rows:
        pv = fmt % paper_s[key]
        ov = fmt % ours_s[key]
        print(f"{label:32s} {pv:>12s} {ov:>12s}")


if __name__ == "__main__":
    main()
