"""
Aggregate the summary.json files from several runs (e.g. a batch-size sweep)
into a single table sorted by effective batch size.

Usage:
    python repro/aggregate_runs.py --runs_dir repro/runs
    python repro/aggregate_runs.py --runs_dir repro/runs --csv repro/runs/sweep_summary.csv
"""
import argparse
import glob
import json
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "repro", "runs"))
    p.add_argument("--csv", default=None, help="optional path to also write a CSV")
    args = p.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.runs_dir, "*", "summary.json"))):
        try:
            with open(f) as fh:
                s = json.load(fh)
        except Exception as e:
            print(f"skip {f}: {e!r}")
            continue
        s["run"] = os.path.basename(os.path.dirname(f))
        rows.append(s)

    if not rows:
        print(f"no summary.json found under {args.runs_dir}")
        return

    df = pd.DataFrame(rows)
    sort_key = "effective_batch_size" if "effective_batch_size" in df.columns else "run"
    df = df.sort_values(sort_key)

    cols = ["run", "batch_size", "accum_steps", "effective_batch_size", "seed",
            "best_val_wer_3gram", "realtime_net_wer", "pseudoblocked_median_wer",
            "pseudoblocked_median_cer", "pseudoblocked_median_per", "median_wpm"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df[cols].to_string(index=False))

    if args.csv:
        df[cols].to_csv(args.csv, index=False)
        print("\nwrote", args.csv)


if __name__ == "__main__":
    main()
