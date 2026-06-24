"""
Violin plots of decoding metrics across models / runs.

Reads either:
  * metrics .pkl files written by eval_model.py  (wers, wpms, raw_wers, ...), or
  * realtime_predictions.csv files written by train_text_decoder.py
    (columns wer, cer, per, wpm -> mapped to wers, cers, pers, wpms).

Examples:
    # compare the batch-size sweep straight from the training CSVs
    python repro/plot_results.py \
        --inputs repro/runs/bs_*/realtime_predictions.csv \
        --metrics wers,wpms --out repro/runs/sweep_plots.png

    # compare two eval pickles (mirrors the lab example)
    python repro/plot_results.py \
        --inputs model_87_eval.pkl model_156_eval.pkl \
        --labels model_87 model_156 \
        --metrics wers,wpms --out eval_plots.png
"""
import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / server-safe
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# nicer axis labels per metric
METRIC_LABELS = {
    "wers": "Test WER", "raw_wers": "Raw WER (no early stop)",
    "cers": "Test CER", "pers": "Test PER", "raw_cers": "Raw CER",
    "raw_pers": "Raw PER", "raw_greedy_cers": "Greedy PER",
    "wpms": "WPM", "speaking_times": "Speaking time (s)",
    "early_stopping": "Early-stop rate",
}
# body colors (match the reference plot: WER blue, WPM red)
METRIC_COLORS = {
    "wers": "#2f6f9f", "raw_wers": "#3a7d44", "cers": "#7b5ea7",
    "pers": "#d1812c", "wpms": "#b24b45", "speaking_times": "#8a6d3b",
}
# value-label number format per metric
METRIC_FMT = {"wpms": "{:.1f}", "speaking_times": "{:.2f}"}
DEFAULT_FMT = "{:.2f}"

# mean/median line styling (mean = dashed gray, median = solid black)
MEAN_COLOR = "#555555"
MEDIAN_COLOR = "#111111"
MEAN_DASH = (0, (5, 2))

# map realtime_predictions.csv columns -> metric keys
CSV_COLMAP = {"wer": "wers", "cer": "cers", "per": "pers", "wpm": "wpms",
              "speak_time": "speaking_times"}


def violin_plot(ax, violin_data, violin_labels, xlabel, ylabel, title,
                color="#2f6f9f", value_fmt=DEFAULT_FMT):
    positions = list(range(1, len(violin_data) + 1))
    parts = ax.violinplot(violin_data, positions=positions,
                          showmeans=True, showmedians=True, showextrema=True)

    # whiskers / extrema
    for partname in ("cbars", "cmins", "cmaxes"):
        vp = parts[partname]
        vp.set_edgecolor("#222222")
        vp.set_linewidth(1.1)

    # mean: dashed gray  |  median: solid black  (kept distinct on purpose)
    vp = parts["cmeans"]
    vp.set_edgecolor(MEAN_COLOR)
    vp.set_linewidth(1.8)
    vp.set_linestyle(MEAN_DASH)

    vp = parts["cmedians"]
    vp.set_edgecolor(MEDIAN_COLOR)
    vp.set_linewidth(2.4)

    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.3)

    # print the actual mean & median numbers beside each violin
    for pos, arr in zip(positions, violin_data):
        med = float(np.median(arr))
        mean = float(np.mean(arr))
        ax.annotate(value_fmt.format(med), (pos, med), textcoords="offset points",
                    xytext=(9, -1), fontsize=8, color=MEDIAN_COLOR,
                    ha="left", va="center", fontweight="bold")
        ax.annotate(value_fmt.format(mean), (pos, mean), textcoords="offset points",
                    xytext=(9, 11), fontsize=8, color=MEAN_COLOR,
                    ha="left", va="center")

    ax.set_xticks(positions)
    ax.set_xticklabels(violin_labels, rotation=30, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    median_handle = mlines.Line2D([], [], color=MEDIAN_COLOR, linewidth=2.4,
                                  label="median")
    mean_handle = mlines.Line2D([], [], color=MEAN_COLOR, linewidth=1.8,
                                linestyle=MEAN_DASH, label="mean")
    ax.legend(handles=[median_handle, mean_handle], frameon=False,
              loc="best", fontsize=9)
    return ax


def load_metrics(path):
    """Return a dict of metric_name -> 1D array, from a .pkl or a .csv."""
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        d = {}
        for col, key in CSV_COLMAP.items():
            if col in df.columns:
                d[key] = df[col].values
        return d
    raise ValueError(f"unsupported input (need .pkl or .csv): {path}")


def batch_key(label):
    """Sort key: the (last) integer embedded in a label, e.g. 'bs_128' -> 128.
    Labels without a number sort to the end, keeping their original order."""
    nums = re.findall(r"\d+", label)
    return int(nums[-1]) if nums else float("inf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help=".pkl (eval_model.py) or realtime_predictions.csv files")
    p.add_argument("--labels", nargs="*", default=None,
                   help="one label per input (default: parent dir / file name)")
    p.add_argument("--metrics", default="wers,wpms",
                   help="comma-separated metric keys to plot (one column each)")
    p.add_argument("--out", default="eval_plots.png")
    p.add_argument("--title", default="",
                   help="optional figure suptitle")
    p.add_argument("--no_sort", action="store_true",
                   help="keep input order instead of sorting by batch size")
    args = p.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    data = [load_metrics(p_) for p_ in args.inputs]

    if args.labels:
        assert len(args.labels) == len(args.inputs), "need one label per input"
        labels = args.labels
    else:
        labels = []
        for p_ in args.inputs:
            parent = os.path.basename(os.path.dirname(os.path.abspath(p_)))
            labels.append(parent if parent else os.path.splitext(os.path.basename(p_))[0])

    paths = list(args.inputs)
    # order from smallest to largest batch size (by the number in each label)
    if not args.no_sort:
        order = sorted(range(len(labels)), key=lambda i: batch_key(labels[i]))
        data = [data[i] for i in order]
        labels = [labels[i] for i in order]
        paths = [paths[i] for i in order]

    # show the resolved left->right mapping so any mislabeling is obvious
    print("plot order (left -> right):")
    for lab, pth in zip(labels, paths):
        print(f"  {lab:<12} <- {pth}")

    # side-by-side: one column per metric
    per_panel_w = max(5.5, len(data) * 0.85)
    fig, axs = plt.subplots(1, len(metrics),
                            figsize=(per_panel_w * len(metrics), 5.2))
    if len(metrics) == 1:
        axs = [axs]

    for ax, metric in zip(axs, metrics):
        vdata, vlab = [], []
        for m, lab in zip(data, labels):
            if metric in m and len(m[metric]) > 0:
                vals = np.asarray(m[metric], dtype=float)
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    vdata.append(vals)
                    vlab.append(lab)
        if not vdata:
            ax.set_title(f"{metric}: no data found")
            continue
        violin_plot(ax, vdata, vlab, "batch size",
                    METRIC_LABELS.get(metric, metric),
                    f"Distribution of {METRIC_LABELS.get(metric, metric)}",
                    color=METRIC_COLORS.get(metric, "#2f6f9f"),
                    value_fmt=METRIC_FMT.get(metric, DEFAULT_FMT))

    if args.title:
        fig.suptitle(args.title, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print("wrote", args.out)

    # text summary
    for metric in metrics:
        print(f"\n{METRIC_LABELS.get(metric, metric)} ({metric}):")
        for m, lab in zip(data, labels):
            if metric in m and len(m[metric]) > 0:
                v = np.asarray(m[metric], dtype=float)
                v = v[~np.isnan(v)]
                print(f"  {lab:<16} median={np.median(v):.4f}  "
                      f"mean={np.mean(v):.4f}  n={len(v)}")


if __name__ == "__main__":
    main()
