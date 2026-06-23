#!/usr/bin/env bash
#
# Evaluate every finished run under repro/runs/ and write a metrics .pkl next to
# each checkpoint. Submits ONE CPU job per run (the flashlight beam search is the
# bottleneck and runs on CPU, so no GPU is needed).
#
# Usage:
#   bash repro/eval_sweep.sh                      # all runs/bs_* dirs
#   bash repro/eval_sweep.sh "runs/bs_64 runs/run1"   # explicit dirs (rel to repro/)
#
# After the jobs finish, plot with:
#   python repro/plot_results.py \
#       --inputs repro/runs/bs_*/metrics.pkl --metrics wers,raw_wers,wpms \
#       --out repro/runs/sweep_metrics.png
#
set -euo pipefail

REPO=/home/zli/b3paper/ChangLab-Bravo-multimodal-decoding
PY=/userdata/zli/b3env/bin/python
DATA=/userdata/dmoses/b3_features/zenodo
QUEUE="${QUEUE:-pia-batch.q}"   # CPU queue; beam search is CPU-bound
CORES="${CORES:-4}"
MEM="${MEM:-96}"                # host RAM in GB; full data load peaks ~18 GB
                                # (float64 file + float32 copy), 96 is comfortable
CKPT="${CKPT:-final_model.pth}" # which checkpoint to evaluate in each run dir

mkdir -p "$REPO/logs"

# build the list of run directories
if [ "$#" -ge 1 ]; then
  RUN_DIRS=""
  for d in $1; do RUN_DIRS="$RUN_DIRS $REPO/repro/$d"; done
else
  RUN_DIRS=$(ls -d "$REPO"/repro/runs/bs_* 2>/dev/null || true)
fi

if [ -z "${RUN_DIRS// }" ]; then
  echo "no run directories found under $REPO/repro/runs/bs_*"
  exit 1
fi

for RUN in $RUN_DIRS; do
  NAME=$(basename "$RUN")
  WEIGHTS="$RUN/$CKPT"
  if [ ! -f "$WEIGHTS" ]; then
    echo "SKIP $NAME (no $CKPT)"
    continue
  fi
  submit_job -q "$QUEUE" -c "$CORES" -m "$MEM" \
    -o "$REPO/logs/eval_${NAME}.txt" -n "eval_${NAME}" \
    -x "$PY" "$REPO/repro/eval_model.py" \
    "$WEIGHTS" "$RUN/metrics.pkl" \
    --data_dir "$DATA"
  echo "submitted eval $NAME -> $RUN/metrics.pkl  (log: logs/eval_${NAME}.txt)"
done

echo
echo "Monitor with:  qstat -u zli"
echo "Plot when done: $PY repro/plot_results.py --inputs repro/runs/bs_*/metrics.pkl --metrics wers,raw_wers,wpms --out repro/runs/sweep_metrics.png"
