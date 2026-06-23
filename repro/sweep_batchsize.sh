#!/usr/bin/env bash
#
# Batch-size sweep for the text decoder.
# Submits ONE single-GPU job per batch size (the model is small enough that even
# bs=512 fits on a single GPU). Each run writes to its own out_dir + log.
#
# Usage:
#   bash repro/sweep_batchsize.sh
#   bash repro/sweep_batchsize.sh "32 64 128"        # custom list
#   QUEUE=mind-gpu bash repro/sweep_batchsize.sh     # different GPU queue
#
# NOTE: do NOT use gpu@pia (incompatible). Use skull-gpu / spirit-gpu / mind-gpu.
#
# After the runs finish, summarize them with:
#   python repro/aggregate_runs.py --runs_dir repro/runs
#
set -euo pipefail

REPO=/home/zli/b3paper/ChangLab-Bravo-multimodal-decoding
PY=/userdata/zli/b3env/bin/python
DATA=/userdata/dmoses/b3_features/zenodo
QUEUE="${QUEUE:-mind-gpu}"    # NEVER gpu@pia; override with: QUEUE=skull-gpu bash ...
MEM="${MEM:-64}"              # host RAM in GB (for the 11.5 GB data load)

# default sweep; override by passing a quoted list as $1
BATCH_SIZES="${1:-32 48 64 96 128 192 256 288 384 512}"

mkdir -p "$REPO/logs"

for B in $BATCH_SIZES; do
  submit_job -q "$QUEUE" -g 1 -m "$MEM" \
    -o "$REPO/logs/bs_${B}.txt" -n "bs_${B}" \
    -x "$PY" "$REPO/repro/train_text_decoder.py" \
    --data_dir "$DATA" \
    --out_dir "$REPO/repro/runs/bs_${B}" \
    --bs "$B"
  echo "submitted batch size $B -> logs/bs_${B}.txt"
done

echo
echo "All jobs submitted. Monitor with:  qstat -u zli"
echo "Aggregate when done:  $PY repro/aggregate_runs.py --runs_dir $REPO/repro/runs"
