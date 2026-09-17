#!/bin/bash

set -euo pipefail

if [[ "${DETACHED_LAUNCH:-0}" != "1" ]]; then
    export DETACHED_LAUNCH=1
    nohup setsid bash "$0" "$@" </dev/null >/dev/null 2>&1 &
    disown || true
    echo "detached_pid=$!"
    exit 0
fi

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
    . "$HOME/.conda/etc/profile.d/conda.sh"
elif [ -f "/gpu01/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/gpu01/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/opt/miniconda3/etc/profile.d/conda.sh"
else
    echo "conda.sh not found"
    exit 1
fi

conda activate chenghao

if [ -f "/opt/rh/devtoolset-11/enable" ]; then
    source /opt/rh/devtoolset-11/enable
fi

ROOT=/home/chenghao/SpeechTokenizer-main
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT"

LOG_DIR="$ROOT/log"
SAVE_DIR="$ROOT/output/controller_checkpoints"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

OUTFILE="$LOG_DIR/controller_train.log"
PIDFILE="$LOG_DIR/controller_train.pid"

GPU_IDS=${GPU_IDS:-2}
PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-0}
SEGMENT_SEC=${SEGMENT_SEC:-1.0}
LOG_EVERY=${LOG_EVERY:-5}
N_LAYERS_LIST=${N_LAYERS_LIST:-"2 3 4 5 6 7 8"}
R_MAX_VALUES=${R_MAX_VALUES:-"1.0 1.5 2.0 2.5 3.0 3.5 4.0"}
PLR_VALUES=${PLR_VALUES:-"0.0 0.05 0.10 0.15 0.20 0.25 0.30"}
TOL_MODE=${TOL_MODE:-rel}
REL_MARGIN=${REL_MARGIN:-0.02}
ABS_MARGIN=${ABS_MARGIN:-0.0}

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

echo "log_file=$OUTFILE"
exec >> "$OUTFILE" 2>&1

echo "host=$(hostname)"
echo "gpu_ids=$GPU_IDS"
echo "log_file=$OUTFILE"
echo "batch_size=$PER_GPU_BATCH_SIZE"
echo "r_max_values=$R_MAX_VALUES"
echo "plr_values=$PLR_VALUES"
echo "tol_mode=$TOL_MODE"
echo "rel_margin=$REL_MARGIN"
echo "abs_margin=$ABS_MARGIN"
echo "$$" > "$PIDFILE"
echo "pid_file=$PIDFILE"
echo "pid=$$"
date

python scripts/train_controller.py --flow_ckpt /home/chenghao/SpeechTokenizer-main/output/flow_checkpoints_stage3/best_train.pt --data_dir /home/chenghao/SpeechTokenizer-main/LibriSpeech --save_dir /home/chenghao/SpeechTokenizer-main/output/controller_checkpoints --batch_size "$PER_GPU_BATCH_SIZE" --num_workers "$NUM_WORKERS" --segment_sec "$SEGMENT_SEC" --max_epochs 20 --patience 5 --n_layers_list $N_LAYERS_LIST --r_max_values $R_MAX_VALUES --plr_values $PLR_VALUES --tol_mode "$TOL_MODE" --rel_margin "$REL_MARGIN" --abs_margin "$ABS_MARGIN" --log_every "$LOG_EVERY"
