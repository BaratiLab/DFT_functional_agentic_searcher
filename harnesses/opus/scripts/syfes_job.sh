#!/bin/bash
# SyFES run + float64 re-check of the best forms (launched by the MCP tool syfes_submit as a systemd user unit).
# usage: syfes_job.sh OUTDIR N_FORMS GPUS SEED POPULATION TOURNAMENT TRIALS
set -e
OUT=$1; N=$2; GPUS=$3; SEED=$4; POP=$5; TOUR=$6; TRIALS=$7
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
$PY -u -W ignore scripts/syfes_run.py --out "$OUT" --n-forms "$N" --gpus "$GPUS" --seed "$SEED" \
    --population "$POP" --tournament "$TOUR" --trials "$TRIALS"
FIRST=${GPUS%%,*}
CUDA_VISIBLE_DEVICES=$FIRST $PY -u -W ignore scripts/syfes_recheck.py --out "$OUT" --top 10
echo "job complete"
