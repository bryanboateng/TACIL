#!/bin/sh
#
# Overnight pipeline: train a diffusion policy for every not-yet-trained
# combination, wait until that sweep's grid is exhausted, then evaluate every
# not-yet-evaluated combination. Each phase runs one wandb agent per GPU and
# blocks until they exit (a grid agent quits once there is no work left).
#
# Both generate_sweeps calls skip combinations that are already done, so this
# only ever schedules new work and is safe to re-run.
#
#   nohup sh run_train_and_eval.sh 0,1 > overnight.log 2>&1 &
#   disown
#
# Args:  $1 = comma-separated GPU ids (default "0")
#        $2 = wandb entity        (default "bryanboateng-team")
#        $3 = wandb project       (default "opq")

GPUS="${1:-0}"
ENTITY="${2:-bryanboateng-team}"
PROJECT="${3:-opq}"

run_phase() {
    phase="$1"
    echo
    echo "======== $(date) : phase '$phase' ========"

    # generate_sweeps exits non-zero when there is nothing new to schedule.
    if ! uv run python generate_sweeps.py "$phase" --entity "$ENTITY" --project "$PROJECT"; then
        echo "Nothing to schedule for '$phase' (or generation failed); skipping."
        return 0
    fi

    sweep_output=$(uv run wandb sweep -e "$ENTITY" -p "$PROJECT" "${phase}_sweep.yaml" 2>&1)
    echo "$sweep_output"
    sweep_path=$(echo "$sweep_output" | sed -n 's/.*wandb agent //p' | tail -1)
    if [ -z "$sweep_path" ]; then
        echo "Could not determine sweep id for '$phase'; aborting phase."
        return 1
    fi
    echo "Running sweep: $sweep_path"

    pids=""
    for gpu in $(echo "$GPUS" | tr ',' ' '); do
        echo "Launching agent on GPU $gpu"
        CUDA_VISIBLE_DEVICES="$gpu" uv run wandb agent "$sweep_path" &
        pids="$pids $!"
    done
    wait $pids
    echo "======== $(date) : phase '$phase' finished ========"
}

run_phase dp
run_phase eval

echo
echo "======== $(date) : ALL DONE ========"
