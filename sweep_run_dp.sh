#!/bin/sh
#
# W&B-sweep shim for diffusion-policy training.
#
# The sweep varies one value  <notes>::<dataset_artifact>  and passes it as
# --spec. This shim splits it and calls the untouched train.py with the notes
# supplied from the outside, so train.py never learns about the notes schema.
#
# POSIX sh (not fish/bash) so it runs on any agent machine. It is the sweep's
# `program`; you don't run it by hand.

# wandb passes either  --spec=<value>  or  --spec <value>
case "$1" in
    --spec=*) spec="${1#--spec=}" ;;
    --spec) spec="$2" ;;
    *) spec="$1" ;;
esac

notes="${spec%%::*}"
artifact="${spec#*::}"

exec uv run python -m scripts.policies.dp.train \
    --wandb.project=opq \
    --wandb.notes="$notes" \
    --dataset_artifact="$artifact"
