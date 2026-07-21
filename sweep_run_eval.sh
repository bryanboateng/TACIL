#!/bin/sh
#
# W&B-sweep shim for policy evaluation.
#
# The sweep varies one value  <notes>::<checkpoint_artifact>  and passes it as
# --spec. This shim splits it and calls the untouched evaluate_policy.py with the
# notes supplied from the outside.
#
# POSIX sh (not fish/bash) so it runs on any agent machine. It is the sweep's
# `program`; you don't run it by hand.

# Optional first arg = wandb project. Old sweeps call this with --spec directly,
# so if the first arg is already --spec, default the project to opq.
case "$1" in
    --spec*) project="opq" ;;
    *) project="$1"; shift ;;
esac

# wandb passes either  --spec=<value>  or  --spec <value>
case "$1" in
    --spec=*) spec="${1#--spec=}" ;;
    --spec) spec="$2" ;;
    *) spec="$1" ;;
esac

notes="${spec%%::*}"
artifact="${spec#*::}"

exec uv run python -m scripts.evaluation.evaluate_policy \
    --wandb.project="$project" \
    --wandb.notes="$notes" \
    dp \
    --checkpoint_artifact="$artifact"
