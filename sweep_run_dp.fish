#!/usr/bin/env fish
#
# W&B-sweep shim for diffusion-policy training.
#
# The sweep varies one value  <notes>::<dataset_artifact>  and passes it as
# --spec. This shim splits it and calls the untouched train.py with the notes
# supplied from the outside, so train.py never learns about the notes schema.
#
# It is the sweep's `program`; you don't run it by hand.

# wandb passes either  --spec=<value>  or  --spec <value>
if test (count $argv) -ge 2
    set spec $argv[2]
else
    set spec (string replace -- '--spec=' '' -- $argv[1])
end

set parts (string split -m1 '::' $spec)

uv run python -m scripts.policies.dp.train \
    --wandb.project=opq \
    --wandb.notes=$parts[1] \
    --dataset_artifact=$parts[2]
