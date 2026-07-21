#!/usr/bin/env fish
#
# Augmentation sweep for the 0.25-subsample experiment. Same as
# run_augmentation_sweep.fish, but writes into project opq-sub025, uses the
# subsampled dataset + its own dynamics models, and tags every note with
# "sub0.25-" so the runs (and their videos) are self-describing.
#
#   nohup fish run_augmentation_sweep_sub.fish > sweep_sub.log 2>&1 &
#   disown
#
# Everything lives in opq-sub025, including its own copy of the expert dataset:
# create_subsampled_dataset resolves "expert-dataset:latest" inside its own project.
#
# Prerequisites, both already done: the expert data was ingested into opq-sub025
# and subsampled there, and the 5 dynamics models were trained on the result.
# Honour a GPU chosen from the outside; fall back to 0.
if not set -q CUDA_VISIBLE_DEVICES
    set -x CUDA_VISIBLE_DEVICES 0
end

# The 5 dynamics models trained on the SUBSAMPLED data, in the same order as
# their lip values below.
set models fluent-plant-5 jumping-music-12 visionary-meadow-13 firm-galaxy-10 drawn-spaceship-8
set lips 1 10 15 5 null

set dataset_artifact subsampled-dataset-winter-feather-2:latest

set noises 0.0 0.1 0.3
set fractions 0.25 0.5 1.0 2.0 4.0 8.0
set project opq-sub025

set logdir logs/augmentation_sweep_sub
mkdir -p $logdir
set nm (count $models)
set nn (count $noises)
set nf (count $fractions)
set total (math "$nm * $nn * $nf")
set i 0
set failed
for mi in (seq $nm)
    set model $models[$mi]
    set lip $lips[$mi]
    for noise in $noises
        for frac in $fractions
            set i (math $i + 1)
            set tag "$model"_noise$noise"_frac"$frac
            set notes "data-sub0.25-lip-$lip-action-$noise-error-$frac"
            set logfile "$logdir/$tag.log"
            echo "[$i/$total] model=$model action_noise_fraction=$noise error_limit_fraction=$frac"
            echo "  -> notes=$notes"
            echo "  -> logging to $logfile"
            uv run python -m scripts.data.augmentation.tacil.create_data.create_augmented_data \
                --model-artifact=tacil-dynamics-model-$model:v0 \
                --dataset-artifact=$dataset_artifact \
                --action_noise_fraction=$noise \
                --error_limit_fraction=$frac \
                --wandb.notes=$notes \
                --wandb.project=$project > $logfile 2>&1
            if test $status -ne 0
                echo "  FAILED: $tag (see $logfile)"
                set -a failed $tag
            end
        end
    end
end
echo ""
echo "Done: $total runs attempted."
if test (count $failed) -gt 0
    echo "Failed runs ("(count $failed)"):"
    for f in $failed
        echo "  - $f"
    end
else
    echo "All runs succeeded."
end
