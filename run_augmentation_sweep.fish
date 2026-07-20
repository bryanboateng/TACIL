#!/usr/bin/env fish
#
# Sweeps create_augmented_data over all combinations of:
#   - model artifact
#   - action_noise_fraction
#   - error_limit_fraction
#
# Usage:
#   fish run_augmentation_sweep.fish
#
# To walk away from your computer while it runs:
#   nohup fish run_augmentation_sweep.fish > sweep_overall.log 2>&1 &
#   disown
set -x CUDA_VISIBLE_DEVICES 1
# Ordered alphabetically by lip (1, 10, 15, 5, null) to match the alphabetical
# order the sweep processes runs in.
set models cosmic-forest-11 fiery-flower-13 robust-hill-15 valiant-haze-12 flowing-serenity-10
# lips[i] is the dynamics-model Lipschitz constant for models[i] (same order)
set lips 1 10 15 5 null
set noises 0.0 0.1 0.3
set fractions 0.25 0.5 1.0 2.0
set logdir logs/augmentation_sweep
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
            set notes "data-lip-$lip-action-$noise-error-$frac"
            set logfile "$logdir/$tag.log"
            echo "[$i/$total] model=$model action_noise_fraction=$noise error_limit_fraction=$frac"
            echo "  -> notes=$notes"
            echo "  -> logging to $logfile"
            uv run python -m scripts.data.augmentation.tacil.create_data.create_augmented_data \
                --model-artifact=tacil-dynamics-model-$model:v0 \
                --dataset-artifact=expert-dataset:v0 \
                --action_noise_fraction=$noise \
                --error_limit_fraction=$frac \
                --wandb.notes=$notes \
                --wandb.project=opq > $logfile 2>&1
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
