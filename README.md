# TACIL: Trajectory Augmentation for Corrective Imitation Learning.

https://github.com/user-attachments/assets/e1bd3b67-b109-469d-b296-28d7a53bba49

https://github.com/user-attachments/assets/dca6d5a9-1f1f-415d-b704-c256ae648456

Same episode: without augmentation (left) and with TACIL (right).

## What TACIL does

A policy trained by imitation learning does not know how to get back when a
disturbance pushes it away from its demonstrations. [CCIL](https://github.com/personalrobotics/CCIL) adds corrective transitions
that lead back from states one step away from the demonstrations. TACIL extends
this concept to corrective trajectories that lead back from states many steps away.
This is accomplished by solving a learned dynamics ensemble backwards until the uncertainty
of the ensemble gets too high.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.13 and only
supports Linux.

```bash
uv sync
```

Everything is tracked in [Weights & Biases](https://wandb.ai), so log in first
with `uv run wandb login`. Datasets and models are stored as W&B artifacts.

## Pipeline

First the expert demonstrations are ingested and uploaded as a dataset.

```bash
uv run python -m scripts.data.expert.ingest.ingest_expert_data \
  --environment PEG_INSERTION \
  --dataset-name peg-insertion
```

A smaller dataset with a fixed random subset of the episodes can be made from the dataset:

```bash
uv run python -m scripts.data.expert.create_subsampled_dataset \
  --source-dataset-artifact peg-insertion-1000-of-1000:latest \
  --episode-count 100
```

A policy is then trained and evaluated in a single run.
An example of a Behavior Cloning run with TACIL looks like this:

```bash
uv run python -m scripts.policies.bc.run \
  --seed 0 \
  --environment PEG_INSERTION \
  --wandb-project tacil \
  --expert-dataset-artifact peg-insertion-100-of-1000:latest \
  policy.validation:none \
  augmentation:tacil \
    --augmentation.dynamics.window-length 4 \
    --augmentation.dynamics.lipschitz-limit None \
    --augmentation.data.action-noise 0.1 \
    --augmentation.data.stopping-rule account \
    --augmentation.data.error-limit-in-steps 10 \
  evaluation.stage:tuning
```

The evaluation stage decides which episode seeds the policy is evaluated on.
`evaluation.stage:tuning` uses 100 episodes per disturbance condition starting
at seed 20000 and is meant for hyperparameter sweeps, `evaluation.stage:final`
uses 200 episodes per condition starting at seed 30000 and is meant for the
reported results, and `evaluation.stage:custom` takes both values explicitly.

With `--evaluation.save-rollouts` the evaluation rollouts are uploaded as an
artifact, and videos can be rendered from them later:

```bash
uv run python -m scripts.evaluation.render_videos \
  --video-project tacil-videos \
  --run-paths <entity>/<project>/<run-id>
```
