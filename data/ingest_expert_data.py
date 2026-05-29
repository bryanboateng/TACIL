from pathlib import Path

import tyro
import wandb


def main(expert_data_file_path: Path):
    with wandb.init(project="abc", job_type="ingest-expert-dataset") as run:
        artifact = wandb.Artifact(name="expert-dataset", type="dataset")
        artifact.add_dir(local_path=str(expert_data_file_path))
        run.log_artifact(artifact_or_path=artifact)


if __name__ == "__main__":
    tyro.cli(main)
