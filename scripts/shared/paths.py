from pathlib import Path


def _find_project_root() -> Path:
    path = Path(__file__).resolve()
    for parent in [path, *path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Project root not found")


PROJECT_ROOT_DIRECTORY_PATH = _find_project_root()
OUTPUT_DIRECTORY_PATH = PROJECT_ROOT_DIRECTORY_PATH / "output"
