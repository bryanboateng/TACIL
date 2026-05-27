from pathlib import Path


def find_project_root() -> Path:
    path = Path(__file__).resolve()
    for parent in [path, *path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Project root not found")


PROJECT_ROOT = find_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output"
POLICY_ROOT = OUTPUT_ROOT / "policies"
CCIL_ROOT = OUTPUT_ROOT / "ccil"
