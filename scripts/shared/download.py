import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from scripts.shared.paths import DOWNLOAD_DIRECTORY_PATH


def download_and_extract_zip(*, url: str) -> Path:
    destination_path = (
        DOWNLOAD_DIRECTORY_PATH / Path(urllib.parse.urlparse(url=url).path).stem
    )

    if destination_path.exists():
        return destination_path

    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / "archive.zip"
        _download(url=url, save_file_path=archive_path)

        with zipfile.ZipFile(file=archive_path) as archive_file:
            archive_file.extractall(path=destination_path)

    return destination_path


# ---- Private --------------------------------------------


def _download(*, url: str, save_file_path: Path) -> None:
    with tqdm(
        desc=f"Downloading {save_file_path.name}",
        unit="B",
        unit_scale=True,
        unit_divisor=1_024,
    ) as progress_bar:

        def report(block_count: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                progress_bar.total = total_size
            progress_bar.update(n=block_count * block_size - progress_bar.n)

        urllib.request.urlretrieve(url=url, filename=save_file_path, reporthook=report)
