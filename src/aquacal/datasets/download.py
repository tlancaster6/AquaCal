"""Dataset download and caching utilities."""

from __future__ import annotations

import hashlib
import shutil
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def get_cache_dir() -> Path:
    """Get or create the AquaCal dataset cache directory.

    The cache is created at `./aquacal_data/` relative to the current working directory.
    A `.gitignore` file is automatically created to prevent accidental commits.

    Returns:
        Path to the cache directory
    """
    cache_dir = Path.cwd() / "aquacal_data"
    cache_dir.mkdir(exist_ok=True)

    # Create .gitignore if it doesn't exist
    gitignore = cache_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n", encoding="utf-8")

    return cache_dir


def download_with_progress(
    url: str,
    dest: Path,
    expected_checksum: str | None = None,
    max_retries: int = 3,
) -> None:
    """Download a file with progress bar and checksum validation.

    Args:
        url: URL to download from
        dest: Destination file path
        expected_checksum: Expected checksum in format "algorithm:hash"
            (e.g., "md5:abc123..." or "sha256:def456...")
        max_retries: Maximum number of retry attempts on failure

    Raises:
        RuntimeError: If download fails after max retries or checksum mismatch
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        try:
            # Download with streaming and progress bar
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            # Write to temporary file first
            temp_dest = dest.with_suffix(dest.suffix + ".tmp")

            with (
                open(temp_dest, "wb") as f,
                tqdm(
                    desc=dest.name,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress_bar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))

            # Verify checksum if provided
            if expected_checksum:
                # Parse algorithm:hash format
                if ":" not in expected_checksum:
                    raise ValueError(
                        f"Invalid checksum format: {expected_checksum}. "
                        "Expected format: 'algorithm:hash' (e.g., 'md5:abc123...')"
                    )

                algorithm, expected_hash = expected_checksum.split(":", 1)

                # Compute actual checksum
                if algorithm == "md5":
                    hash_obj = hashlib.md5()
                elif algorithm == "sha256":
                    hash_obj = hashlib.sha256()
                else:
                    raise ValueError(f"Unsupported checksum algorithm: {algorithm}")

                with open(temp_dest, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hash_obj.update(chunk)

                actual_hash = hash_obj.hexdigest()
                if actual_hash != expected_hash:
                    temp_dest.unlink()  # Delete corrupted file
                    raise RuntimeError(
                        f"Checksum mismatch for {dest.name}. "
                        f"Expected: {expected_hash}, Got: {actual_hash}"
                    )

            # Move to final destination
            temp_dest.replace(dest)
            return

        except (requests.RequestException, RuntimeError) as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"Download failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Download failed after {max_retries} attempts: {e}")


def download_and_extract(dataset_name: str, dataset_info: dict) -> Path:
    """Download and extract a dataset from Zenodo.

    If the dataset is already extracted in the cache, returns the cached path
    without re-downloading.

    Args:
        dataset_name: Name of the dataset (e.g., 'medium', 'large')
        dataset_info: Dataset metadata from manifest

    Returns:
        Path to the extracted dataset directory

    Raises:
        ValueError: If Zenodo metadata is missing
        RuntimeError: If download or extraction fails
    """
    cache_dir = get_cache_dir()
    extracted_dir = cache_dir / dataset_name

    # Check if already extracted (cache hit)
    if extracted_dir.exists():
        return extracted_dir

    # Validate Zenodo metadata
    record_id = dataset_info.get("zenodo_record_id")
    filename = dataset_info.get("zenodo_filename")
    checksum = dataset_info.get("checksum")

    if not record_id or not filename:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing Zenodo metadata in manifest. "
            "Cannot download."
        )

    # Construct Zenodo URL
    url = f"https://zenodo.org/records/{record_id}/files/{filename}"

    # Download to cache/downloads/
    download_dir = cache_dir / "downloads"
    download_dir.mkdir(exist_ok=True)
    zip_path = download_dir / filename

    if not zip_path.exists():
        print(f"Downloading {dataset_name} from Zenodo...")
        download_with_progress(url, zip_path, expected_checksum=checksum)
    else:
        print(f"Using cached download: {zip_path}")

    # Extract
    print(f"Extracting {dataset_name}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_dir)

    return extracted_dir


def clear_cache(dataset_name: str | None = None) -> None:
    """Clear the dataset cache.

    Args:
        dataset_name: If provided, clear only that dataset's cache.
            If None, clear the entire cache directory.
    """
    cache_dir = get_cache_dir()

    if dataset_name:
        # Clear specific dataset
        dataset_dir = cache_dir / dataset_name
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        # Also remove download if it exists
        download_dir = cache_dir / "downloads"
        if download_dir.exists():
            for file in download_dir.glob(f"*{dataset_name}*"):
                file.unlink()
    else:
        # Clear entire cache
        if cache_dir.exists():
            shutil.rmtree(cache_dir)


def get_cache_info() -> dict:
    """Get information about the current cache state.

    Returns:
        Dict with cache_dir path, list of cached datasets, and total size in MB
    """
    cache_dir = get_cache_dir()

    cached_datasets = []
    total_size_bytes = 0

    if cache_dir.exists():
        for item in cache_dir.iterdir():
            if item.is_dir() and item.name != "downloads":
                cached_datasets.append(item.name)
                # Calculate directory size
                for file in item.rglob("*"):
                    if file.is_file():
                        total_size_bytes += file.stat().st_size

    return {
        "cache_dir": str(cache_dir),
        "cached_datasets": cached_datasets,
        "total_size_mb": total_size_bytes / (1024 * 1024),
    }
