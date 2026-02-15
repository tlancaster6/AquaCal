"""Image directory loading for multi-camera calibration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from natsort import natsorted
from numpy.typing import NDArray


class ImageSet:
    """
    Manages multiple synchronized image directories.

    Images are assumed to be temporally synchronized (frame 0 in all directories
    corresponds to the same moment in time). All directories must have the same
    number of images (strict validation).

    Images are loaded in natural sort order (img_1, img_2, img_10 not img_1, img_10, img_2).
    Supports JPEG (.jpg, .jpeg) and PNG (.png) files (case-insensitive).

    Mirrors VideoSet API for drop-in compatibility via FrameSet protocol.

    Example:
        >>> paths = {'cam0': 'data/cam0/', 'cam1': 'data/cam1/'}
        >>> with ImageSet(paths) as images:
        ...     print(f"Frame count: {images.frame_count}")
        ...     for idx, frames in images.iterate_frames(step=5):
        ...         # frames is dict[str, NDArray | None]
        ...         process(frames)

    Attributes:
        image_dirs: Dict mapping camera names to image directory paths.
    """

    def __init__(self, image_dirs: dict[str, str]) -> None:
        """
        Initialize ImageSet with paths to synchronized image directories.

        Validates that:
        - All directories exist
        - All directories contain at least one image
        - All directories have the same number of images

        Args:
            image_dirs: Dict mapping camera_name to image directory path.
                        Must have at least one camera.

        Raises:
            FileNotFoundError: If any directory does not exist.
            ValueError: If image_dirs is empty, or directories have
                        different image counts, or no images found.
        """
        if not image_dirs:
            raise ValueError("image_dirs cannot be empty")

        # Validate all directories exist
        for camera_name, path in image_dirs.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Directory not found: {path}")
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Not a directory: {path}")

        self.image_dirs = image_dirs
        self._image_paths: dict[str, list[Path]] = {}
        self._frame_count: int | None = None
        self._is_open = True  # No resources to open, always "open"

        # Load and validate image paths
        self._load_image_paths()

    def _load_image_paths(self) -> None:
        """
        Load image file paths from all directories.

        Collects all JPEG and PNG files, sorts them naturally,
        and validates that all directories have the same count.

        Raises:
            ValueError: If no images found or counts don't match
        """
        image_extensions = {".jpg", ".jpeg", ".png"}
        counts = []

        for camera_name in sorted(self.image_dirs.keys()):
            dir_path = Path(self.image_dirs[camera_name])

            # Collect all image files (case-insensitive extension check)
            image_files = [
                f
                for f in dir_path.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]

            if not image_files:
                raise ValueError(
                    f"No images found in directory: {dir_path} "
                    f"(looking for {image_extensions})"
                )

            # Natural sort by filename
            sorted_files = natsorted(image_files, key=lambda p: p.name)

            self._image_paths[camera_name] = sorted_files
            counts.append(len(sorted_files))

        # Validate all cameras have same count
        if len(set(counts)) > 1:
            count_str = ", ".join(
                f"{cam}={count}"
                for cam, count in zip(sorted(self.image_dirs.keys()), counts)
            )
            raise ValueError(
                f"All camera directories must have the same number of images. "
                f"Found different counts: {count_str}"
            )

        self._frame_count = counts[0]

    @property
    def camera_names(self) -> list[str]:
        """List of camera names (sorted for deterministic ordering)."""
        return sorted(self.image_dirs.keys())

    @property
    def frame_count(self) -> int:
        """
        Number of synchronized frames (validated equal across all cameras).

        Returns:
            Total number of image frames
        """
        if self._frame_count is None:
            self._load_image_paths()
        return self._frame_count  # type: ignore[return-value]

    @property
    def is_open(self) -> bool:
        """Whether image set is ready (always True - no resources to manage)."""
        return self._is_open

    def open(self) -> None:
        """
        Open image set (no-op for compatibility with FrameSet protocol).

        ImageSet has no resources to open (unlike VideoSet with VideoCapture).
        Images are loaded on-demand during iteration.
        """
        self._is_open = True

    def close(self) -> None:
        """
        Close image set (no-op for compatibility with FrameSet protocol).

        ImageSet has no resources to release (unlike VideoSet with VideoCapture).
        """
        self._is_open = True  # Keep open for potential reuse

    def __enter__(self) -> ImageSet:
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def get_frame(self, frame_idx: int) -> dict[str, NDArray[np.uint8] | None]:
        """
        Get a single synchronized frame from all cameras.

        Args:
            frame_idx: Frame index (0-based). Must be < frame_count.

        Returns:
            Dict mapping camera_name to BGR image (H, W, 3) as uint8.
            Value is None if that camera's image could not be read.

        Raises:
            IndexError: If frame_idx < 0 or frame_idx >= frame_count.
        """
        # Validate frame index
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(
                f"frame_idx {frame_idx} out of range [0, {self.frame_count})"
            )

        # Load images from all cameras
        frames = {}
        for camera_name in self.camera_names:
            img_path = self._image_paths[camera_name][frame_idx]
            img = cv2.imread(str(img_path))
            frames[camera_name] = img  # type: ignore[assignment]

        return frames  # type: ignore[return-value]

    def iterate_frames(
        self,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> Iterator[tuple[int, dict[str, NDArray[np.uint8] | None]]]:
        """
        Iterate over synchronized frames.

        Loads images on-demand from disk. Step parameter is supported but
        images are assumed to be pre-curated (no automatic subsampling needed).

        Args:
            start: Starting frame index (inclusive). Default 0.
            stop: Ending frame index (exclusive). None means frame_count.
            step: Frame step. 1 = every frame, 2 = every other frame, etc.
                  Must be >= 1.

        Yields:
            Tuple of (frame_idx, frame_dict) where frame_dict maps
            camera_name to BGR image (H, W, 3) as uint8, or None if read failed.

        Raises:
            ValueError: If start < 0, stop < start, or step < 1.

        Example:
            >>> with ImageSet(paths) as images:
            ...     for idx, frames in images.iterate_frames(step=5):
            ...         for cam, img in frames.items():
            ...             if img is not None:
            ...                 process_image(img)
        """
        # Validate parameters
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")

        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")

        if stop is None:
            stop = self.frame_count

        if stop < start:
            raise ValueError(f"stop ({stop}) must be >= start ({start})")

        # Iterate over frame indices
        for frame_idx in range(start, stop, step):
            frames = {}
            for camera_name in self.camera_names:
                img_path = self._image_paths[camera_name][frame_idx]
                img = cv2.imread(str(img_path))
                frames[camera_name] = img  # type: ignore[assignment]

            yield frame_idx, frames  # type: ignore[misc]
