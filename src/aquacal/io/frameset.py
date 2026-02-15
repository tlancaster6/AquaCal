"""Protocol for frame source abstraction."""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class FrameSet(Protocol):
    """
    Protocol for synchronized multi-camera frame sources.

    Defines the common interface for both VideoSet and ImageSet,
    allowing the detection pipeline to work with either video files
    or image directories transparently.

    Implementations must support:
    - Property access: camera_names, frame_count, is_open
    - Frame iteration: iterate_frames(start, stop, step)
    - Random access: get_frame(frame_idx)
    - Context manager: __enter__, __exit__
    - Resource management: open(), close()

    Example:
        >>> def process_frames(source: FrameSet):
        ...     with source:
        ...         for idx, frames in source.iterate_frames(step=5):
        ...             # Process frames dict[str, NDArray | None]
        ...             pass
    """

    @property
    def camera_names(self) -> list[str]:
        """
        List of camera names (sorted for deterministic ordering).

        Returns:
            Sorted list of camera names
        """
        ...

    @property
    def frame_count(self) -> int:
        """
        Number of synchronized frames available.

        For VideoSet: minimum frame count across all videos
        For ImageSet: validated equal count across all directories

        Returns:
            Total number of frames
        """
        ...

    @property
    def is_open(self) -> bool:
        """
        Whether the frame source is currently open.

        Returns:
            True if resources are open and ready
        """
        ...

    def open(self) -> None:
        """
        Open/initialize frame source resources.

        For VideoSet: opens cv2.VideoCapture objects
        For ImageSet: no-op (no resources to open)

        Safe to call multiple times.
        """
        ...

    def close(self) -> None:
        """
        Close/release frame source resources.

        For VideoSet: releases cv2.VideoCapture objects
        For ImageSet: no-op (no resources to release)

        Safe to call multiple times.
        """
        ...

    def get_frame(self, frame_idx: int) -> dict[str, NDArray[np.uint8] | None]:
        """
        Get a single synchronized frame from all cameras.

        Args:
            frame_idx: Frame index (0-based). Must be < frame_count.

        Returns:
            Dict mapping camera_name to BGR image (H, W, 3) as uint8.
            Value is None if that camera's frame could not be read.

        Raises:
            IndexError: If frame_idx < 0 or frame_idx >= frame_count
        """
        ...

    def iterate_frames(
        self,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> Iterator[tuple[int, dict[str, NDArray[np.uint8] | None]]]:
        """
        Iterate over synchronized frames.

        Args:
            start: Starting frame index (inclusive). Default 0.
            stop: Ending frame index (exclusive). None means frame_count.
            step: Frame step. 1 = every frame, 2 = every other frame, etc.
                  Must be >= 1.

        Yields:
            Tuple of (frame_idx, frame_dict) where frame_dict maps
            camera_name to BGR image (H, W, 3) as uint8, or None if read failed.

        Raises:
            ValueError: If start < 0, stop < start, or step < 1
        """
        ...

    def __enter__(self) -> FrameSet:
        """Context manager entry. Opens frame source."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit. Closes frame source."""
        ...
