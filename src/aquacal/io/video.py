"""Video loading and synchronized frame extraction."""

from __future__ import annotations

import os
from typing import Iterator

import cv2
import numpy as np
from numpy.typing import NDArray


class VideoSet:
    """
    Manages multiple synchronized video files.

    Videos are assumed to be temporally synchronized (frame 0 in all videos
    corresponds to the same moment in time). Videos may have different total
    frame counts; the synchronized length is the minimum across all videos.

    Supports context manager protocol for automatic resource cleanup.

    Example:
        >>> paths = {'cam0': 'video0.mp4', 'cam1': 'video1.mp4'}
        >>> with VideoSet(paths) as videos:
        ...     print(f"Frame count: {videos.frame_count}")
        ...     for idx, frames in videos.iterate_frames(step=10):
        ...         # frames is dict[str, NDArray | None]
        ...         process(frames)

    Attributes:
        video_paths: Dict mapping camera names to video file paths.
    """

    def __init__(self, video_paths: dict[str, str]) -> None:
        """
        Initialize VideoSet with paths to synchronized videos.

        Does NOT open video files immediately (lazy initialization).
        Validates that all files exist.

        Args:
            video_paths: Dict mapping camera_name to video file path.
                         Must have at least one camera.

        Raises:
            FileNotFoundError: If any video file does not exist.
            ValueError: If video_paths is empty.
        """
        if not video_paths:
            raise ValueError("video_paths cannot be empty")

        # Validate all files exist
        for camera_name, path in video_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Video file not found: {path}")

        self.video_paths = video_paths
        self._captures: dict[str, cv2.VideoCapture] = {}
        self._is_open = False
        self._frame_count: int | None = None

    @property
    def camera_names(self) -> list[str]:
        """List of camera names (sorted for deterministic ordering)."""
        return sorted(self.video_paths.keys())

    @property
    def frame_count(self) -> int:
        """
        Synchronized frame count (minimum across all videos).

        Opens videos if not already open (to read frame counts).
        """
        if self._frame_count is None:
            # Ensure videos are open to get frame counts
            if not self._is_open:
                self.open()

            # Get minimum frame count across all videos
            counts = []
            for camera_name in self.camera_names:
                cap = self._captures[camera_name]
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                counts.append(count)

            self._frame_count = min(counts)

        return self._frame_count

    @property
    def is_open(self) -> bool:
        """Whether video captures are currently open."""
        return self._is_open

    def open(self) -> None:
        """
        Open all video captures.

        Called automatically on first frame access. Safe to call multiple times.

        Raises:
            RuntimeError: If any video file cannot be opened by OpenCV.
        """
        if self._is_open:
            return

        # Open all video captures
        for camera_name in self.camera_names:
            path = self.video_paths[camera_name]
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                # Clean up any already-opened captures
                for opened_cap in self._captures.values():
                    opened_cap.release()
                self._captures = {}
                raise RuntimeError(f"Failed to open video file: {path}")
            self._captures[camera_name] = cap

        self._is_open = True

    def close(self) -> None:
        """
        Release all video captures.

        Safe to call multiple times or when already closed.
        """
        if not self._is_open:
            return

        # Release all captures
        for cap in self._captures.values():
            cap.release()

        self._captures = {}
        self._is_open = False

    def __enter__(self) -> VideoSet:
        """Context manager entry. Opens videos."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit. Releases video captures."""
        self.close()

    def get_frame(self, frame_idx: int) -> dict[str, NDArray[np.uint8] | None]:
        """
        Get a single synchronized frame from all cameras.

        Opens videos if not already open.

        Args:
            frame_idx: Frame index (0-based). Must be < frame_count.

        Returns:
            Dict mapping camera_name to BGR image (H, W, 3) as uint8.
            Value is None if that camera's frame could not be read.

        Raises:
            IndexError: If frame_idx < 0 or frame_idx >= frame_count.

        Note:
            This method seeks to the requested frame, which may be slow for
            non-sequential access. For sequential iteration, use iterate_frames().
        """
        if not self._is_open:
            self.open()

        # Validate frame index
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(
                f"frame_idx {frame_idx} out of range [0, {self.frame_count})"
            )

        # Seek and read from all cameras
        frames = {}
        for camera_name in self.camera_names:
            cap = self._captures[camera_name]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            frames[camera_name] = frame if ret else None  # type: ignore[assignment]

        return frames  # type: ignore[return-value]

    def iterate_frames(
        self,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> Iterator[tuple[int, dict[str, NDArray[np.uint8] | None]]]:
        """
        Iterate over synchronized frames.

        Opens videos if not already open. Frames are read sequentially for
        efficiency (no seeking when step=1).

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
            >>> with VideoSet(paths) as videos:
            ...     for idx, frames in videos.iterate_frames(step=5):
            ...         for cam, img in frames.items():
            ...             if img is not None:
            ...                 cv2.imwrite(f'{cam}_{idx}.png', img)
        """
        if not self._is_open:
            self.open()

        # Validate parameters
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")

        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")

        if stop is None:
            stop = self.frame_count

        if stop < start:
            raise ValueError(f"stop ({stop}) must be >= start ({start})")

        # Iterate over frames
        if step == 1:
            # Sequential reading - no seeking needed
            # Position all captures at start frame
            for camera_name in self.camera_names:
                cap = self._captures[camera_name]
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            for frame_idx in range(start, stop):
                frames = {}
                for camera_name in self.camera_names:
                    cap = self._captures[camera_name]
                    ret, frame = cap.read()
                    frames[camera_name] = frame if ret else None  # type: ignore[assignment]
                yield frame_idx, frames  # type: ignore[misc]
        else:
            # Non-sequential - use seeking for each frame
            for frame_idx in range(start, stop, step):
                frames = {}
                for camera_name in self.camera_names:
                    cap = self._captures[camera_name]
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    frames[camera_name] = frame if ret else None  # type: ignore[assignment]
                yield frame_idx, frames  # type: ignore[misc]
