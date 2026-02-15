"""Tests for ImageSet image directory loading."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from aquacal.io.images import ImageSet


class TestImageSet:
    """Tests for ImageSet class."""

    @staticmethod
    def create_test_images(
        base_dir: Path, camera_dirs: dict[str, int]
    ) -> dict[str, str]:
        """
        Create test image directories with dummy images.

        Args:
            base_dir: Base directory to create camera subdirs in
            camera_dirs: Dict mapping camera name to number of images

        Returns:
            Dict mapping camera name to directory path
        """
        paths = {}
        for cam_name, num_images in camera_dirs.items():
            cam_dir = base_dir / cam_name
            cam_dir.mkdir(parents=True, exist_ok=True)

            # Create test images (small black images)
            for i in range(num_images):
                img = np.zeros((100, 100, 3), dtype=np.uint8)
                img_path = cam_dir / f"img_{i}.jpg"
                cv2.imwrite(str(img_path), img)

            paths[cam_name] = str(cam_dir)

        return paths

    def test_basic_initialization(self, tmp_path: Path):
        """Test basic ImageSet initialization and properties."""
        # Create 2 cameras with 5 images each
        paths = self.create_test_images(tmp_path, {"cam0": 5, "cam1": 5})

        img_set = ImageSet(paths)
        assert img_set.camera_names == ["cam0", "cam1"]
        assert img_set.frame_count == 5

    def test_natural_sort_ordering(self, tmp_path: Path):
        """Test that images are loaded in natural sort order."""
        cam_dir = tmp_path / "cam0"
        cam_dir.mkdir()

        # Create images in non-lexicographic order
        # Use PNG to avoid JPEG compression artifacts
        for i in [1, 2, 10, 20, 3]:
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            img[10, 10, 0] = i * 10  # Mark each image differently (larger values)
            cv2.imwrite(str(cam_dir / f"img_{i}.png"), img)

        img_set = ImageSet({"cam0": str(cam_dir)})

        # Read all frames and verify order
        frames = list(img_set.iterate_frames())
        assert len(frames) == 5

        # Check indices are in natural order: 1, 2, 3, 10, 20
        expected_indices = [1, 2, 3, 10, 20]
        for frame_idx, (idx, frame_dict) in enumerate(frames):
            assert idx == frame_idx
            img = frame_dict["cam0"]
            # Check the marker we set (img[10, 10, 0] = i * 10)
            marker_val = img[10, 10, 0]
            assert marker_val == expected_indices[frame_idx] * 10

    def test_mismatched_frame_counts_raises(self, tmp_path: Path):
        """Test that mismatched frame counts across cameras raises ValueError."""
        paths = self.create_test_images(tmp_path, {"cam0": 5, "cam1": 3})

        with pytest.raises(ValueError, match="different.*counts"):
            ImageSet(paths)

    def test_empty_directory_raises(self, tmp_path: Path):
        """Test that empty directory raises ValueError."""
        cam_dir = tmp_path / "cam0"
        cam_dir.mkdir()

        with pytest.raises(ValueError, match="No images found|empty"):
            ImageSet({"cam0": str(cam_dir)})

    def test_mixed_extensions(self, tmp_path: Path):
        """Test loading mixed jpg/png files."""
        cam_dir = tmp_path / "cam0"
        cam_dir.mkdir()

        # Create mix of .jpg and .png
        for i in range(3):
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            ext = ".jpg" if i % 2 == 0 else ".png"
            cv2.imwrite(str(cam_dir / f"img_{i}{ext}"), img)

        img_set = ImageSet({"cam0": str(cam_dir)})
        assert img_set.frame_count == 3

    def test_iterate_frames(self, tmp_path: Path):
        """Test frame iteration."""
        paths = self.create_test_images(tmp_path, {"cam0": 10, "cam1": 10})
        img_set = ImageSet(paths)

        # Test full iteration
        frames = list(img_set.iterate_frames())
        assert len(frames) == 10

        # Test with start/stop
        frames = list(img_set.iterate_frames(start=2, stop=5))
        assert len(frames) == 3
        assert frames[0][0] == 2  # First frame index is 2

        # Test with step
        frames = list(img_set.iterate_frames(step=2))
        assert len(frames) == 5
        assert [idx for idx, _ in frames] == [0, 2, 4, 6, 8]

    def test_get_frame(self, tmp_path: Path):
        """Test random access with get_frame."""
        paths = self.create_test_images(tmp_path, {"cam0": 5})
        img_set = ImageSet(paths)

        # Test valid access
        frame = img_set.get_frame(2)
        assert "cam0" in frame
        assert frame["cam0"].shape == (100, 100, 3)

        # Test out of bounds
        with pytest.raises(IndexError):
            img_set.get_frame(10)

        with pytest.raises(IndexError):
            img_set.get_frame(-1)

    def test_context_manager(self, tmp_path: Path):
        """Test context manager protocol."""
        paths = self.create_test_images(tmp_path, {"cam0": 3})

        with ImageSet(paths) as img_set:
            assert img_set.is_open
            frames = list(img_set.iterate_frames())
            assert len(frames) == 3

        # After context exit, should still be usable (no resources to clean)
        assert img_set.is_open

    def test_nonexistent_directory_raises(self):
        """Test that nonexistent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ImageSet({"cam0": "/nonexistent/path"})

    def test_case_insensitive_extensions(self, tmp_path: Path):
        """Test that uppercase extensions are recognized."""
        cam_dir = tmp_path / "cam0"
        cam_dir.mkdir()

        # Create images with uppercase extensions
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(cam_dir / "img_1.JPG"), img)
        cv2.imwrite(str(cam_dir / "img_2.PNG"), img)

        img_set = ImageSet({"cam0": str(cam_dir)})
        assert img_set.frame_count == 2


class TestFrameSourceAutoDetection:
    """Tests for _create_frame_source auto-detection."""

    def test_directory_creates_imageset(self, tmp_path: Path):
        """Test that directory paths create ImageSet."""
        from aquacal.io.detection import _create_frame_source

        # Create test directories
        cam0_dir = tmp_path / "cam0"
        cam1_dir = tmp_path / "cam1"
        cam0_dir.mkdir()
        cam1_dir.mkdir()

        # Add test images
        for i in range(3):
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2.imwrite(str(cam0_dir / f"img_{i}.jpg"), img)
            cv2.imwrite(str(cam1_dir / f"img_{i}.jpg"), img)

        paths = {"cam0": str(cam0_dir), "cam1": str(cam1_dir)}
        frame_source = _create_frame_source(paths)

        assert isinstance(frame_source, ImageSet)
        assert frame_source.frame_count == 3

    def test_file_creates_videoset(self, tmp_path: Path):
        """Test that file paths create VideoSet."""
        from aquacal.io.detection import _create_frame_source
        from aquacal.io.video import VideoSet

        # Create dummy video files (just need them to exist for this test)
        vid0 = tmp_path / "vid0.mp4"
        vid1 = tmp_path / "vid1.mp4"
        vid0.touch()
        vid1.touch()

        paths = {"cam0": str(vid0), "cam1": str(vid1)}

        # This will fail when trying to open the videos (not real videos)
        # but we can catch that and verify the type
        try:
            frame_source = _create_frame_source(paths)
            # If we get here, it created a VideoSet (might fail to open later)
            assert isinstance(frame_source, VideoSet)
        except RuntimeError:
            # Expected if VideoSet tries to open the fake videos
            pass

    def test_mixed_types_raises(self, tmp_path: Path):
        """Test that mixed directory and file paths raise ValueError."""
        from aquacal.io.detection import _create_frame_source

        cam_dir = tmp_path / "cam0"
        cam_dir.mkdir()
        vid_file = tmp_path / "vid1.mp4"
        vid_file.touch()

        paths = {"cam0": str(cam_dir), "cam1": str(vid_file)}

        with pytest.raises(ValueError, match="mix"):
            _create_frame_source(paths)


class TestDetectionWithImageSet:
    """Test detect_all_frames integration with ImageSet."""

    def test_detect_all_frames_with_image_dirs(self, tmp_path: Path):
        """Test that detect_all_frames works with image directories."""
        from aquacal.config.schema import BoardConfig
        from aquacal.core.board import BoardGeometry
        from aquacal.io.detection import detect_all_frames

        # Create test images
        cam0_dir = tmp_path / "cam0"
        cam1_dir = tmp_path / "cam1"
        cam0_dir.mkdir()
        cam1_dir.mkdir()

        for i in range(5):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(cam0_dir / f"img_{i}.jpg"), img)
            cv2.imwrite(str(cam1_dir / f"img_{i}.jpg"), img)

        paths = {"cam0": str(cam0_dir), "cam1": str(cam1_dir)}

        # Create a minimal board config
        board_config = BoardConfig(
            squares_x=5,
            squares_y=7,
            square_size=0.04,
            marker_size=0.03,
            dictionary="DICT_4X4_50",
        )
        board = BoardGeometry(board_config)

        # Run detection (won't find any boards in black images, but should work)
        result = detect_all_frames(paths, board, frame_step=1)

        assert result.camera_names == ["cam0", "cam1"]
        assert result.total_frames == 5
        # No detections expected (black images)
        assert len(result.frames) == 0
