"""Tests for video loading and frame extraction."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from aquacal.io.video import VideoSet


@pytest.fixture
def test_videos(tmp_path: Path) -> dict[str, str]:
    """Create small test videos for testing."""
    paths = {}
    for cam_name, num_frames, color in [
        ("cam0", 10, (255, 0, 0)),  # 10 frames, blue
        ("cam1", 12, (0, 255, 0)),  # 12 frames, green (2 extra)
    ]:
        path = tmp_path / f"{cam_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 30.0, (64, 48))
        for i in range(num_frames):
            frame = np.full((48, 64, 3), color, dtype=np.uint8)
            # Add frame number as visual marker
            cv2.putText(
                frame,
                str(i),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            writer.write(frame)
        writer.release()
        paths[cam_name] = str(path)
    return paths


class TestVideoSetInit:
    def test_valid_paths(self, test_videos):
        """VideoSet created with valid paths, not yet open."""
        vs = VideoSet(test_videos)
        assert not vs.is_open
        assert vs.video_paths == test_videos

    def test_missing_file_raises(self, tmp_path):
        """Missing video file raises FileNotFoundError."""
        paths = {"cam0": str(tmp_path / "nonexistent.mp4")}
        with pytest.raises(FileNotFoundError):
            VideoSet(paths)

    def test_empty_paths_raises(self):
        """Empty video_paths raises ValueError."""
        with pytest.raises(ValueError):
            VideoSet({})


class TestVideoSetProperties:
    def test_camera_names_sorted(self, test_videos):
        """camera_names returns sorted list."""
        vs = VideoSet(test_videos)
        assert vs.camera_names == ["cam0", "cam1"]

    def test_frame_count_minimum(self, test_videos):
        """frame_count returns minimum (10, not 12)."""
        vs = VideoSet(test_videos)
        assert vs.frame_count == 10
        vs.close()

    def test_is_open_initially_false(self, test_videos):
        """is_open is False before opening."""
        vs = VideoSet(test_videos)
        assert not vs.is_open

    def test_is_open_true_after_open(self, test_videos):
        """is_open is True after open()."""
        vs = VideoSet(test_videos)
        vs.open()
        assert vs.is_open
        vs.close()


class TestVideoSetOpenClose:
    def test_open_makes_captures_available(self, test_videos):
        """open() creates VideoCapture objects."""
        vs = VideoSet(test_videos)
        vs.open()
        assert vs.is_open
        vs.close()

    def test_close_releases_captures(self, test_videos):
        """close() releases captures and sets is_open to False."""
        vs = VideoSet(test_videos)
        vs.open()
        vs.close()
        assert not vs.is_open

    def test_multiple_close_safe(self, test_videos):
        """Multiple close() calls are safe."""
        vs = VideoSet(test_videos)
        vs.open()
        vs.close()
        vs.close()  # Should not raise

    def test_context_manager(self, test_videos):
        """Context manager opens and closes properly."""
        with VideoSet(test_videos) as vs:
            assert vs.is_open
        assert not vs.is_open


class TestVideoSetGetFrame:
    def test_returns_dict_with_correct_keys(self, test_videos):
        """get_frame returns dict with all camera names."""
        with VideoSet(test_videos) as vs:
            frames = vs.get_frame(0)
            assert set(frames.keys()) == {"cam0", "cam1"}

    def test_images_correct_shape_dtype(self, test_videos):
        """Frames are BGR images with correct shape and dtype."""
        with VideoSet(test_videos) as vs:
            frames = vs.get_frame(0)
            for img in frames.values():
                assert img is not None
                assert img.dtype == np.uint8
                assert img.shape == (48, 64, 3)

    def test_negative_index_raises(self, test_videos):
        """Negative frame index raises IndexError."""
        with VideoSet(test_videos) as vs:
            with pytest.raises(IndexError):
                vs.get_frame(-1)

    def test_index_at_frame_count_raises(self, test_videos):
        """frame_idx >= frame_count raises IndexError."""
        with VideoSet(test_videos) as vs:
            with pytest.raises(IndexError):
                vs.get_frame(10)  # frame_count is 10

    def test_auto_opens(self, test_videos):
        """get_frame auto-opens if not already open."""
        vs = VideoSet(test_videos)
        assert not vs.is_open
        _frames = vs.get_frame(0)
        assert vs.is_open
        vs.close()


class TestVideoSetIterateFrames:
    def test_default_yields_all_frames(self, test_videos):
        """Default iteration yields frames 0 to frame_count-1."""
        with VideoSet(test_videos) as vs:
            indices = [idx for idx, _ in vs.iterate_frames()]
            assert indices == list(range(10))

    def test_start_parameter(self, test_videos):
        """start parameter sets starting index."""
        with VideoSet(test_videos) as vs:
            indices = [idx for idx, _ in vs.iterate_frames(start=5)]
            assert indices == [5, 6, 7, 8, 9]

    def test_stop_parameter(self, test_videos):
        """stop parameter sets ending index (exclusive)."""
        with VideoSet(test_videos) as vs:
            indices = [idx for idx, _ in vs.iterate_frames(stop=5)]
            assert indices == [0, 1, 2, 3, 4]

    def test_step_parameter(self, test_videos):
        """step parameter skips frames."""
        with VideoSet(test_videos) as vs:
            indices = [idx for idx, _ in vs.iterate_frames(step=2)]
            assert indices == [0, 2, 4, 6, 8]

    def test_step_3(self, test_videos):
        """step=3 yields correct indices."""
        with VideoSet(test_videos) as vs:
            indices = [idx for idx, _ in vs.iterate_frames(step=3)]
            assert indices == [0, 3, 6, 9]

    def test_combined_params(self, test_videos):
        """start, stop, step work together."""
        with VideoSet(test_videos) as vs:
            indices = [idx for idx, _ in vs.iterate_frames(start=1, stop=8, step=2)]
            assert indices == [1, 3, 5, 7]

    def test_invalid_start_raises(self, test_videos):
        """Negative start raises ValueError."""
        with VideoSet(test_videos) as vs:
            with pytest.raises(ValueError):
                list(vs.iterate_frames(start=-1))

    def test_invalid_step_raises(self, test_videos):
        """step < 1 raises ValueError."""
        with VideoSet(test_videos) as vs:
            with pytest.raises(ValueError):
                list(vs.iterate_frames(step=0))

    def test_stop_less_than_start_raises(self, test_videos):
        """stop < start raises ValueError."""
        with VideoSet(test_videos) as vs:
            with pytest.raises(ValueError):
                list(vs.iterate_frames(start=5, stop=3))

    def test_auto_opens(self, test_videos):
        """iterate_frames auto-opens if not already open."""
        vs = VideoSet(test_videos)
        assert not vs.is_open
        for idx, frames in vs.iterate_frames(stop=1):
            pass
        assert vs.is_open
        vs.close()

    def test_yields_correct_frame_data(self, test_videos):
        """Yielded frames contain valid image data."""
        with VideoSet(test_videos) as vs:
            for idx, frames in vs.iterate_frames():
                assert "cam0" in frames
                assert "cam1" in frames
                for cam, img in frames.items():
                    assert img is not None
                    assert img.shape == (48, 64, 3)
