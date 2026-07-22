import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "tools"
    / "long_video_eval"
    / "scripts"
    / "run_helios_long_timeseries_metric.py"
)


def _load_metric_module():
    spec = importlib.util.spec_from_file_location("helios_timeseries_metric", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load metric script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TimeseriesMetricTest(unittest.TestCase):
    def test_synthetic_video_schema_and_finite_slopes(self):
        metric = _load_metric_module()
        chunk_frames = 33
        n_chunks = 5
        height = width = 64

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = temp_path / "moving_gradient.mp4"
            output_path = temp_path / "metrics.json"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                12.0,
                (width, height),
            )
            self.assertTrue(writer.isOpened())
            try:
                x = np.arange(width, dtype=np.uint16)[None, :]
                y = np.arange(height, dtype=np.uint16)[:, None]
                for frame_index in range(n_chunks * chunk_frames):
                    blue = np.broadcast_to((x * 4 + frame_index * 3) % 256, (height, width))
                    green = np.broadcast_to((y * 4 + frame_index * 2) % 256, (height, width))
                    red = (x * 2 + y * 2 + frame_index * 5) % 256
                    frame = np.stack((blue, green, red), axis=-1).astype(np.uint8)
                    writer.write(frame)
            finally:
                writer.release()

            exit_code = metric.main(
                [str(video_path), "--chunk_frames", str(chunk_frames), "--out", str(output_path)]
            )
            self.assertEqual(exit_code, 0)

            with output_path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)

            self.assertEqual(
                set(result),
                {
                    "video",
                    "n_chunks",
                    "chunk_frames",
                    "motion_resolution",
                    "distance_backend",
                    "per_chunk",
                    "boundary_excess",
                    "slopes",
                    "auc",
                },
            )
            self.assertEqual(result["n_chunks"], n_chunks)
            self.assertEqual(result["chunk_frames"], chunk_frames)
            self.assertEqual(result["motion_resolution"], [384, 640])
            self.assertEqual(result["distance_backend"], "l2")
            self.assertEqual(
                set(result["per_chunk"]), {"motion", "saturation", "chroma"}
            )
            for values in result["per_chunk"].values():
                self.assertEqual(len(values), n_chunks)
            self.assertEqual(len(result["boundary_excess"]), n_chunks - 1)
            self.assertEqual(
                set(result["slopes"]),
                {"motion", "saturation", "chroma", "boundary_excess"},
            )
            for slope_pair in result["slopes"].values():
                self.assertEqual(set(slope_pair), {"ols", "theil_sen"})
                self.assertTrue(math.isfinite(slope_pair["ols"]))
                self.assertTrue(math.isfinite(slope_pair["theil_sen"]))
            self.assertEqual(set(result["auc"]), {"saturation", "chroma"})
            self.assertTrue(all(math.isfinite(value) for value in result["auc"].values()))

    def test_motion_series_comparable_across_encoded_resolutions(self):
        metric = _load_metric_module()
        chunk_frames = 8
        n_chunks = 2
        high_size = 128
        rng = np.random.default_rng(7)
        texture = rng.integers(0, 256, (high_size, high_size, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_paths = []
            for size in (64, high_size):
                video_path = temp_path / f"moving_texture_{size}.mp4"
                video_paths.append(video_path)
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    12.0,
                    (size, size),
                )
                self.assertTrue(writer.isOpened())
                try:
                    for frame_index in range(n_chunks * chunk_frames):
                        high_frame = np.roll(texture, frame_index * 2, axis=1)
                        frame = cv2.resize(
                            high_frame,
                            (size, size),
                            interpolation=cv2.INTER_AREA,
                        )
                        writer.write(frame)
                finally:
                    writer.release()

            motion_series = [
                metric.analyze_video(path, chunk_frames=chunk_frames)["per_chunk"]["motion"]
                for path in video_paths
            ]

            np.testing.assert_allclose(
                motion_series[0],
                motion_series[1],
                rtol=0.08,
                atol=0.02,
            )


if __name__ == "__main__":
    unittest.main()
