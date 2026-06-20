"""Tests for eval/runner.py — CLI evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from data.base import PoseSample
from eval.runner import _parse_args, _build_segmentor, _serialisable, main
from pose.parts import Part


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_sample(h: int = 200, w: int = 200) -> PoseSample:
    """Synthetic PoseSample with a simple part mask and visible keypoints."""
    image = np.zeros((h, w, 3), dtype=np.uint8)

    # Part mask: top half TORSO, bottom-left UPPER_LEG_L, bottom-right UPPER_LEG_R
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:h // 2, :]       = int(Part.TORSO)
    mask[h // 2:, :w // 2] = int(Part.UPPER_LEG_L)
    mask[h // 2:, w // 2:] = int(Part.UPPER_LEG_R)

    # 17 visible keypoints spread across the image
    keypoints = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        keypoints[i] = [w * 0.3 + i * 5, h * 0.3 + i * 4, 2.0]

    return PoseSample(
        image=image,
        keypoints=keypoints,
        part_mask=mask,
        bbox=np.array([10.0, 10.0, w - 20.0, h - 20.0], dtype=np.float32),
        image_id=i,
        ann_id=i,
    )


def _make_dataset(n: int = 3):
    """List-based fake dataset of n synthetic samples."""
    samples = [_make_sample() for _ in range(n)]

    class _FakeDataset:
        def __len__(self): return n
        def __getitem__(self, i): return samples[i]

    return _FakeDataset()


# ── Argument parsing ──────────────────────────────────────────────────────────

class TestParseArgs:
    def test_defaults(self):
        args = _parse_args([])
        assert args.data_root    == "data/raw"
        assert args.split        == "val2017"
        assert args.subset       is None
        assert args.segmentor    == "oracle"
        assert args.pck_threshold == 0.2
        assert args.out          is None

    def test_custom_values(self):
        args = _parse_args([
            "--data-root", "my/data",
            "--split", "train2017",
            "--subset", "50",
            "--segmentor", "segformer",
            "--pck-threshold", "0.5",
            "--out", "results.json",
        ])
        assert args.data_root     == "my/data"
        assert args.split         == "train2017"
        assert args.subset        == 50
        assert args.segmentor     == "segformer"
        assert args.pck_threshold == 0.5
        assert args.out           == "results.json"

    def test_invalid_split_raises(self):
        with pytest.raises(SystemExit):
            _parse_args(["--split", "test2020"])

    def test_invalid_segmentor_raises(self):
        with pytest.raises(SystemExit):
            _parse_args(["--segmentor", "hrnet"])


# ── Segmentor factory ─────────────────────────────────────────────────────────

class TestBuildSegmentor:
    def test_oracle_returns_gt_oracle(self):
        from segmentation.base import GTOracleSegmentor
        seg = _build_segmentor("oracle")
        assert isinstance(seg, GTOracleSegmentor)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown segmentor"):
            _build_segmentor("unknown_model")


# ── JSON serialisation ────────────────────────────────────────────────────────

class TestSerialisable:
    def test_numpy_arrays_become_lists(self):
        summary = {
            "pck_per_joint": np.array([0.5, np.nan, 0.8]),
            "n_samples":     10,
        }
        out = _serialisable(summary)
        assert isinstance(out["pck_per_joint"], list)
        assert out["pck_per_joint"][1] is None    # NaN → None
        assert out["pck_per_joint"][0] == pytest.approx(0.5)

    def test_nan_float_becomes_none(self):
        summary = {"pck_overall": float("nan")}
        out = _serialisable(summary)
        assert out["pck_overall"] is None

    def test_finite_float_preserved(self):
        summary = {"oks_mean": 0.42}
        out = _serialisable(summary)
        assert out["oks_mean"] == pytest.approx(0.42)

    def test_json_serialisable(self):
        """Output must be JSON-dumpable without errors."""
        summary = {
            "pck_per_joint": np.array([0.5, np.nan]),
            "pck_overall":   float("nan"),
            "n_samples":     5,
            "joint_names":   ["nose", "left_eye"],
        }
        out = _serialisable(summary)
        json.dumps(out)   # must not raise


# ── Full evaluation loop (mocked dataset + segmentor) ────────────────────────

class TestEvaluate:
    def _run(self, extra_args: list[str] | None = None, n_samples: int = 3):
        """Run evaluate() with a fake dataset and oracle segmentor."""
        from eval.runner import evaluate

        args = _parse_args(extra_args or [])
        dataset = _make_dataset(n_samples)

        with (
            patch("eval.runner._build_dataset", return_value=dataset),
            patch("eval.runner._build_segmentor",
                  return_value=_build_segmentor("oracle")),
        ):
            return evaluate(args)

    def test_n_samples_matches(self):
        summary = self._run(n_samples=3)
        assert summary["n_samples"] == 3

    def test_summary_has_required_keys(self):
        summary = self._run()
        for key in ("pck_overall", "pck_per_joint", "oks_mean",
                    "mpjpe_overall", "mpjpe_per_joint",
                    "n_samples", "skipped", "segmentor",
                    "split", "pck_threshold", "joint_names"):
            assert key in summary

    def test_pck_overall_in_0_1(self):
        summary = self._run()
        v = summary["pck_overall"]
        assert np.isnan(v) or (0.0 <= v <= 1.0)

    def test_oks_mean_in_0_1(self):
        summary = self._run()
        v = summary["oks_mean"]
        assert np.isnan(v) or (0.0 <= v <= 1.0)

    def test_skipped_counted_on_file_not_found(self):
        """FileNotFoundError on dataset[i] → sample skipped."""
        from eval.runner import evaluate

        class _BrokenDataset:
            def __len__(self): return 3
            def __getitem__(self, i):
                if i == 1:
                    raise FileNotFoundError("image not on disk")
                return _make_sample()

        args = _parse_args([])
        with (
            patch("eval.runner._build_dataset", return_value=_BrokenDataset()),
            patch("eval.runner._build_segmentor",
                  return_value=_build_segmentor("oracle")),
        ):
            summary = evaluate(args)

        assert summary["skipped"] == 1
        assert summary["n_samples"] == 2

    def test_skipped_when_no_part_mask(self):
        """Sample with part_mask=None → oracle raises ValueError → skipped."""
        from eval.runner import evaluate

        sample_no_mask = _make_sample()
        sample_no_mask.part_mask = None

        class _Dataset:
            def __len__(self): return 2
            def __getitem__(self, i):
                return sample_no_mask if i == 0 else _make_sample()

        args = _parse_args([])
        with (
            patch("eval.runner._build_dataset", return_value=_Dataset()),
            patch("eval.runner._build_segmentor",
                  return_value=_build_segmentor("oracle")),
        ):
            summary = evaluate(args)

        assert summary["skipped"] == 1
        assert summary["n_samples"] == 1

    def test_custom_pck_threshold_recorded(self):
        summary = self._run(["--pck-threshold", "0.5"])
        assert summary["pck_threshold"] == pytest.approx(0.5)


# ── JSON output ───────────────────────────────────────────────────────────────

class TestJSONOutput:
    def test_json_file_written(self, tmp_path):
        out_file = tmp_path / "results.json"
        dataset  = _make_dataset(2)

        with (
            patch("eval.runner._build_dataset", return_value=dataset),
            patch("eval.runner._build_segmentor",
                  return_value=_build_segmentor("oracle")),
        ):
            main(["--out", str(out_file)])

        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert "pck_overall" in data
        assert "n_samples"   in data

    def test_json_parent_dir_created(self, tmp_path):
        out_file = tmp_path / "nested" / "dir" / "results.json"
        dataset  = _make_dataset(1)

        with (
            patch("eval.runner._build_dataset", return_value=dataset),
            patch("eval.runner._build_segmentor",
                  return_value=_build_segmentor("oracle")),
        ):
            main(["--out", str(out_file)])

        assert out_file.exists()
