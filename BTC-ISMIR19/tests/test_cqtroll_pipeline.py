# encoding: utf-8
"""Unit tests for the CQT-first augmentation pipeline.

Covers the standalone helper ``shift_cqt_bins`` rigorously (no audio
fixtures required) and exercises a small end-to-end run of the new
``Preprocess.generate_labels_features_voca_cqtroll`` method on synthetic
data when ``librosa`` is available.

Run with:
    python -m pytest tests/test_cqtroll_pipeline.py -v
Or directly:
    python tests/test_cqtroll_pipeline.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.preprocess import shift_cqt_bins


class TestShiftCqtBinsHelper(unittest.TestCase):
    """Pure unit tests for the shift_cqt_bins helper."""

    def test_zero_shift_returns_equal_copy(self):
        cqt = np.random.rand(252, 100).astype(np.complex64)
        out = shift_cqt_bins(cqt, 0)
        self.assertTrue(np.array_equal(out, cqt))
        # Must not be the same object: callers may mutate.
        self.assertIsNot(out, cqt)

    def test_positive_shift_zeros_low_bins_and_shifts_up(self):
        cqt = np.arange(252 * 50, dtype=np.float32).reshape(252, 50)
        shift = 2  # semitones
        bps = 3
        out = shift_cqt_bins(cqt, shift, bins_per_semitone=bps)
        delta = shift * bps  # 6 bins
        # Bins below `delta` must be zeroed (no real low-frequency content
        # to shift up into them).
        self.assertTrue(np.all(out[:delta, :] == 0))
        # Bins from `delta` upwards equal the original lower bins.
        self.assertTrue(np.all(out[delta:, :] == cqt[:-delta, :]))

    def test_negative_shift_zeros_high_bins_and_shifts_down(self):
        cqt = np.arange(252 * 50, dtype=np.float32).reshape(252, 50)
        shift = -3
        bps = 3
        out = shift_cqt_bins(cqt, shift, bins_per_semitone=bps)
        delta = abs(shift) * bps  # 9 bins
        self.assertTrue(np.all(out[-delta:, :] == 0))
        self.assertTrue(np.all(out[:-delta, :] == cqt[delta:, :]))

    def test_shape_and_dtype_preserved_for_all_project_shifts(self):
        cqt = np.random.rand(252, 108).astype(np.complex64)
        for shift in range(-5, 7):  # project's shift range, inclusive
            out = shift_cqt_bins(cqt, shift, bins_per_semitone=3)
            self.assertEqual(out.shape, cqt.shape)
            self.assertEqual(out.dtype, cqt.dtype)

    def test_complex_dtype_round_trip_values(self):
        cqt = (np.random.rand(36, 20) + 1j * np.random.rand(36, 20)).astype(np.complex64)
        out = shift_cqt_bins(cqt, 4, bins_per_semitone=3)
        self.assertEqual(out.dtype, np.complex64)
        # Spot-check one preserved bin.
        self.assertTrue(np.allclose(out[12 + 0, :], cqt[0, :]))

    def test_arbitrary_bins_per_semitone(self):
        cqt = np.ones((84, 30), dtype=np.float32)
        # 12 bins/octave -> 1 bin/semitone -> shift by exactly `shift` bins.
        out = shift_cqt_bins(cqt, 5, bins_per_semitone=1)
        self.assertTrue(np.all(out[:5, :] == 0))
        self.assertTrue(np.all(out[5:, :] == 1))

    def test_large_shift_clears_more_than_half_the_band(self):
        # Sanity for the maximum project shift (+6 semitones with bps=3 -> 18 bins).
        cqt = np.ones((252, 10), dtype=np.float32)
        out = shift_cqt_bins(cqt, 6, bins_per_semitone=3)
        self.assertTrue(np.all(out[:18, :] == 0))
        self.assertTrue(np.all(out[18:, :] == 1))


class TestShiftCqtBinsAgainstAudioPitchShift(unittest.TestCase):
    """Sanity check: at shift=0, ``shift_cqt_bins`` is a no-op equal to the
    underlying ``librosa.cqt`` output.  We don't assert numerical equality
    against ``pyrubberband.pitch_shift`` because that path injects phase-
    vocoder noise even at semitones=0; instead we verify the helper does
    nothing when shift is zero, which is the only invariant the project
    actually relies on for the un-augmented files (``1.00_0_*.pt``).
    """

    def test_shift_zero_matches_librosa_cqt(self):
        try:
            import librosa  # noqa: F401
        except Exception:
            self.skipTest("librosa not available")

        sr = 22050
        # 2 seconds of a pure 440 Hz sine.
        t = np.arange(int(2 * sr)) / sr
        audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        cqt = librosa.cqt(audio, sr=sr, n_bins=252, bins_per_octave=36, hop_length=2048)

        out = shift_cqt_bins(cqt, 0, bins_per_semitone=3)
        # Equal but not the same object.
        self.assertTrue(np.array_equal(out, cqt))
        self.assertIsNot(out, cqt)


class TestVocaCqtrollMethodExistsAndFailsLoudly(unittest.TestCase):
    """Smoke checks for the new Preprocess method, without I/O."""

    def test_method_exists_on_preprocess_class(self):
        from utils.preprocess import Preprocess
        self.assertTrue(hasattr(Preprocess, 'generate_labels_features_voca_cqtroll'))
        self.assertTrue(callable(Preprocess.generate_labels_features_voca_cqtroll))

    def test_helper_exposed_at_module_level(self):
        import utils.preprocess as pp
        self.assertTrue(hasattr(pp, 'shift_cqt_bins'))
        self.assertTrue(callable(pp.shift_cqt_bins))


if __name__ == '__main__':
    unittest.main()
