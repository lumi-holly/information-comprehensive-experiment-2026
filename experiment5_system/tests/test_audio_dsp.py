"""音频 DSP 重要性质验证：连续性、频率选择性、量化与延迟评价。"""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from audio_dsp import (FIRFilter, add_tone, analyze, design_bandpass, metrics,
                       read_wav, save_analysis_csv, write_wav)


class AudioDSPTests(unittest.TestCase):
    def test_filter_block_continuity(self):
        coefficients = design_bandpass(16000)
        samples = np.random.default_rng(2026).normal(size=1200)
        filtering = FIRFilter(coefficients)
        chunks = [samples[:1], samples[1:17], samples[17:23],
                  np.empty(0), samples[23:800], samples[800:]]
        actual = np.concatenate([filtering.process(chunk) for chunk in chunks])
        expected = np.convolve(samples, coefficients)[:len(samples)]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(filtering.delay_samples, 64)

    def test_filter_suppresses_out_of_band_tone(self):
        coefficients = design_bandpass(16000)
        points = np.arange(coefficients.size)
        gain = lambda frequency: abs(np.sum(coefficients * np.exp(
            -2j * np.pi * frequency * points / 16000)))
        self.assertGreater(gain(1000), 0.95)
        self.assertLess(gain(5000), 0.01)
        self.assertLess(gain(0), 0.02)
        np.testing.assert_allclose(coefficients, coefficients[::-1], atol=1e-14)

    def test_add_tone_preserves_chunk_phase(self):
        samples = np.ones(500) * 0.95
        complete = add_tone(samples, 16000)
        chunked = np.concatenate((add_tone(samples[:117], 16000),
                                  add_tone(samples[117:], 16000, start_sample=117)))
        np.testing.assert_array_equal(complete, chunked)
        self.assertGreater(np.max(complete), 1)  # 不暗中截幅或归一化。

    def test_invalid_design_and_nyquist(self):
        for kwargs in ({"low": 0}, {"high": 8000}, {"low": 4000, "high": 3000},
                       {"taps": 128}, {"taps": 1}):
            with self.assertRaises(ValueError):
                design_bandpass(16000, **kwargs)
        with self.assertRaises(ValueError):
            add_tone(np.zeros(16), 8000, frequency=5000)

    def test_pcm_io_full_scale_and_stereo(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mono.wav"
            samples = np.array([-2., -1., -.5, 0., .5, 1., 2.])
            write_wav(path, samples, 16000)
            recovered, fs = read_wav(path)
            np.testing.assert_array_equal(recovered,
                [-1, -1, -.5, 0, .5, 32767 / 32768, 32767 / 32768])
            self.assertEqual(fs, 16000)
            stereo = np.array([[100, 300], [-200, 0]], dtype="<i2")
            with wave.open(str(path), "wb") as writer:
                writer.setnchannels(2)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(stereo.tobytes())
            np.testing.assert_array_equal(read_wav(path)[0], np.array([200, -100]) / 32768)

    def test_metrics_delay_and_json_finiteness(self):
        samples = np.arange(20, dtype=float)
        delayed = np.concatenate((np.zeros(3), samples[:-3]))
        result = metrics(samples, delayed, delay_samples=3)
        self.assertEqual(result["mse"], 0)
        self.assertTrue(result["exact_match"])
        self.assertIsNone(result["snr_db"])
        self.assertEqual(result["compared_samples"], 17)
        json.dumps(result, allow_nan=False)
        self.assertEqual(metrics(np.ones(20), np.ones(20) + .1)["snr_status"], "finite")
        self.assertIsNone(metrics(np.zeros(20), np.ones(20))["snr_db"])
        with self.assertRaises(ValueError):
            metrics(np.zeros(3), np.zeros(4))

    def test_analysis_and_csv_boundaries(self):
        fs = 16000
        samples = np.sin(2 * np.pi * 1000 * np.arange(40000) / fs)
        result = analyze(samples, fs)
        self.assertAlmostEqual(result["fft_peak_hz"], 1000)
        self.assertEqual(result["fft_samples"], 32768)
        self.assertTrue(result["fft_uses_prefix_subset"])
        with tempfile.TemporaryDirectory() as temporary:
            paths = save_analysis_csv(Path(temporary) / "voice", samples, fs)
            with open(paths["time_csv"], encoding="utf-8", newline="") as handle:
                self.assertLessEqual(len(list(csv.reader(handle))) - 1, 4000)
            with open(paths["spectrum_csv"], encoding="utf-8", newline="") as handle:
                rows = csv.reader(handle)
                self.assertIn("prefix_subset", next(rows))
                self.assertEqual(next(rows)[3], "True")


if __name__ == "__main__":
    unittest.main()
