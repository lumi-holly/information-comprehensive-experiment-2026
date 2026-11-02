"""无需声卡或开发板的算法/异常帧验证，不代表真实网络性能测试。"""
import json
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import codec


class CodecTests(unittest.TestCase):
    def setUp(self):
        self.enc = bytes(range(16))
        self.mac = bytes(range(32))
        self.sid = "test-session"
        self.x = np.sin(np.arange(321) * 0.07) * 0.6

    def encode(self, source="dpcm8", channel="hamming74", **kwargs):
        return codec.encode_frame(self.x, 0, 16000, source, channel,
                                  self.enc, self.mac, self.sid, **kwargs)

    def decode(self, packet, **kwargs):
        return codec.decode_frame(packet, kwargs.get("enc", self.enc),
                                  kwargs.get("mac", self.mac), kwargs.get("sid", self.sid))

    def test_all_hamming_nibbles_all_single_errors(self):
        # 256 个字节覆盖所有 4 位信息组，逐个翻转码字的七种位置。
        original = bytes(range(256))
        encoded, bit_count = codec.hamming74_encode_bytes(original)
        for position in range(7):
            bits = np.unpackbits(np.frombuffer(encoded, dtype=np.uint8))[:bit_count].reshape(-1, 7)
            bits[:, position] ^= 1
            changed = np.packbits(bits.reshape(-1)).tobytes()
            recovered, syndromes = codec.hamming74_decode_bytes(changed, bit_count)
            self.assertEqual(recovered, original)
            self.assertEqual(syndromes, len(original) * 2)

    def test_hamming_byte_padding(self):
        for length in range(9):
            original = bytes(range(length))
            encoded, count = codec.hamming74_encode_bytes(original)
            decoded, syndromes = codec.hamming74_decode_bytes(encoded, count)
            self.assertEqual(decoded, original)
            self.assertEqual(syndromes, 0)
        with self.assertRaises(ValueError):
            codec.hamming74_decode_bytes(b"\0", 14)

    def test_dpcm_matches_independent_matlab_recurrence(self):
        samples = np.array([0, 1 / 127, -1 / 127, 0.9, -0.9, 1, -1, 0], dtype=float)
        expected_codes, expected_rebuilt = [], []
        previous = 0.0
        for value in samples:
            scaled = (value - previous) / (2 / 127)
            code = int(np.sign(scaled) * np.floor(abs(scaled) + 0.5))
            code = min(127, max(-127, code))
            expected_codes.append(code + 128)
            previous = min(1, max(-1, previous + code * (2 / 127)))
            expected_rebuilt.append(previous)
        encoded = codec.dpcm_encode(samples)
        self.assertEqual(encoded, bytes(expected_codes))
        np.testing.assert_array_equal(codec.dpcm_decode(encoded), expected_rebuilt)

    def test_all_mode_combinations(self):
        for source in codec.SOURCE_CODECS:
            for channel in codec.CHANNEL_CODECS:
                packet, stats = self.encode(source, channel)
                samples, meta, decoded_stats = self.decode(packet)
                self.assertEqual(samples.size, self.x.size)
                self.assertTrue(decoded_stats["authenticated"])
                self.assertEqual(meta["source_codec"], source)
                self.assertEqual(stats["source_bytes"], self.x.size * (1 if source == "dpcm8" else 2))
                if source == "dpcm8":
                    np.testing.assert_array_equal(samples, codec.dpcm_decode(codec.dpcm_encode(self.x)))
                else:
                    self.assertLess(np.max(np.abs(samples - self.x)), 1 / 32768)

    def test_random_iv(self):
        first, _ = self.encode()
        second, _ = self.encode()
        self.assertNotEqual(first, second)
        np.testing.assert_array_equal(self.decode(first)[0], self.decode(second)[0])

    def test_single_error_injection_hamming_recovers_none_rejects(self):
        packet, stats = self.encode(channel_test="single")
        _, _, result = self.decode(packet)
        self.assertEqual(stats["injected_bits"], 16)
        self.assertEqual(result["syndrome_codewords"], 16)
        packet, _ = self.encode(channel="none", channel_test="single")
        with self.assertRaisesRegex(ValueError, "HMAC"):
            self.decode(packet)

    def test_double_error_hamming_is_rejected_by_hmac(self):
        packet, _ = self.encode()
        meta_size = codec.META_PREFIX.unpack_from(packet)[1]
        offset = codec.META_PREFIX.size + meta_size
        changed = bytearray(packet)
        changed[offset] ^= 0xC0  # 同一 7 位码字中的两位错误。
        with self.assertRaisesRegex(ValueError, "HMAC"):
            self.decode(bytes(changed))

    def test_header_tampering(self):
        packet, _ = self.encode()
        old = b'"sequence":0'
        self.assertIn(old, packet)
        changed = packet.replace(old, b'"sequence":1', 1)
        with self.assertRaisesRegex(ValueError, "HMAC"):
            self.decode(changed)

    def test_invalid_packet_and_wrong_keys(self):
        packet, _ = self.encode()
        for changed in (b"", b"123", packet[:-1], packet + b"\0", packet[:10]):
            with self.assertRaises(ValueError):
                self.decode(changed)
        with self.assertRaises(ValueError):
            self.decode(packet, mac=b"x" * 32)
        with self.assertRaises(ValueError):
            self.decode(packet, sid="another-session")

    def test_reject_bad_inputs(self):
        for bad in (np.array([]), np.array([np.nan]), np.array([1.1])):
            with self.assertRaises(ValueError):
                codec.encode_frame(bad, 0, 16000, "dpcm8", "hamming74", self.enc, self.mac, self.sid)
        for kwargs in (dict(ber=0.03, channel_test="random"), dict(ber=0.01), dict(channel_test="bad")):
            with self.assertRaises(ValueError):
                self.encode(**kwargs)

    def test_boolean_sample_count_rejected(self):
        packet, _ = self.encode()
        meta_len = codec.META_PREFIX.unpack_from(packet)[1]
        meta = json.loads(packet[codec.META_PREFIX.size:codec.META_PREFIX.size + meta_len])
        meta["sample_count"] = True
        changed_meta = json.dumps(meta).encode()
        changed = codec.META_PREFIX.pack(codec.MAGIC, len(changed_meta)) + changed_meta + packet[codec.META_PREFIX.size + meta_len:]
        with self.assertRaises(ValueError):
            self.decode(changed)


if __name__ == "__main__":
    unittest.main()
