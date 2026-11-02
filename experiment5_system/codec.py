"""实验五：信源编码、数字加密、信道编码，以及接收端的逆过程。

每帧独立初始化 DPCM 预测器，因此断开前一帧也不会污染后一帧的状态。
AES-CBC 配合 HMAC 验证帧头和密文；先信道译码、再验签、最后解密。
HMAC 密钥必须与 AES 密钥分开，由调用方加载，不在源码中保存密钥。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import struct
import time

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7

MAGIC = b"EX5A"
META_PREFIX = struct.Struct("!4sI")
MAX_PACKET_BYTES = 1024 * 1024  # 接收缓冲上限，工程建议值，不是课件指标。
MAX_META_BYTES = 4096
MAX_FRAME_SAMPLES = 48000
MAX_SEQUENCE = 2 ** 32 - 1
SOURCE_CODECS = ("dpcm8", "pcm16")
CHANNEL_CODECS = ("hamming74", "none")
CHANNEL_TESTS = ("none", "single", "random")


def _strict_int(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("%s 必须是 %d..%d 范围内的整数。" % (name, lower, upper))
    return value


def _keys(enc_key, mac_key):
    if not isinstance(enc_key, bytes) or len(enc_key) != 16:
        raise ValueError("AES-128 密钥必须是 16 字节。")
    if not isinstance(mac_key, bytes) or len(mac_key) != 32:
        raise ValueError("HMAC 密钥必须是 32 字节。")


def _session(value):
    if (not isinstance(value, str) or not 1 <= len(value) <= 128
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in value)):
        raise ValueError("session_id 必须是 1..128 个字母、数字、横线、下划线或点。")
    return value


def _samples(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not 1 <= values.size <= MAX_FRAME_SAMPLES:
        raise ValueError("音频帧必须是非空一维数组，最多 48000 个样本。")
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1.0):
        raise ValueError("音频必须是 [-1,1] 内的有限浮点数。")
    return values


def dpcm_encode(samples):
    """兼容实验三：8 bit、delta=2/127、以重建值预测、半整数远离零取整。"""
    values = _samples(samples)
    codes = np.empty(values.size, dtype=np.uint8)
    previous = 0.0
    delta = 2.0 / 127
    # 算法有递推关系，不能把这个循环直接改成独立样本的向量计算。
    for i, sample in enumerate(values):
        scaled = (float(sample) - previous) / delta
        code = int(math.copysign(math.floor(abs(scaled) + 0.5), scaled))
        code = max(-127, min(127, code))
        codes[i] = code + 128
        previous = max(-1.0, min(1.0, previous + code * delta))
    return codes.tobytes()


def dpcm_decode(data):
    """独立解码，不依赖发送端的原始样本或预测器状态。"""
    if not isinstance(data, bytes) or not 1 <= len(data) <= MAX_FRAME_SAMPLES:
        raise ValueError("DPCM 数据长度不正确。")
    codes = np.frombuffer(data, dtype=np.uint8)
    if np.any(codes == 0):
        raise ValueError("DPCM 码字 0 不在实验三使用的 [1,255] 编码范围内。")
    decoded = np.empty(codes.size, dtype=np.float64)
    previous = 0.0
    delta = 2.0 / 127
    for i, code in enumerate(codes):
        previous = max(-1.0, min(1.0, previous + (int(code) - 128) * delta))
        decoded[i] = previous
    return decoded


def hamming74_encode_bytes(data):
    """每字节分为两个 4 bit 信息组，输出 [p1,p2,d1,p4,d2,d3,d4]。"""
    if not isinstance(data, bytes) or len(data) > MAX_PACKET_BYTES // 2:
        raise ValueError("汉明编码输入长度不正确。")
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")
    groups = bits.astype(bool).reshape(-1, 4)
    d1, d2, d3, d4 = groups.T
    coded = np.column_stack((d1 ^ d2 ^ d4, d1 ^ d3 ^ d4, d1,
                             d2 ^ d3 ^ d4, d2, d3, d4)).reshape(-1)
    return np.packbits(coded, bitorder="big").tobytes(), int(coded.size)


def hamming74_decode_bytes(data, coded_bit_count):
    """返回恢复字节、非零综合症码字数。该计数不能单独证明纠错成功。"""
    if not isinstance(data, bytes) or len(data) > MAX_PACKET_BYTES:
        raise ValueError("汉明译码输入长度不正确。")
    _strict_int(coded_bit_count, "coded_bits", 0, MAX_PACKET_BYTES * 8)
    if coded_bit_count % 14 or len(data) != (coded_bit_count + 7) // 8:
        raise ValueError("汉明码有效位数与字节长度不一致。")
    all_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")
    if np.any(all_bits[coded_bit_count:]):
        raise ValueError("汉明码尾部填充位必须为零。")
    received = all_bits[:coded_bit_count].astype(bool).reshape(-1, 7)
    s1 = received[:, 0] ^ received[:, 2] ^ received[:, 4] ^ received[:, 6]
    s2 = received[:, 1] ^ received[:, 2] ^ received[:, 5] ^ received[:, 6]
    s4 = received[:, 3] ^ received[:, 4] ^ received[:, 5] ^ received[:, 6]
    syndrome = s1.astype(np.uint8) + 2 * s2.astype(np.uint8) + 4 * s4.astype(np.uint8)
    corrected = received.copy()
    rows = np.flatnonzero(syndrome)
    corrected[rows, syndrome[rows] - 1] ^= True
    decoded = corrected[:, [2, 4, 5, 6]].reshape(-1)
    return np.packbits(decoded, bitorder="big").tobytes(), int(rows.size)


def _test_positions(bit_count, channel_codec, channel_test, ber, sequence):
    if channel_test not in CHANNEL_TESTS:
        raise ValueError("channel_test 只能是 none、single 或 random。")
    if type(ber) not in (int, float) or not 0 <= ber <= 0.02 or not math.isfinite(ber):
        raise ValueError("模拟 BER 必须位于 0..0.02（本实现建议范围，非课件阈值）。")
    if channel_test != "random" and ber != 0:
        raise ValueError("只有 random 模拟模式可以指定非零 BER。")
    if channel_test == "none":
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(2026 + sequence)
    if channel_test == "single":
        # 每组最多 1 位，保证汉明码可纠正；none 模式保留同样的故障位置作对照。
        count = min(16, bit_count // 7)
        return np.arange(count, dtype=np.int64) * 7 + rng.integers(0, 7, size=count)
    return np.flatnonzero(rng.random(bit_count) < ber)


def encode_frame(samples, sequence, sample_rate, source_codec, channel_codec,
                 enc_key, mac_key, session_id, channel_test="none", ber=0.0):
    """返回 (网络帧 bytes, 性能统计 dict)。样本是归一化 float64，不是 int16。"""
    _keys(enc_key, mac_key)
    _session(session_id)
    _strict_int(sequence, "sequence", 0, MAX_SEQUENCE)
    _strict_int(sample_rate, "sample_rate", 8000, 48000)
    if source_codec not in SOURCE_CODECS or channel_codec not in CHANNEL_CODECS:
        raise ValueError("未知信源编码或信道编码方式。")
    values = _samples(samples)
    started = time.perf_counter()
    if source_codec == "dpcm8":
        source = dpcm_encode(values)
    else:
        source = np.clip(np.rint(values * 32768), -32768, 32767).astype("<i2").tobytes()
    source_ms = (time.perf_counter() - started) * 1000

    # CBC 加密后长度只依赖输入长度，可先确定元数据和故障位置，再计算 HMAC。
    encrypted_size = 16 + (len(source) // 16 + 1) * 16 + 32
    bit_count = encrypted_size * (14 if channel_codec == "hamming74" else 8)
    positions = _test_positions(bit_count, channel_codec, channel_test, ber, sequence)
    meta = dict(version=1, session_id=session_id, sequence=sequence,
                sample_rate=sample_rate, sample_count=int(values.size),
                source_codec=source_codec, channel_codec=channel_codec,
                coded_bits=bit_count, encrypted_bytes=encrypted_size,
                channel_test=channel_test, ber=float(ber), injected_bits=int(positions.size))
    meta_bytes = json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    started = time.perf_counter()
    iv = os.urandom(16)
    padder = PKCS7(128).padder()
    padded = padder.update(source) + padder.finalize()
    encryptor = Cipher(algorithms.AES(enc_key), modes.CBC(iv)).encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()
    encrypted = iv + ciphertext
    tag = hmac.new(mac_key, meta_bytes + encrypted, hashlib.sha256).digest()
    encrypted += tag
    encrypt_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    if channel_codec == "hamming74":
        coded, actual_bits = hamming74_encode_bytes(encrypted)
        assert actual_bits == bit_count
    else:
        coded = encrypted
    channel_ms = (time.perf_counter() - started) * 1000
    if positions.size:
        bits = np.unpackbits(np.frombuffer(coded, dtype=np.uint8), bitorder="big")
        bits[positions] ^= 1
        coded = np.packbits(bits, bitorder="big").tobytes()
    packet = META_PREFIX.pack(MAGIC, len(meta_bytes)) + meta_bytes + coded
    return packet, dict(source_encode_ms=source_ms, encrypt_ms=encrypt_ms,
                        channel_encode_ms=channel_ms, source_bytes=len(source),
                        encrypted_bytes=len(encrypted), coded_bytes=len(coded),
                        coded_bits=bit_count, injected_bits=int(positions.size),
                        packet_bytes=len(packet))


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("帧头存在重复字段。")
        result[key] = value
    return result


def _read_meta(packet, expected_session):
    if not isinstance(packet, bytes) or not META_PREFIX.size < len(packet) <= MAX_PACKET_BYTES:
        raise ValueError("网络帧长度不正确。")
    magic, meta_length = META_PREFIX.unpack_from(packet)
    if magic != MAGIC or not 1 <= meta_length <= MAX_META_BYTES or len(packet) <= META_PREFIX.size + meta_length:
        raise ValueError("网络帧标识或元数据长度不正确。")
    meta_bytes = packet[META_PREFIX.size:META_PREFIX.size + meta_length]
    try:
        meta = json.loads(meta_bytes.decode("utf-8"), object_pairs_hook=_unique_pairs)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("帧头不是有效的 UTF-8 JSON。") from exc
    expected_fields = {"version", "session_id", "sequence", "sample_rate", "sample_count",
                       "source_codec", "channel_codec", "coded_bits", "encrypted_bytes",
                       "channel_test", "ber", "injected_bits"}
    if not isinstance(meta, dict) or set(meta) != expected_fields:
        raise ValueError("帧头字段不完整或包含未知字段。")
    _strict_int(meta["version"], "version", 1, 1)
    _session(meta["session_id"])
    if meta["session_id"] != expected_session:
        raise ValueError("帧的 session_id 与当前会话不一致。")
    _strict_int(meta["sequence"], "sequence", 0, MAX_SEQUENCE)
    _strict_int(meta["sample_rate"], "sample_rate", 8000, 48000)
    _strict_int(meta["sample_count"], "sample_count", 1, MAX_FRAME_SAMPLES)
    _strict_int(meta["encrypted_bytes"], "encrypted_bytes", 64, MAX_PACKET_BYTES // 2)
    _strict_int(meta["coded_bits"], "coded_bits", 1, MAX_PACKET_BYTES * 8)
    _strict_int(meta["injected_bits"], "injected_bits", 0, MAX_PACKET_BYTES * 8)
    if meta["source_codec"] not in SOURCE_CODECS or meta["channel_codec"] not in CHANNEL_CODECS:
        raise ValueError("帧头包含未知编解码方式。")
    if meta["channel_test"] not in CHANNEL_TESTS:
        raise ValueError("帧头包含未知模拟模式。")
    ber = meta["ber"]
    if type(ber) not in (int, float) or not 0 <= ber <= 0.02 or not math.isfinite(ber):
        raise ValueError("帧头的模拟 BER 无效。")
    if meta["channel_test"] != "random" and ber != 0:
        raise ValueError("非随机模拟模式的 BER 必须为零。")
    source_size = meta["sample_count"] * (1 if meta["source_codec"] == "dpcm8" else 2)
    encrypted_size = 16 + (source_size // 16 + 1) * 16 + 32
    bit_count = encrypted_size * (14 if meta["channel_codec"] == "hamming74" else 8)
    coded = packet[META_PREFIX.size + meta_length:]
    if meta["encrypted_bytes"] != encrypted_size or meta["coded_bits"] != bit_count or len(coded) != (bit_count + 7) // 8:
        raise ValueError("帧头长度与样本数、编码方式不一致。")
    if meta["injected_bits"] > bit_count:
        raise ValueError("模拟翻转位数超过帧有效位数。")
    return meta_bytes, meta, coded


def decode_frame(packet, enc_key, mac_key, expected_session):
    """验签成功后才返回语音。HMAC 失败、截断、错误密钥均抛出 ValueError。"""
    _keys(enc_key, mac_key)
    _session(expected_session)
    meta_bytes, meta, coded = _read_meta(packet, expected_session)
    started = time.perf_counter()
    if meta["channel_codec"] == "hamming74":
        encrypted, syndrome_count = hamming74_decode_bytes(coded, meta["coded_bits"])
    else:
        encrypted, syndrome_count = coded, 0
    channel_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    protected, tag = encrypted[:-32], encrypted[-32:]
    expected_tag = hmac.new(mac_key, meta_bytes + protected, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected_tag):
        raise ValueError("HMAC 校验失败：错误密钥、帧损坏或汉明码无法纠正的多比特错误。")
    iv, ciphertext = protected[:16], protected[16:]
    try:
        decryptor = Cipher(algorithms.AES(enc_key), modes.CBC(iv)).decryptor()
        padded = decryptor.update(ciphertext) + decryptor.finalize()
        unpadder = PKCS7(128).unpadder()
        source = unpadder.update(padded) + unpadder.finalize()
    except ValueError as exc:
        raise ValueError("AES 解密或填充验证失败，请检查密钥。") from exc
    decrypt_ms = (time.perf_counter() - started) * 1000
    expected_size = meta["sample_count"] * (1 if meta["source_codec"] == "dpcm8" else 2)
    if len(source) != expected_size:
        raise ValueError("解密后的信源数据长度与样本数不一致。")
    started = time.perf_counter()
    if meta["source_codec"] == "dpcm8":
        samples = dpcm_decode(source)
    else:
        samples = np.frombuffer(source, dtype="<i2").astype(np.float64) / 32768.0
    source_ms = (time.perf_counter() - started) * 1000
    return samples, meta, dict(channel_decode_ms=channel_ms, decrypt_ms=decrypt_ms,
                               source_decode_ms=source_ms, syndrome_codewords=syndrome_count,
                               authenticated=True)
