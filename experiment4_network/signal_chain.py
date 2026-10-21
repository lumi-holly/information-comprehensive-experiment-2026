from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7


MAGIC = b"EX4P"
PACKET_HEADER = struct.Struct("!4sIIIII32s")
DEFAULT_KEY = b"Exp4AESKey123456"


@dataclass(frozen=True)
class DecodedFrame:
    sequence: int
    sample_rate: int
    samples: np.ndarray
    corrected_codewords: int


def _matlab_round(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def dpcm_encode(samples: np.ndarray, quant_bits: int = 8) -> bytes:
    """把单声道 int16 PCM 编码为 8 bit DPCM 字节。每个网络帧独立预测。"""
    normalized = np.asarray(samples, dtype=np.float64).reshape(-1) / 32768.0
    max_code = 2 ** (quant_bits - 1) - 1
    delta = 2.0 / max_code
    encoded = np.empty(normalized.size, dtype=np.uint8)
    previous = 0.0
    for index, sample in enumerate(normalized):
        code = int(np.clip(_matlab_round((sample - previous) / delta), -max_code, max_code))
        encoded[index] = code + 128
        previous = float(np.clip(previous + code * delta, -1.0, 1.0))
    return encoded.tobytes()


def dpcm_decode(data: bytes, quant_bits: int = 8) -> np.ndarray:
    max_code = 2 ** (quant_bits - 1) - 1
    delta = 2.0 / max_code
    encoded = np.frombuffer(data, dtype=np.uint8)
    decoded = np.empty(encoded.size, dtype=np.float64)
    previous = 0.0
    for index, byte_value in enumerate(encoded):
        current = previous + (int(byte_value) - 128) * delta
        current = float(np.clip(current, -1.0, 1.0))
        decoded[index] = current
        previous = current
    return np.round(decoded * 32767).astype(np.int16)


def aes_encrypt(data: bytes, key: bytes = DEFAULT_KEY) -> bytes:
    iv = np.random.default_rng().bytes(16)
    padder = PKCS7(algorithms.AES.block_size).padder()
    padded = padder.update(data) + padder.finalize()
    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    return iv + encryptor.update(padded) + encryptor.finalize()


def aes_decrypt(data: bytes, key: bytes = DEFAULT_KEY) -> bytes:
    if len(data) < 32 or len(data) % 16 != 0:
        raise ValueError("AES 数据长度不正确。")
    iv, ciphertext = data[:16], data[16:]
    decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = PKCS7(algorithms.AES.block_size).unpadder()
    return unpadder.update(padded) + unpadder.finalize()


def hamming74_encode_bytes(data: bytes) -> tuple[bytes, int]:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big").astype(bool)
    groups = bits.reshape(-1, 4)
    d1, d2, d3, d4 = groups.T
    coded = np.column_stack((d1 ^ d2 ^ d4, d1 ^ d3 ^ d4, d1, d2 ^ d3 ^ d4, d2, d3, d4))
    coded_bits = coded.reshape(-1)
    packed = np.packbits(coded_bits.astype(np.uint8), bitorder="big")
    return packed.tobytes(), int(coded_bits.size)


def hamming74_decode_bytes(data: bytes, coded_bit_count: int) -> tuple[bytes, int]:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")[:coded_bit_count]
    if bits.size % 7:
        raise ValueError("汉明码比特数不是 7 的整数倍。")
    received = bits.astype(bool).reshape(-1, 7)
    s1 = received[:, 0] ^ received[:, 2] ^ received[:, 4] ^ received[:, 6]
    s2 = received[:, 1] ^ received[:, 2] ^ received[:, 5] ^ received[:, 6]
    s4 = received[:, 3] ^ received[:, 4] ^ received[:, 5] ^ received[:, 6]
    syndrome = s1.astype(np.uint8) + 2 * s2.astype(np.uint8) + 4 * s4.astype(np.uint8)
    corrected = received.copy()
    rows = np.flatnonzero(syndrome)
    corrected[rows, syndrome[rows] - 1] ^= True
    decoded = corrected[:, [2, 4, 5, 6]].reshape(-1)
    return np.packbits(decoded.astype(np.uint8), bitorder="big").tobytes(), int(rows.size)


def simulate_channel(data: bytes, coded_bit_count: int, bit_error_rate: float, seed: int) -> bytes:
    if not 0.0 <= bit_error_rate <= 1.0:
        raise ValueError("误码率必须位于 0 到 1 之间。")
    if bit_error_rate == 0:
        return data
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")
    rng = np.random.default_rng(seed)
    bits[:coded_bit_count] ^= (rng.random(coded_bit_count) < bit_error_rate).astype(np.uint8)
    return np.packbits(bits, bitorder="big").tobytes()


def encode_frame(
    sequence: int,
    sample_rate: int,
    samples: np.ndarray,
    bit_error_rate: float = 0.0,
) -> bytes:
    dpcm = dpcm_encode(samples)
    encrypted = aes_encrypt(dpcm)
    digest = hashlib.sha256(encrypted).digest()
    coded, coded_bit_count = hamming74_encode_bytes(encrypted)
    coded = simulate_channel(coded, coded_bit_count, bit_error_rate, seed=2026 + sequence)
    header = PACKET_HEADER.pack(
        MAGIC,
        sequence,
        sample_rate,
        len(samples),
        coded_bit_count,
        len(coded),
        digest,
    )
    return header + coded


def decode_frame(packet: bytes) -> DecodedFrame:
    if len(packet) < PACKET_HEADER.size:
        raise ValueError("网络帧过短。")
    magic, sequence, sample_rate, sample_count, coded_bit_count, coded_length, digest = (
        PACKET_HEADER.unpack(packet[: PACKET_HEADER.size])
    )
    if magic != MAGIC:
        raise ValueError("网络帧标识不正确。")
    coded = packet[PACKET_HEADER.size :]
    if len(coded) != coded_length:
        raise ValueError("网络帧长度不正确。")
    encrypted, corrected = hamming74_decode_bytes(coded, coded_bit_count)
    if hashlib.sha256(encrypted).digest() != digest:
        raise ValueError("帧校验失败，可能存在同一码字内的多比特错误。")
    dpcm = aes_decrypt(encrypted)
    if len(dpcm) != sample_count:
        raise ValueError("DPCM 样本数与帧头不一致。")
    return DecodedFrame(sequence, sample_rate, dpcm_decode(dpcm), corrected)

