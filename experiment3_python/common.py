from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = BASE_DIR.parent
_LOCAL_INPUT_WAV = BASE_DIR / "original.wav"
_LEGACY_INPUT_WAV = EXPERIMENT_DIR.parent / "实验一" / "original.wav"
INPUT_WAV = Path(os.environ.get("EXPERIMENT_AUDIO", _LOCAL_INPUT_WAV))
if "EXPERIMENT_AUDIO" not in os.environ and not INPUT_WAV.exists():
    INPUT_WAV = _LEGACY_INPUT_WAV
RESULT_DIR = BASE_DIR / "python_results"

AES_KEY = b"Exp3AESKey123456"
AES_IV = bytes(range(16))


def ensure_result_dir() -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR


def matlab_round(values: np.ndarray | float) -> np.ndarray:
    """实现 MATLAB round：0.5 向远离 0 的方向取整。"""
    arr = np.asarray(values, dtype=np.float64)
    return np.sign(arr) * np.floor(np.abs(arr) + 0.5)


def dpcm_decode(src_bytes: np.ndarray, delta: float) -> np.ndarray:
    decoded = np.zeros(src_bytes.size, dtype=np.float64)
    previous = 0.0
    for index, byte_value in enumerate(src_bytes):
        quantized_code = int(byte_value) - 128
        current = previous + quantized_code * delta
        current = float(np.clip(current, -1.0, 1.0))
        decoded[index] = current
        previous = current
    return decoded


def aes_encrypt(plain_bytes: bytes) -> bytes:
    padder = PKCS7(algorithms.AES.block_size).padder()
    padded = padder.update(plain_bytes) + padder.finalize()
    encryptor = Cipher(algorithms.AES(AES_KEY), modes.CBC(AES_IV)).encryptor()
    return encryptor.update(padded) + encryptor.finalize()


def aes_decrypt(cipher_bytes: bytes) -> bytes:
    decryptor = Cipher(algorithms.AES(AES_KEY), modes.CBC(AES_IV)).decryptor()
    padded = decryptor.update(cipher_bytes) + decryptor.finalize()
    unpadder = PKCS7(algorithms.AES.block_size).unpadder()
    return unpadder.update(padded) + unpadder.finalize()


def bytes_to_bits(byte_array: np.ndarray) -> np.ndarray:
    return np.unpackbits(byte_array.astype(np.uint8), bitorder="big").astype(bool)


def bits_to_bytes(bits: np.ndarray) -> np.ndarray:
    if bits.size % 8 != 0:
        raise ValueError("比特数必须是 8 的整数倍。")
    return np.packbits(bits.astype(np.uint8), bitorder="big")


def hamming74_encode(input_bits: np.ndarray) -> tuple[np.ndarray, int]:
    padding_count = (-input_bits.size) % 4
    padded = np.pad(input_bits.astype(bool), (0, padding_count), constant_values=False)
    data = padded.reshape(-1, 4)
    d1, d2, d3, d4 = data.T
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p4 = d2 ^ d3 ^ d4
    coded = np.column_stack((p1, p2, d1, p4, d2, d3, d4))
    return coded, padding_count


def hamming74_decode(received: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    received = received.astype(bool, copy=False)
    syndrome1 = received[:, 0] ^ received[:, 2] ^ received[:, 4] ^ received[:, 6]
    syndrome2 = received[:, 1] ^ received[:, 2] ^ received[:, 5] ^ received[:, 6]
    syndrome4 = received[:, 3] ^ received[:, 4] ^ received[:, 5] ^ received[:, 6]
    syndrome = (
        syndrome1.astype(np.uint8)
        + 2 * syndrome2.astype(np.uint8)
        + 4 * syndrome4.astype(np.uint8)
    )

    corrected = received.copy()
    rows = np.flatnonzero(syndrome)
    corrected[rows, syndrome[rows] - 1] ^= True
    decoded = corrected[:, [2, 4, 5, 6]].reshape(-1)
    return decoded, corrected, syndrome


def set_chinese_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

