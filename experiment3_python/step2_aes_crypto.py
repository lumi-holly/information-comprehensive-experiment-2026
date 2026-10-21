from __future__ import annotations

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from common import (
    AES_IV,
    AES_KEY,
    RESULT_DIR,
    aes_decrypt,
    aes_encrypt,
    dpcm_decode,
    ensure_result_dir,
    set_chinese_plot_style,
)


def run() -> dict[str, float | int | bool]:
    ensure_result_dir()
    set_chinese_plot_style()
    input_file = RESULT_DIR / "dpcm_encoded_data.npz"
    if not input_file.exists():
        raise FileNotFoundError("找不到第一步数据，请先运行 step1_source_coding.py。")

    data = np.load(input_file)
    src_bytes = data["src_bytes"].astype(np.uint8)
    original_audio = data["original_audio"].astype(np.float64)
    fs = int(data["fs"])
    delta = float(data["delta"])

    start = perf_counter()
    cipher_raw = aes_encrypt(src_bytes.tobytes())
    encryption_time = perf_counter() - start
    start = perf_counter()
    plain_raw = aes_decrypt(cipher_raw)
    decryption_time = perf_counter() - start

    cipher_bytes = np.frombuffer(cipher_raw, dtype=np.uint8).copy()
    plain_bytes = np.frombuffer(plain_raw, dtype=np.uint8).copy()
    different_byte_count = int(np.count_nonzero(src_bytes != plain_bytes))
    is_correct = np.array_equal(src_bytes, plain_bytes)

    changed_plain = src_bytes.copy()
    changed_plain[0] ^= np.uint8(1)
    changed_cipher = np.frombuffer(aes_encrypt(changed_plain.tobytes()), dtype=np.uint8)
    changed_bit_count = int(np.unpackbits(cipher_bytes ^ changed_cipher).sum())
    avalanche_rate = 100 * changed_bit_count / (cipher_bytes.size * 8)

    recovered_audio = dpcm_decode(plain_bytes, delta)
    error = original_audio - recovered_audio
    end_to_end_mse = float(np.mean(error**2))
    end_to_end_snr = float(10 * np.log10(np.sum(original_audio**2) / (np.sum(error**2) + np.finfo(float).eps)))

    print("===== AES-128-CBC加密与解密结果 =====")
    print(f"密钥长度：{len(AES_KEY) * 8} bit")
    print(f"加密前长度：{src_bytes.size} bytes")
    print(f"密文长度：{cipher_bytes.size} bytes")
    print(f"加密时间：{encryption_time:.6f} s")
    print(f"解密时间：{decryption_time:.6f} s")
    print(f"不同字节数量：{different_byte_count}")
    print(f"解密数据与原始DPCM数据是否完全一致：{int(is_correct)}")
    print("===== AES雪崩效应 =====")
    print(f"密文改变：{changed_bit_count} bit")
    print(f"密文比特变化率：{avalanche_rate:.2f} %")
    print(f"端到端 MSE：{end_to_end_mse:.12f}")
    print(f"端到端 SNR：{end_to_end_snr:.8f} dB")
    if not is_correct:
        raise RuntimeError("AES解密数据与原始数据不一致。")

    np.savez_compressed(
        RESULT_DIR / "aes_encrypted_data.npz",
        cipher_bytes=cipher_bytes,
        iv_bytes=np.frombuffer(AES_IV, dtype=np.uint8),
        encryption_time=encryption_time,
        decryption_time=decryption_time,
        is_correct=is_correct,
        changed_bit_count=changed_bit_count,
        avalanche_rate=avalanche_rate,
    )
    wavfile.write(
        RESULT_DIR / "aes_decrypted_recovered.wav",
        fs,
        np.round(np.clip(recovered_audio, -1, 1) * 32767).astype(np.int16),
    )

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    axes[0].hist(src_bytes, bins=np.arange(257) - 0.5, color=(0.1, 0.45, 0.85))
    axes[0].set(title="AES加密前：DPCM编码字节分布", xlabel="字节值", ylabel="出现次数", xlim=(0, 255))
    axes[1].hist(cipher_bytes, bins=np.arange(257) - 0.5, color=(0.85, 0.2, 0.2))
    axes[1].set(title="AES加密后：密文字节分布", xlabel="字节值", ylabel="出现次数", xlim=(0, 255))
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.savefig(RESULT_DIR / "03_AES加密前后字节分布.png", dpi=180)
    plt.close(fig)

    print("第二步 AES 数据加密与解密完成。\n")
    return {
        "plain_bytes": src_bytes.size,
        "cipher_bytes": cipher_bytes.size,
        "correct": is_correct,
        "avalanche_rate": avalanche_rate,
        "encryption_time": encryption_time,
        "decryption_time": decryption_time,
    }


if __name__ == "__main__":
    run()

