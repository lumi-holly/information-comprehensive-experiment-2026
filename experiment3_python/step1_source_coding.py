from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from common import (
    INPUT_WAV,
    RESULT_DIR,
    dpcm_decode,
    ensure_result_dir,
    matlab_round,
    set_chinese_plot_style,
)


def read_audio() -> tuple[int, np.ndarray]:
    if not INPUT_WAV.exists():
        raise FileNotFoundError(f"找不到语音文件：{INPUT_WAV}")
    fs, raw = wavfile.read(INPUT_WAV)
    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        x = raw.astype(np.float64) / max(abs(info.min), info.max)
    else:
        x = raw.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    peak = np.max(np.abs(x))
    if peak > 0:
        x /= peak
    return int(fs), x.reshape(-1)


def run() -> dict[str, float | int | bool]:
    ensure_result_dir()
    set_chinese_plot_style()
    fs, x = read_audio()

    quant_bits = 8
    max_code = 2 ** (quant_bits - 1) - 1
    delta = 2.0 / max_code
    src_bytes = np.zeros(x.size, dtype=np.uint8)
    encoder_reconstructed = np.zeros(x.size, dtype=np.float64)

    previous = 0.0
    for index, sample in enumerate(x):
        prediction_error = sample - previous
        code = int(np.clip(matlab_round(prediction_error / delta), -max_code, max_code))
        src_bytes[index] = code + 128
        current = float(np.clip(previous + code * delta, -1.0, 1.0))
        encoder_reconstructed[index] = current
        previous = current

    decoded = dpcm_decode(src_bytes, delta)
    decoder_consistent = np.array_equal(encoder_reconstructed, decoded)
    error = x - decoded
    mse = float(np.mean(error**2))
    snr = float(10 * np.log10(np.sum(x**2) / (np.sum(error**2) + np.finfo(float).eps)))
    compression_ratio = 16 / quant_bits

    print("===== 原始语音信息 =====")
    print(f"采样率：{fs} Hz")
    print(f"样本数：{x.size}")
    print(f"语音时长：{x.size / fs:.2f} s\n")
    print("===== DPCM编码与解码结果 =====")
    print(f"量化位数：{quant_bits} bit")
    print(f"量化步长：{delta:.8f}")
    print(f"均方误差 MSE：{mse:.12f}")
    print(f"信噪比 SNR：{snr:.8f} dB")
    print(f"压缩比：{compression_ratio:.2f} : 1")
    print(f"编码器重建结果与独立解码结果是否一致：{int(decoder_consistent)}")
    if not decoder_consistent:
        raise RuntimeError("编码器和解码器的重建结果不一致。")

    np.savez_compressed(
        RESULT_DIR / "dpcm_encoded_data.npz",
        src_bytes=src_bytes,
        fs=fs,
        quant_bits=quant_bits,
        delta=delta,
        sample_count=x.size,
        original_audio=x,
        decoded_audio=decoded,
        mse=mse,
        snr=snr,
        compression_ratio=compression_ratio,
    )
    wavfile.write(
        RESULT_DIR / "dpcm_recovered.wav",
        fs,
        np.round(np.clip(decoded, -1, 1) * 32767).astype(np.int16),
    )

    show_count = max(1, min(int(0.05 * fs), x.size))
    peak_index = int(np.argmax(np.abs(x)))
    start = max(0, peak_index - show_count // 2)
    start = min(start, x.size - show_count)
    end = start + show_count
    t_short = np.arange(start, end) / fs
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    axes[0].plot(t_short, x[start:end], "b", linewidth=0.7)
    axes[0].set(title="原始语音波形（高能量50 ms片段）", ylabel="幅值")
    axes[1].plot(t_short, decoded[start:end], "r", linewidth=0.7)
    axes[1].set(title="DPCM解码语音波形（同一50 ms片段）", xlabel="时间 / s", ylabel="幅值")
    for axis in axes:
        axis.grid(True, alpha=0.35)
    fig.savefig(RESULT_DIR / "01_DPCM时域波形对比.png", dpi=180)
    plt.close(fig)

    fft_length = 2 ** int(np.ceil(np.log2(x.size)))
    frequency = np.arange(fft_length // 2) * fs / fft_length
    original_spectrum = np.abs(np.fft.fft(x, fft_length))[: fft_length // 2]
    decoded_spectrum = np.abs(np.fft.fft(decoded, fft_length))[: fft_length // 2]
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.plot(frequency, 20 * np.log10(original_spectrum + np.finfo(float).eps), "b", label="原始语音")
    axis.plot(frequency, 20 * np.log10(decoded_spectrum + np.finfo(float).eps), "r--", label="DPCM解码语音")
    axis.set(title="原始语音与DPCM解码语音频谱对比", xlabel="频率 / Hz", ylabel="幅度 / dB")
    axis.grid(True, alpha=0.35)
    axis.legend()
    fig.savefig(RESULT_DIR / "02_DPCM频谱对比.png", dpi=180)
    plt.close(fig)

    print(f"恢复语音：{RESULT_DIR / 'dpcm_recovered.wav'}")
    print("第一步信源编解码完成。\n")
    return {"fs": fs, "samples": x.size, "mse": mse, "snr": snr, "consistent": decoder_consistent}


if __name__ == "__main__":
    run()

