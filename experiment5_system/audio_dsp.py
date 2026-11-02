"""实验五音频读写、加噪、自编 FIR 与质量评价（仅依赖 NumPy）。

默认 FIR 参数是本工程建议值，不是课件的性能阈值。
这里使用因果线性相位 FIR；旧 MATLAB 的 filtfilt 是离线零相位算法，
二者不应声称逐样本相同。质量评价可补偿 FIR 固定群延迟。
"""
from pathlib import Path
import csv
import math
import wave

import numpy as np


def _samples(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("音频必须是一维、有限数值的单声道数组。")
    return values


def _sample_rate(fs):
    if isinstance(fs, bool) or not isinstance(fs, (int, np.integer)) or fs <= 0:
        raise ValueError("采样率必须为正整数。")
    return int(fs)


def read_wav(path):
    """读取 16 bit PCM WAV；立体声按声道平均，按 32768 满量程缩放。"""
    with wave.open(str(path), "rb") as reader:
        if reader.getsampwidth() != 2 or reader.getcomptype() != "NONE":
            raise ValueError("输入必须是未压缩的 16 bit PCM WAV。")
        channels = reader.getnchannels()
        fs = _sample_rate(reader.getframerate())
        frame_count = reader.getnframes()
        raw = reader.readframes(frame_count)
    if channels < 1 or len(raw) != frame_count * channels * 2:
        raise ValueError("WAV 内容截断或声道数非法。")
    pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)
    return pcm.astype(np.float64).mean(axis=1) / 32768.0, fs


def write_wav(path, samples, fs):
    """截幅后以单声道 16 bit PCM 写入；不进行逐文件峰值归一化。"""
    samples = _samples(samples)
    fs = _sample_rate(fs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # 与 read_wav 的满量程约定一致，+1 只能表示为 32767。
    pcm = np.clip(np.rint(np.clip(samples, -1.0, 1.0) * 32768.0),
                  -32768, 32767).astype("<i2")
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(fs)
        writer.writeframes(pcm.tobytes())
    return str(target)


def add_tone(samples, fs, frequency=5000, amplitude=0.2, start_sample=0):
    """叠加正弦干扰；start_sample 保证跨帧相位连续，不归一化、不截幅。"""
    samples = _samples(samples)
    fs = _sample_rate(fs)
    if not math.isfinite(frequency) or not 0 < frequency < fs / 2:
        raise ValueError("干扰频率必须满足 0 < frequency < fs/2（采样定理）。")
    if not math.isfinite(amplitude) or amplitude < 0:
        raise ValueError("干扰幅值必须是非负有限数。")
    if (isinstance(start_sample, bool)
            or not isinstance(start_sample, (int, np.integer)) or start_sample < 0):
        raise ValueError("起始样本编号必须是非负整数。")
    indices = np.arange(samples.size, dtype=np.float64) + start_sample
    return samples + amplitude * np.sin(2 * np.pi * frequency * indices / fs)


def design_bandpass(fs, low=300, high=3400, taps=129):
    """使用窗函数法自编奇数长度带通 FIR，中心通带增益归一为 1。"""
    fs = _sample_rate(fs)
    if not (math.isfinite(low) and math.isfinite(high)
            and 0 < low < high < fs / 2):
        raise ValueError("滤波频率必须满足 0 < low < high < fs/2。")
    if (isinstance(taps, bool) or not isinstance(taps, (int, np.integer))
            or taps < 3 or taps % 2 == 0):
        raise ValueError("FIR taps 必须是大于等于 3 的奇数。")
    m = np.arange(taps, dtype=np.float64) - (taps - 1) / 2
    high_normalized = high / fs
    low_normalized = low / fs
    # 两个理想低通脉冲响应相减，再加 Hamming 窗。
    coefficients = (2 * high_normalized * np.sinc(2 * high_normalized * m)
                    - 2 * low_normalized * np.sinc(2 * low_normalized * m))
    coefficients *= np.hamming(taps)
    center = (low + high) / 2
    response = abs(np.sum(coefficients * np.exp(-2j * np.pi * center * m / fs)))
    if response < 1e-12:
        raise ValueError("滤波器通带增益过小，请增加 taps 或调整频率。")
    return coefficients / response


class FIRFilter:
    """保存前一帧历史样本，避免每个网络帧重新滤波产生接缝。"""

    def __init__(self, taps):
        self.taps = _samples(taps).copy()
        if self.taps.size < 1:
            raise ValueError("FIR 系数不能为空。")
        self.delay_samples = (self.taps.size - 1) // 2
        self.reset()

    def reset(self):
        self._history = np.zeros(self.taps.size - 1, dtype=np.float64)

    def process(self, samples):
        samples = _samples(samples)
        if samples.size == 0:
            return samples.copy()
        combined = np.concatenate((self._history, samples))
        output = np.convolve(combined, self.taps, mode="valid")
        if self._history.size:
            self._history = combined[-self._history.size:].copy()
        return output


def analyze(samples, fs):
    """时域 RMS/峰值/过零率，FFT 主峰；FFT 最多使用前 32768 个点。"""
    samples = _samples(samples)
    fs = _sample_rate(fs)
    count = samples.size
    fft_count = min(count, 32768)
    result = {
        "sample_count": int(count), "sample_rate": fs,
        "duration_s": count / fs,
        "rms": float(np.sqrt(np.mean(samples ** 2))) if count else 0.0,
        "peak": float(np.max(np.abs(samples))) if count else 0.0,
        "zero_crossing_rate": (float(np.mean(np.signbit(samples[1:])
                                          != np.signbit(samples[:-1])))
                               if count > 1 else 0.0),
        "fft_samples": int(fft_count), "fft_uses_prefix_subset": count > fft_count,
        "fft_peak_hz": None,
    }
    if fft_count:
        segment = samples[:fft_count]
        window = np.hanning(fft_count) if fft_count > 2 else np.ones(fft_count)
        magnitudes = np.abs(np.fft.rfft(segment * window))
        if np.any(magnitudes > 0):
            result["fft_peak_hz"] = float(np.fft.rfftfreq(fft_count, 1 / fs)
                                         [np.argmax(magnitudes)])
    return result


def save_analysis_csv(path, samples, fs):
    """导出抽取时域点和前缀 FFT，CSV 明确标记抽取与分析样本数。"""
    samples = _samples(samples)
    fs = _sample_rate(fs)
    prefix = Path(path).with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    time_path = prefix.with_name(prefix.name + "_time.csv")
    spectrum_path = prefix.with_name(prefix.name + "_spectrum.csv")
    count = samples.size
    step = max(1, int(math.ceil(count / 4000)))
    with time_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "time_s", "amplitude",
                         "display_decimation_step", "total_samples"])
        for index in range(0, count, step):
            writer.writerow([index, index / fs, samples[index], step, count])
    fft_count = min(count, 32768)
    with spectrum_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_hz", "single_sided_amplitude", "fft_samples",
                         "prefix_subset", "total_samples", "window"])
        if fft_count:
            window = np.hanning(fft_count) if fft_count > 2 else np.ones(fft_count)
            amplitude = np.abs(np.fft.rfft(samples[:fft_count] * window)) / window.sum()
            if fft_count % 2 == 0:
                amplitude[1:-1] *= 2
            else:
                amplitude[1:] *= 2
            for frequency, value in zip(np.fft.rfftfreq(fft_count, 1 / fs), amplitude):
                writer.writerow([frequency, value, fft_count, count > fft_count,
                                 count, "Hann" if fft_count > 2 else "rectangular"])
    return {"time_csv": str(time_path), "spectrum_csv": str(spectrum_path),
            "display_decimation_step": step, "fft_samples": int(fft_count),
            "fft_uses_prefix_subset": count > fft_count}


def metrics(reference, test, delay_samples=0):
    """用同长度信号评价；补偿 test 的因果延迟后计算，不输出 NaN/Inf。"""
    reference, test = _samples(reference), _samples(test)
    if reference.size != test.size:
        raise ValueError("质量评价要求参考信号和待测信号长度相同。")
    if (isinstance(delay_samples, bool)
            or not isinstance(delay_samples, (int, np.integer)) or delay_samples < 0):
        raise ValueError("延迟补偿样本数必须是非负整数。")
    if reference.size <= delay_samples:
        raise ValueError("信号为空或延迟补偿后没有可比较的样本。")
    if delay_samples:
        reference = reference[:-delay_samples]
        test = test[delay_samples:]
    mse = float(np.mean((reference - test) ** 2))
    power = float(np.mean(reference ** 2))
    exact = bool(np.array_equal(reference, test))
    snr = 10 * math.log10(power / mse) if mse > 0 and power > 0 else None
    if snr is not None and not math.isfinite(snr):
        snr = None
    return {"mse": mse, "snr_db": snr, "compared_samples": int(reference.size),
            "delay_compensation_samples": int(delay_samples), "exact_match": exact,
            "snr_status": ("exact_match" if exact else
                           "zero_reference_power" if power == 0 else "finite")}
