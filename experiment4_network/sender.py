from __future__ import annotations

import argparse
import socket
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from network_io import send_packet
from signal_chain import encode_frame


def wav_blocks(path: Path, block_samples: int) -> tuple[int, Iterator[np.ndarray]]:
    reader = wave.open(str(path), "rb")
    if reader.getsampwidth() != 2:
        reader.close()
        raise ValueError("输入 WAV 必须是 16 bit PCM。")
    sample_rate = reader.getframerate()
    channels = reader.getnchannels()

    def iterator() -> Iterator[np.ndarray]:
        try:
            while True:
                raw = reader.readframes(block_samples)
                if not raw:
                    break
                samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)
                yield samples.mean(axis=1).astype(np.int16)
        finally:
            reader.close()

    return sample_rate, iterator()


def microphone_blocks(sample_rate: int, block_samples: int, seconds: float) -> Iterator[np.ndarray]:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("麦克风模式需要安装 sounddevice：pip install sounddevice") from exc
    block_count = max(1, int(np.ceil(seconds * sample_rate / block_samples)))
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", blocksize=block_samples) as stream:
        for _ in range(block_count):
            data, overflowed = stream.read(block_samples)
            if overflowed:
                print("警告：麦克风输入发生溢出。")
            yield data[:, 0].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="实验四语音发送端")
    parser.add_argument("--host", required=True, help="接收端 IP 地址")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--wav", type=Path, help="发送 16 bit PCM WAV 文件")
    parser.add_argument("--mic", action="store_true", help="实时采集麦克风")
    parser.add_argument("--seconds", type=float, default=10.0, help="麦克风采集秒数")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--block-samples", type=int, default=2048)
    parser.add_argument("--ber", type=float, default=0.0, help="仿真信道误码率")
    parser.add_argument("--realtime", action="store_true", help="发送 WAV 时按真实时长限速")
    args = parser.parse_args()
    if args.mic == (args.wav is not None):
        parser.error("必须且只能选择 --wav 或 --mic。")

    if args.wav is not None:
        sample_rate, blocks = wav_blocks(args.wav, args.block_samples)
    else:
        sample_rate = args.sample_rate
        blocks = microphone_blocks(sample_rate, args.block_samples, args.seconds)

    total_samples = 0
    started = time.perf_counter()
    with socket.create_connection((args.host, args.port), timeout=10) as connection:
        connection.settimeout(None)
        for sequence, samples in enumerate(blocks):
            packet = encode_frame(sequence, sample_rate, samples, args.ber)
            send_packet(connection, packet)
            total_samples += samples.size
            print(f"发送帧 {sequence:04d}，样本 {samples.size}，网络字节 {len(packet)}")
            if args.realtime and args.wav is not None:
                expected = total_samples / sample_rate
                delay = expected - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(delay)

    elapsed = time.perf_counter() - started
    print(f"发送完成：{total_samples} 个样本，音频 {total_samples / sample_rate:.2f} s，耗时 {elapsed:.3f} s")


if __name__ == "__main__":
    main()

