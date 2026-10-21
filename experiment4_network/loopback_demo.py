from __future__ import annotations

import argparse
import socket
import threading
import wave
from pathlib import Path

import numpy as np

from network_io import recv_packet, send_packet
from signal_chain import decode_frame, encode_frame


def make_test_audio(sample_rate: int, seconds: float) -> np.ndarray:
    time_axis = np.arange(int(sample_rate * seconds)) / sample_rate
    signal = 0.55 * np.sin(2 * np.pi * 440 * time_axis) + 0.2 * np.sin(2 * np.pi * 880 * time_axis)
    return np.round(signal * 32767).astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser(description="不需要两台设备的实验四回环测试")
    parser.add_argument("--output", type=Path, default=Path("loopback_received.wav"))
    parser.add_argument("--ber", type=float, default=0.0)
    args = parser.parse_args()
    sample_rate = 16000
    block_samples = 2048
    source = make_test_audio(sample_rate, 1.0)
    received_blocks: list[np.ndarray] = []
    errors: list[BaseException] = []

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def receive() -> None:
            try:
                connection, _ = server.accept()
                with connection:
                    while True:
                        packet = recv_packet(connection)
                        if packet is None:
                            break
                        received_blocks.append(decode_frame(packet).samples)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=receive)
        thread.start()
        with socket.create_connection(("127.0.0.1", port)) as sender:
            for sequence, start in enumerate(range(0, source.size, block_samples)):
                block = source[start : start + block_samples]
                send_packet(sender, encode_frame(sequence, sample_rate, block, args.ber))
        thread.join()

    if errors:
        raise errors[0]
    received = np.concatenate(received_blocks)
    if received.size != source.size:
        raise RuntimeError("回环测试的样本数不一致。")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(received.astype("<i2", copy=False).tobytes())
    mse = float(np.mean((source.astype(np.float64) - received.astype(np.float64)) ** 2))
    print(f"回环测试通过：{source.size} 个样本，MSE={mse:.2f}")
    print(f"输出文件：{args.output.resolve()}")


if __name__ == "__main__":
    main()

