from __future__ import annotations

import argparse
import socket
import time
import wave
from pathlib import Path

from network_io import recv_packet
from signal_chain import decode_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="实验四语音接收端")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=Path("received_audio.wav"))
    parser.add_argument("--play", action="store_true", help="接收时播放语音，需要 sounddevice")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    player = None
    frame_count = 0
    corrected_total = 0
    started = time.perf_counter()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.host, args.port))
            server.listen(1)
            print(f"等待发送端连接：{args.host}:{args.port}")
            connection, address = server.accept()
            with connection:
                print(f"发送端已连接：{address[0]}:{address[1]}")
                while True:
                    packet = recv_packet(connection)
                    if packet is None:
                        break
                    frame = decode_frame(packet)
                    if frame.sequence != frame_count:
                        raise RuntimeError(f"帧序号不连续：期望 {frame_count}，收到 {frame.sequence}")
                    if writer is None:
                        writer = wave.open(str(args.output), "wb")
                        writer.setnchannels(1)
                        writer.setsampwidth(2)
                        writer.setframerate(frame.sample_rate)
                        if args.play:
                            try:
                                import sounddevice as sd
                            except ImportError as exc:
                                raise RuntimeError("播放模式需要安装 sounddevice。") from exc
                            player = sd.RawOutputStream(
                                samplerate=frame.sample_rate, channels=1, dtype="int16"
                            )
                            player.start()
                    raw = frame.samples.astype("<i2", copy=False).tobytes()
                    writer.writeframesraw(raw)
                    if player is not None:
                        player.write(raw)
                    frame_count += 1
                    corrected_total += frame.corrected_codewords
                    print(
                        f"接收帧 {frame.sequence:04d}，样本 {frame.samples.size}，"
                        f"执行纠错 {frame.corrected_codewords} 个码字"
                    )
    finally:
        if player is not None:
            player.stop()
            player.close()
        if writer is not None:
            writer.close()

    elapsed = time.perf_counter() - started
    print(f"接收完成：{frame_count} 帧，累计纠错 {corrected_total} 个码字，耗时 {elapsed:.3f} s")
    print(f"恢复语音：{args.output.resolve()}")


if __name__ == "__main__":
    main()

