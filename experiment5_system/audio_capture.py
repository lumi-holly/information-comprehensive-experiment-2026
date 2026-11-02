"""独立采集线程与有界队列，避免网络等待直接阻塞麦克风读取。

默认缓存 16 块是实现建议，不是课件阈值。缓存耗尽、音频输入溢出都会
明确失败，不能静默丢失采样后继续报告传输完整。
"""
import queue
import threading
import time

import numpy as np


class QueuedMicrophone:
    def __init__(self, sample_rate, block_samples, total_samples, device=None,
                 queue_blocks=16, stream_factory=None):
        for name, value in (("sample_rate", sample_rate),
                            ("block_samples", block_samples),
                            ("total_samples", total_samples),
                            ("queue_blocks", queue_blocks)):
            if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                    or value <= 0):
                raise ValueError("%s 必须为正整数。" % name)
        if stream_factory is None:
            try:
                import sounddevice as sd
            except ImportError as exc:
                raise RuntimeError("麦克风模式需要安装 sounddevice。") from exc
            stream_factory = sd.InputStream
        self.sample_rate = int(sample_rate)
        self.block_samples = int(block_samples)
        self.total_samples = int(total_samples)
        self._factory = stream_factory
        self._device = device
        self._queue = queue.Queue(maxsize=int(queue_blocks))
        self._stop = threading.Event()
        self._done = threading.Event()
        self._error = None
        self._stream = None
        self._thread = None
        self._started = False
        self._closed = False
        self._consumed = 0

    def start(self):
        if self._closed or self._stop.is_set():
            raise RuntimeError("麦克风已停止或关闭，不能重新启动。")
        if self._started:
            raise RuntimeError("麦克风已经启动。")
        self._stream = self._factory(samplerate=self.sample_rate, channels=1,
                                     dtype="int16", blocksize=self.block_samples,
                                     device=self._device)
        try:
            self._stream.start()
        except Exception:
            self._stream.close()
            self._closed = True
            raise
        self._started = True
        self._thread = threading.Thread(target=self._produce,
                                        name="experiment5-microphone", daemon=True)
        self._thread.start()
        return self

    def _produce(self):
        produced = 0
        try:
            while produced < self.total_samples and not self._stop.is_set():
                n = min(self.block_samples, self.total_samples - produced)
                data, overflowed = self._stream.read(n)
                if self._stop.is_set():
                    break
                if overflowed:
                    raise RuntimeError("麦克风输入发生溢出，采样不完整，已停止本次传输。")
                data = np.asarray(data)
                if data.shape != (n, 1) or data.dtype != np.dtype("int16"):
                    raise RuntimeError("音频设备返回的样本数、声道或类型不符合 int16 单声道。")
                try:
                    self._queue.put_nowait(data.copy())
                except queue.Full as exc:
                    raise RuntimeError("麦克风缓存队列已满，网络或处理速度不足，"
                                       "已停止采集，未静默丢弃音频块。") from exc
                produced += n
        except Exception as exc:
            if not self._stop.is_set():
                self._error = RuntimeError("麦克风采集失败：%s" % exc)
        finally:
            self._done.set()

    def _check_error(self):
        if self._error is not None:
            raise self._error

    def read(self, n):
        """依次读取整块，最后一块长度为剩余样本数，不额外补齐采样。"""
        if not self._started or self._closed or self._stop.is_set():
            raise RuntimeError("麦克风尚未启动，或已停止/关闭。")
        self._check_error()
        expected = min(self.block_samples, self.total_samples - self._consumed)
        if (isinstance(n, bool) or not isinstance(n, (int, np.integer))
                or n <= 0 or n != expected):
            raise ValueError("本次 read(n) 需要 n=%d，且不能超出录音总样本数。" % expected)
        deadline = time.monotonic() + 10.0
        while True:
            self._check_error()
            if self._closed or self._stop.is_set():
                raise RuntimeError("麦克风已停止或关闭。")
            try:
                data = self._queue.get(timeout=0.1)
            except queue.Empty:
                self._check_error()
                if self._done.is_set():
                    raise RuntimeError("麦克风采集提前结束，音频块不完整。")
                if time.monotonic() >= deadline:
                    raise RuntimeError("等待麦克风音频块超过 10 秒。")
                continue
            self._check_error()
            if data.shape != (n, 1):
                raise RuntimeError("麦克风队列样本数与读取请求不一致。")
            self._consumed += n
            return data, False

    def stop(self):
        """主动中止设备读操作，最多等后台线程退出两秒。"""
        already_stopped = self._stop.is_set()
        self._stop.set()
        if self._stream is not None and not already_stopped:
            try:
                self._stream.abort()
            except Exception:
                # 某些设备无法 abort 时关闭流，仍需解除阻塞读。
                try:
                    self._stream.close()
                except Exception:
                    pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.stop()
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass

