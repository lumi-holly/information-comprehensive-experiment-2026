"""使用虚拟设备验证队列采集，测试不需要实际麦克风。"""
import threading
import time
import unittest

import numpy as np

from audio_capture import QueuedMicrophone


class FakeStream:
    def __init__(self, overflow=False, blocking=False, **kwargs):
        self.position = 0
        self.overflow = overflow
        self.blocking = blocking
        self.requests = []
        self.started = False
        self.closed = False
        self.aborted = threading.Event()

    def start(self):
        self.started = True

    def read(self, n):
        self.requests.append(n)
        if self.blocking:
            self.aborted.wait(5)
            raise RuntimeError("device aborted")
        data = np.arange(self.position, self.position + n, dtype=np.int16)[:, None]
        self.position += n
        return data, self.overflow

    def abort(self):
        self.aborted.set()

    def close(self):
        self.closed = True
        self.aborted.set()


class AudioCaptureTests(unittest.TestCase):
    def make_capture(self, total=10, block=4, queue_blocks=16, **kwargs):
        stream = FakeStream(**kwargs)
        capture = QueuedMicrophone(16000, block, total,
                                  queue_blocks=queue_blocks,
                                  stream_factory=lambda **_: stream)
        return capture, stream

    def test_exact_total_and_last_partial_block(self):
        capture, stream = self.make_capture()
        try:
            capture.start()
            blocks = [capture.read(n)[0] for n in (4, 4, 2)]
            np.testing.assert_array_equal(np.concatenate(blocks)[:, 0], np.arange(10))
            self.assertEqual(stream.requests, [4, 4, 2])
            with self.assertRaises(ValueError):
                capture.read(1)
        finally:
            capture.close()
        self.assertTrue(stream.closed)

    def test_input_overflow_fails_in_consumer(self):
        capture, _ = self.make_capture(overflow=True)
        try:
            capture.start()
            self.assertTrue(capture._done.wait(1))
            with self.assertRaisesRegex(RuntimeError, "溢出"):
                capture.read(4)
        finally:
            capture.close()

    def test_full_queue_fails_without_dropping(self):
        capture, _ = self.make_capture(total=40, queue_blocks=1)
        try:
            capture.start()
            self.assertTrue(capture._done.wait(1))
            with self.assertRaisesRegex(RuntimeError, "队列已满"):
                capture.read(4)
        finally:
            capture.close()

    def test_close_unblocks_read_and_is_idempotent(self):
        capture, stream = self.make_capture(blocking=True)
        capture.start()
        deadline = time.monotonic() + 1
        while not stream.requests and time.monotonic() < deadline:
            time.sleep(.001)
        started = time.monotonic()
        capture.close()
        capture.close()
        capture.stop()
        self.assertLess(time.monotonic() - started, 2.5)
        self.assertFalse(capture._thread.is_alive())
        self.assertTrue(stream.closed)
        with self.assertRaises(RuntimeError):
            capture.read(4)

    def test_invalid_read_size_and_constructor(self):
        capture, _ = self.make_capture()
        try:
            capture.start()
            with self.assertRaises(ValueError):
                capture.read(3)
        finally:
            capture.close()
        with self.assertRaises(ValueError):
            QueuedMicrophone(16000, 0, 10, stream_factory=FakeStream)


if __name__ == "__main__":
    unittest.main()
