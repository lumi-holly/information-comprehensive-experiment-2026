"""软件集成测试使用本机 TCP，仅为开发验证，不是课程真实网络证据。"""
import hashlib
import json
import os
from pathlib import Path
import queue
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np

from app import build_parser, checked_config
from audio_dsp import read_wav, write_wav, FIRFilter, design_bandpass
from network_io import (send_packet, recv_packet, receive_control, send_control,
                        LENGTH, MAX_PACKET_BYTES)

BASE = Path(__file__).resolve().parents[1]


class SystemTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.key = self.directory / 'test.key'
        self.key.write_bytes(os.urandom(48))
        self.wav = self.directory / 'input.wav'
        self.samples = .5 * np.sin(2 * np.pi * 900 * np.arange(4777) / 16000)
        write_wav(self.wav, self.samples, 16000)
        self.children = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)
        self.temp.cleanup()

    def cli(self, arguments, timeout=20):
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        return subprocess.run([sys.executable, str(BASE / 'app.py')] + list(map(str, arguments)),
                              capture_output=True, encoding='utf-8', env=env, timeout=timeout)

    def receiver(self, suffix='rx'):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        child = subprocess.Popen([sys.executable, str(BASE / 'app.py'), 'receive',
            '--bind', '127.0.0.1', '--port', str(port), '--key', str(self.key),
            '--timeout', '8', '--out', str(self.directory / suffix)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', env=env)
        self.children.append(child)
        lines = queue.Queue()
        thread = threading.Thread(target=lambda: lines.put(child.stdout.readline()), daemon=True)
        thread.start()
        self.assertIn('等待发送端', lines.get(timeout=8))
        return child, port, self.directory / suffix

    def test_complete_modes_and_evaluation(self):
        for source, channel, test in [('dpcm8', 'hamming74', 'single'),
                                      ('pcm16', 'none', 'none')]:
            with self.subTest(source=source):
                rx, port, rxdir = self.receiver('rx_' + source)
                txdir = self.directory / ('tx_' + source)
                run = self.cli(['send', '--host', '127.0.0.1', '--port', port,
                    '--key', self.key, '--wav', self.wav, '--out', txdir,
                    '--source', source, '--channel', channel, '--channel-test', test,
                    '--tone-amplitude', '0'])
                self.assertEqual(run.returncode, 0, run.stderr)
                stdout, stderr = rx.communicate(timeout=20)
                self.assertEqual(rx.returncode, 0, stderr)
                tx = json.loads((txdir / 'send_summary.json').read_text(encoding='utf-8'))
                result = json.loads((rxdir / 'receive_summary.json').read_text(encoding='utf-8'))
                self.assertEqual(tx['status'], 'complete')
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(result['samples'], 4777)
                self.assertEqual(result['frames'], 5)
                self.assertEqual(tx['audio_framed_bytes'], result['audio_framed_bytes'])
                reference, fs = read_wav(txdir / 'reference.wav')
                decoded, _ = read_wav(rxdir / 'decoded_noisy.wav')
                recovered, _ = read_wav(rxdir / 'recovered.wav')
                if source == 'pcm16':
                    np.testing.assert_array_equal(reference, decoded)
                expected = FIRFilter(design_bandpass(fs)).process(decoded)
                self.assertLess(float(np.max(np.abs(expected - recovered))), 2 / 32768)
                evaluation = self.cli(['evaluate', '--reference', txdir / 'reference.wav',
                    '--received', rxdir / 'recovered.wav', '--decoded', rxdir / 'decoded_noisy.wav',
                    '--out', self.directory / ('eval_' + source + '.json')])
                self.assertEqual(evaluation.returncode, 0, evaluation.stderr)
                data = json.loads(evaluation.stdout)
                self.assertEqual(data['against_clean_delay_aligned']['compared_samples'], 4777 - 64)

    def test_wrong_key_never_completes(self):
        rx, port, rxdir = self.receiver()
        wrong = self.directory / 'wrong.key'
        wrong.write_bytes(os.urandom(48))
        run = self.cli(['send', '--host', '127.0.0.1', '--port', port,
            '--key', wrong, '--wav', self.wav, '--out', self.directory / 'tx'])
        self.assertNotEqual(run.returncode, 0)
        rx.communicate(timeout=20)
        self.assertNotEqual(rx.returncode, 0)
        self.assertEqual(json.loads((rxdir / 'receive_summary.json').read_text(encoding='utf-8'))['status'], 'incomplete')

    def test_disconnect_without_end_is_incomplete(self):
        rx, port, rxdir = self.receiver()
        material = self.key.read_bytes()
        with socket.create_connection(('127.0.0.1', port), timeout=5) as conn:
            challenge = receive_control(conn, material[16:], 'challenge')
            args = build_parser().parse_args(['send', '--host', 'unused', '--wav', str(self.wav)])
            config = checked_config({'sample_rate': 16000, 'block_samples': args.block_samples,
                'low_hz': args.low_hz, 'high_hz': args.high_hz, 'taps': args.taps,
                'tone_hz': args.tone_hz, 'tone_amplitude': args.tone_amplitude,
                'headroom': args.headroom, 'source_codec': args.source,
                'channel_codec': args.channel, 'channel_test': args.channel_test, 'ber': args.ber})
            send_control(conn, {'type': 'start', 'session_id': challenge['session_id'],
                         'client_nonce': 'a' * 32, 'config': config, 'expected_samples': 4777}, material[16:])
            receive_control(conn, material[16:], 'ready')
        rx.communicate(timeout=20)
        self.assertNotEqual(rx.returncode, 0)
        summary = json.loads((rxdir / 'receive_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['status'], 'incomplete')
        self.assertEqual(summary['samples'], 0)

    def test_fragmented_tcp_and_invalid_length(self):
        left, right = socket.socketpair()
        payload = b'hello' * 100
        def fragmented():
            with left:
                packet = LENGTH.pack(len(payload)) + payload
                for index in range(0, len(packet), 3):
                    left.sendall(packet[index:index + 3])
        thread = threading.Thread(target=fragmented)
        thread.start()
        with right:
            self.assertEqual(recv_packet(right), payload)
            with self.assertRaises(ConnectionError):
                recv_packet(right)
        thread.join(timeout=5)
        left, right = socket.socketpair()
        with left, right:
            left.sendall(struct.pack('!I', MAX_PACKET_BYTES + 1))
            with self.assertRaises(ValueError):
                recv_packet(right)


if __name__ == '__main__':
    unittest.main()
