"""会话重放和未完成结果误用的回归测试；仅使用本机 TCP 做开发验证。"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from audio_dsp import write_wav
from network_io import receive_control, recv_packet, send_control


class SessionSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.key_material = os.urandom(48)
        self.key = self.directory / 'test.key'
        self.key.write_bytes(self.key_material)
        self.wav = self.directory / 'input.wav'
        self.samples = .4 * np.sin(2 * np.pi * 700 * np.arange(2048) / 16000)
        write_wav(self.wav, self.samples, 16000)

    def tearDown(self):
        self.temp.cleanup()

    def cli(self, arguments):
        return subprocess.run(
            [sys.executable, str(BASE / 'app.py')] + list(map(str, arguments)),
            capture_output=True, encoding='utf-8',
            env=dict(os.environ, PYTHONIOENCODING='utf-8'), timeout=15)

    def test_replayed_ready_is_rejected_before_audio(self):
        """旧合法签名不代表当前接收端确认了本次客户端的随机数。"""
        old_session = '1' * 32
        old_client_nonce = '2' * 32
        mac_key = self.key_material[16:]
        observed = {'audio_packets': [], 'errors': []}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(('127.0.0.1', 0))
            server.listen(1)
            server.settimeout(8)
            port = server.getsockname()[1]

            def replay_old_server():
                try:
                    connection, _ = server.accept()
                    with connection:
                        connection.settimeout(8)
                        # 模拟此前已抓取、签名有效的两条旧控制消息。
                        send_control(connection, {'type': 'challenge',
                            'session_id': old_session}, mac_key)
                        start = receive_control(connection, mac_key, 'start')
                        observed['fresh_client_nonce'] = start.get('client_nonce')
                        send_control(connection, {'type': 'ready',
                            'session_id': old_session,
                            'client_nonce': old_client_nonce}, mac_key)
                        try:
                            observed['audio_packets'].append(recv_packet(connection))
                        except (ConnectionError, ConnectionResetError):
                            observed['closed_before_audio'] = True
                except Exception as exc:
                    observed['errors'].append(repr(exc))

            replay = threading.Thread(target=replay_old_server, daemon=True)
            replay.start()
            txdir = self.directory / 'sender'
            run = self.cli(['send', '--host', '127.0.0.1', '--port', port,
                            '--wav', self.wav, '--key', self.key, '--out', txdir,
                            '--timeout', 5])
            replay.join(timeout=10)
            self.assertFalse(replay.is_alive(), '伪服务端线程未正常退出。')
            self.assertEqual(observed['errors'], [])
            self.assertNotEqual(run.returncode, 0, run.stdout)
            self.assertIn('拒绝旧会话响应', run.stderr)
            self.assertIsInstance(observed.get('fresh_client_nonce'), str)
            self.assertNotEqual(observed['fresh_client_nonce'], old_client_nonce)
            self.assertEqual(observed['audio_packets'], [])
            self.assertTrue(observed.get('closed_before_audio'))
            summary = json.loads((txdir / 'send_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['status'], 'incomplete')
            self.assertEqual(summary['frames'], 0)
            self.assertEqual(summary['samples'], 0)

    def test_evaluate_rejects_either_incomplete_summary(self):
        """两份 WAV 即使同长度，也不能把中断的运行评价成正式完整结果。"""
        txdir, rxdir = self.directory / 'sender', self.directory / 'receiver'
        txdir.mkdir()
        rxdir.mkdir()
        reference = txdir / 'reference.wav'
        recovered = rxdir / 'recovered.wav'
        write_wav(reference, self.samples, 16000)
        write_wav(recovered, self.samples, 16000)
        for incomplete_role in ('sender', 'receiver'):
            with self.subTest(incomplete_role=incomplete_role):
                for role, path in [('sender', txdir / 'send_summary.json'),
                                   ('receiver', rxdir / 'receive_summary.json')]:
                    path.write_text(json.dumps({'status': 'incomplete'
                        if role == incomplete_role else 'complete'}), encoding='utf-8')
                output = self.directory / ('evaluation_' + incomplete_role + '.json')
                run = self.cli(['evaluate', '--reference', reference, '--received', recovered,
                                '--delay-samples', 0, '--taps', 0, '--out', output])
                self.assertNotEqual(run.returncode, 0, run.stdout)
                self.assertIn('禁止评价未完成的运行', run.stderr)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
