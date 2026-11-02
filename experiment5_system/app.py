"""实验五：语音采集/分析/加噪/编码/安全网络传输/译码/滤波/评价。

正式运行：两台设备，接收端先启动。命令见 README.md。
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import secrets
import socket
import sys
import time
import wave
from pathlib import Path

import numpy as np
import cryptography

from audio_dsp import (read_wav, write_wav, add_tone, design_bandpass,
                       FIRFilter, analyze, save_analysis_csv, metrics)
from codec import encode_frame, decode_frame
from audio_capture import QueuedMicrophone
from network_io import (send_packet, recv_packet, send_control, receive_control,
                        parse_control, configure_connection)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def environment():
    return {'os': platform.platform(), 'machine': platform.machine(),
            'processor': platform.processor(), 'cpu_count': os.cpu_count(),
            'python': platform.python_version(), 'numpy': np.__version__,
            'cryptography': cryptography.__version__,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z')}


def load_key(path):
    material = Path(path).read_bytes()
    if len(material) != 48:
        raise ValueError('密钥文件必须是 keygen 生成的 48 字节文件。')
    return material[:16], material[16:]


def prepare_run_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    if any(path.glob('*.wav')) or any(path.glob('*_summary.json')):
        raise ValueError('结果目录已含运行记录，请使用新的 --out 目录，避免覆盖已有实验数据。')


def keygen(args):
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # x 模式避免误覆盖，两端使用同一份文件，密钥内容不打印。
    with args.out.open('xb') as handle:
        handle.write(secrets.token_bytes(48))
    if os.name != 'nt':
        args.out.chmod(0o600)
    print('已生成密钥文件：' + str(args.out.resolve()))
    print('将同一份密钥通过可信方式复制到另一端，再启动接收程序。')


class WaveSink:
    def __init__(self, path, fs):
        self.handle = wave.open(str(path), 'wb')
        self.handle.setnchannels(1)
        self.handle.setsampwidth(2)
        self.handle.setframerate(fs)

    def write(self, samples):
        raw = np.clip(np.rint(np.asarray(samples) * 32768), -32768, 32767).astype('<i2')
        self.handle.writeframesraw(raw.tobytes())

    def close(self):
        self.handle.close()


def close_resources(resources):
    """一个设备关闭失败也继续关闭其他文件，并把失败写入摘要。"""
    errors = []
    for resource in resources:
        if resource is not None:
            try:
                resource.close()
            except Exception as exc:
                errors.append(str(exc))
    return errors


def audio_module():
    try:
        import sounddevice
    except (ImportError, OSError) as exc:
        raise RuntimeError('麦克风/播放需要 sounddevice 和 PortAudio，见 README 的音频安装部分。') from exc
    return sounddevice


def checked_config(config):
    """握手参数逐项检查；这些保护范围不是课程验收阈值。"""
    integer_ranges = {'sample_rate': (8000, 48000), 'block_samples': (1, 48000),
                      'taps': (3, 1025)}
    for field, (low, high) in integer_ranges.items():
        value = config.get(field)
        if type(value) is not int or not low <= value <= high:
            raise ValueError('非法配置：' + field)
    if config['taps'] % 2 != 1:
        raise ValueError('FIR taps 必须为奇数。')
    for field in ['low_hz', 'high_hz', 'tone_hz', 'tone_amplitude', 'headroom', 'ber']:
        value = config.get(field)
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError('非法配置：' + field)
    fs = config['sample_rate']
    if not 0 < config['low_hz'] < config['high_hz'] < fs / 2:
        raise ValueError('滤波截止频率必须满足 0 < low < high < fs/2。')
    if not 0 <= config['tone_amplitude'] <= 0.5:
        raise ValueError('加噪幅值范围为 0～0.5（实现范围）。')
    if config['tone_amplitude'] > 0 and not 0 < config['tone_hz'] < fs / 2:
        raise ValueError('采样率必须大于干扰频率的两倍。')
    if not 0 < config['headroom'] <= 1 or config['headroom'] + config['tone_amplitude'] > 1:
        raise ValueError('headroom 必须大于 0，且 headroom + 加噪幅值不能超过 1，避免削波。')
    if config.get('source_codec') not in ('dpcm8', 'pcm16'):
        raise ValueError('信源编码只支持 dpcm8/pcm16。')
    if config.get('channel_codec') not in ('hamming74', 'none'):
        raise ValueError('信道编码只支持 hamming74/none。')
    if config.get('channel_test') not in ('none', 'single', 'random'):
        raise ValueError('未知模拟误码模式。')
    if not 0 <= config['ber'] <= 0.02:
        raise ValueError('BER 仅支持 0～0.02 的实验范围，非课件阈值。')
    if config['channel_test'] != 'random' and config['ber'] != 0:
        raise ValueError('--ber 只用于 --channel-test random。')
    design_bandpass(fs, config['low_hz'], config['high_hz'], config['taps'])
    return config


def ensure_session(message, session_id):
    if message.get('session_id') != session_id:
        raise ValueError('收到不属于当前会话的消息。')


def total_ms(rows):
    fields = sorted({k for row in rows for k in row if k.endswith('_ms')})
    return {field: sum(float(row.get(field, 0)) for row in rows) for field in fields}


def save_wav_analysis(directory, filenames):
    result = {}
    for filename in filenames:
        path = directory / filename
        if path.exists():
            samples, fs = read_wav(path)
            if samples.size:
                result[path.stem] = analyze(samples, fs)
                result[path.stem]['csv'] = save_analysis_csv(directory / path.stem, samples, fs)
    return result


def send(args):
    enc_key, mac_key = load_key(args.key)
    if not 0 < args.timeout <= 3600:
        raise ValueError('timeout 必须为正且不超过 3600 秒。')
    if args.wav:
        wav_samples, fs = read_wav(args.wav)
        if not wav_samples.size:
            raise ValueError('输入录音为空。')
        total_expected = wav_samples.size
        if total_expected > fs * 3600:
            raise ValueError('单次 WAV 传输不超过一小时（实现保护范围）。')
    else:
        if not math.isfinite(args.seconds) or not 0 < args.seconds <= 600:
            raise ValueError('录音时长须大于 0 且不超过 600 秒（实现保护范围）。')
        fs = args.sample_rate
        total_expected = int(round(fs * args.seconds))
        if total_expected < 1:
            raise ValueError('录音长度不足一个样本。')
    config = checked_config({'sample_rate': fs, 'block_samples': args.block_samples,
        'low_hz': args.low_hz, 'high_hz': args.high_hz, 'taps': args.taps,
        'tone_hz': args.tone_hz, 'tone_amplitude': args.tone_amplitude,
        'headroom': args.headroom, 'source_codec': args.source,
        'channel_codec': args.channel, 'channel_test': args.channel_test, 'ber': args.ber})
    prepare_run_directory(args.out)
    rows, sinks, connection, microphone = [], [], None, None
    summary = {'role': 'sender', 'status': 'incomplete', 'environment': environment(),
               'config': config, 'input_mode': 'wav' if args.wav else 'microphone',
               'expected_samples': int(total_expected), 'peer_host': args.host,
               'measurement_note': '往返时间含接收端解码、滤波、保存和ACK；不等于单向网络延迟。'}
    started, cpu_started = time.perf_counter(), time.process_time()
    sample_count, wire_bytes = 0, 0
    digest = hashlib.sha256()
    try:
        if args.mic:
            microphone = QueuedMicrophone(fs, args.block_samples, total_expected,
                                          device=args.input_device)
        for name in ['reference.wav', 'noisy.wav']:
            sinks.append(WaveSink(args.out / name, fs))
        connection = socket.create_connection((args.host, args.port), timeout=args.timeout)
        configure_connection(connection, args.timeout)
        summary['local_address'] = list(connection.getsockname())
        summary['peer_address'] = list(connection.getpeername())
        challenge = receive_control(connection, mac_key, 'challenge')
        session_id = challenge.get('session_id')
        if not isinstance(session_id, str) or len(session_id) != 32:
            raise ValueError('接收端会话随机数不合法。')
        summary['session_id'] = session_id
        client_nonce = secrets.token_hex(16)
        send_control(connection, {'type': 'start', 'session_id': session_id, 'client_nonce': client_nonce,
                     'config': config, 'expected_samples': int(total_expected)}, mac_key)
        ready = receive_control(connection, mac_key, 'ready')
        ensure_session(ready, session_id)
        if ready.get('client_nonce') != client_nonce:
            raise ValueError('接收端未确认本次发送随机数，拒绝旧会话响应。')
        if microphone:
            microphone.start()
        transfer_started = time.perf_counter()
        while sample_count < total_expected:
            n = min(args.block_samples, total_expected - sample_count)
            capture_started = time.perf_counter()
            if microphone:
                data, overflowed = microphone.read(n)
                if overflowed:
                    raise RuntimeError('麦克风缓冲溢出，采集已丢样本。本次运行记为失败，请增大帧长/关闭播放。')
                block = data[:, 0].astype(np.float64) / 32768
            else:
                block = wav_samples[sample_count:sample_count + n]
            capture_ms = (time.perf_counter() - capture_started) * 1000
            active_started = time.perf_counter()
            reference = np.asarray(block) * args.headroom
            noise_started = time.perf_counter()
            noisy = (add_tone(reference, fs, args.tone_hz, args.tone_amplitude, sample_count)
                     if args.tone_amplitude else reference.copy())
            noise_ms = (time.perf_counter() - noise_started) * 1000
            sequence = len(rows)
            packet, stats = encode_frame(noisy, sequence, fs, args.source, args.channel,
                                        enc_key, mac_key, session_id, args.channel_test, args.ber)
            roundtrip_started = time.perf_counter()
            wire_bytes += send_packet(connection, b'A' + packet)
            ack = receive_control(connection, mac_key, 'frame_ack')
            ensure_session(ack, session_id)
            if type(ack.get('sequence')) is not int or ack['sequence'] != sequence:
                raise ValueError('接收确认帧序号不一致。')
            turnaround_ms = (time.perf_counter() - roundtrip_started) * 1000
            digest.update(packet)
            sinks[0].write(reference)
            sinks[1].write(noisy)
            sample_count += n
            stats.update({'sequence': sequence, 'samples': n, 'capture_ms': capture_ms,
                          'noise_ms': noise_ms, 'turnaround_ms': turnaround_ms,
                          'active_ms': (time.perf_counter() - active_started) * 1000,
                          'audio_budget_ms': n / fs * 1000})
            rows.append(stats)
            print('发送帧 {:04d}，{} 样本，往返 {:.2f} ms'.format(sequence, n, turnaround_ms), flush=True)
            if args.pace and not microphone:
                remaining = sample_count / fs - (time.perf_counter() - transfer_started)
                if remaining > 0:
                    time.sleep(remaining)
        if microphone:
            microphone.stop()
        send_control(connection, {'type': 'end', 'session_id': session_id,
                     'frames': len(rows), 'samples': sample_count, 'sha256': digest.hexdigest()}, mac_key)
        done = receive_control(connection, mac_key, 'done')
        ensure_session(done, session_id)
        if done.get('frames') != len(rows) or done.get('samples') != sample_count:
            raise ValueError('接收端最终计数不一致。')
        summary.update({'status': 'complete', 'receiver_confirmation': done,
                        'transfer_session_seconds': time.perf_counter() - transfer_started})
    except Exception as exc:
        summary['error'] = str(exc)
        raise
    finally:
        cleanup_errors = close_resources([microphone, connection] + sinks)
        if cleanup_errors:
            summary.update(status='incomplete', cleanup_errors=cleanup_errors)
        summary.update({'frames': len(rows), 'samples': sample_count,
                        'audio_seconds': sample_count / fs, 'audio_framed_bytes': wire_bytes,
                        'wall_seconds': time.perf_counter() - started,
                        'cpu_seconds': time.process_time() - cpu_started, 'stage_total_ms': total_ms(rows),
                        'over_budget_frames': sum(row['active_ms'] > row['audio_budget_ms'] for row in rows)})
        if summary.get('transfer_session_seconds', 0) > 0:
            summary['effective_session_bps'] = wire_bytes * 8 / summary['transfer_session_seconds']
        save_rows(args.out / 'send_frames.csv', rows)
        save_json(args.out / 'send_summary.json', summary)
        if cleanup_errors:
            raise RuntimeError('资源关闭失败：' + '; '.join(cleanup_errors))
    summary['analysis'] = save_wav_analysis(args.out, ['reference.wav', 'noisy.wav'])
    save_json(args.out / 'send_summary.json', summary)
    print('发送完成，结果目录：' + str(args.out.resolve()))


def receive(args):
    enc_key, mac_key = load_key(args.key)
    if not 0 < args.timeout <= 3600:
        raise ValueError('timeout 必须为正且不超过 3600 秒。')
    prepare_run_directory(args.out)
    summary = {'role': 'receiver', 'status': 'incomplete', 'environment': environment(),
               'measurement_note': 'recv_wait含发送端采集/处理/节奏等待，不是网络单向延迟；不统计等待连接时间。'}
    sinks, rows, player = [], [], None
    samples_total, wire_bytes = 0, 0
    digest = hashlib.sha256()
    started, cpu_started = None, time.process_time()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.bind, args.port))
            server.listen(1)
            server.settimeout(args.timeout)
            print('等待发送端连接 {}:{}（{} 秒超时）'.format(args.bind, args.port, args.timeout), flush=True)
            connection, address = server.accept()
            with connection:
                started = time.perf_counter()
                configure_connection(connection, args.timeout)
                summary['peer_address'] = list(address)
                summary['local_address'] = list(connection.getsockname())
                session_id = secrets.token_hex(16)
                summary['session_id'] = session_id
                send_control(connection, {'type': 'challenge', 'session_id': session_id}, mac_key)
                hello = receive_control(connection, mac_key, 'start')
                ensure_session(hello, session_id)
                client_nonce = hello.get('client_nonce')
                if not isinstance(client_nonce, str) or len(client_nonce) != 32:
                    raise ValueError('发送端会话随机数不合法。')
                if not isinstance(hello.get('config'), dict):
                    raise ValueError('缺少运行配置。')
                config = checked_config(hello['config'])
                expected_samples = hello.get('expected_samples')
                if type(expected_samples) is not int or not 0 < expected_samples <= config['sample_rate'] * 3600:
                    raise ValueError('声明的音频长度非法。')
                summary.update({'config': config, 'expected_samples': expected_samples})
                fs = config['sample_rate']
                fir = FIRFilter(design_bandpass(fs, config['low_hz'], config['high_hz'], config['taps']))
                summary['filter_delay_samples'] = fir.delay_samples
                summary['filter_delay_ms'] = fir.delay_samples / fs * 1000
                for name in ['decoded_noisy.wav', 'recovered.wav']:
                    sinks.append(WaveSink(args.out / name, fs))
                if args.play:
                    sd = audio_module()
                    player = sd.RawOutputStream(samplerate=fs, channels=1, dtype='int16',
                                               device=args.output_device)
                    player.start()
                send_control(connection, {'type': 'ready', 'session_id': session_id,
                             'client_nonce': client_nonce}, mac_key)
                while True:
                    wait_started = time.perf_counter()
                    packet = recv_packet(connection)
                    wait_ms = (time.perf_counter() - wait_started) * 1000
                    if packet.startswith(b'C'):
                        end = parse_control(packet, mac_key, 'end')
                        ensure_session(end, session_id)
                        if (end.get('frames') != len(rows) or end.get('samples') != samples_total
                                or end.get('sha256') != digest.hexdigest() or samples_total != expected_samples):
                            raise ValueError('结束标记的总帧数、样本数或校验不一致。')
                        if player:
                            player.stop()
                        errors = close_resources([player] + sinks)
                        sinks.clear()
                        player = None
                        if errors:
                            raise RuntimeError('输出文件或音频设备关闭失败：' + '; '.join(errors))
                        send_control(connection, {'type': 'done', 'session_id': session_id,
                                     'frames': len(rows), 'samples': samples_total}, mac_key)
                        summary['status'] = 'complete'
                        break
                    if not packet.startswith(b'A'):
                        raise ValueError('未知消息类型。')
                    processing_started = time.perf_counter()
                    decoded, meta, stats = decode_frame(packet[1:], enc_key, mac_key, session_id)
                    n = min(config['block_samples'], expected_samples - samples_total)
                    if meta['sequence'] != len(rows) or n <= 0 or decoded.size != n:
                        raise ValueError('帧序号或每帧样本数不连续。')
                    for field in ('sample_rate', 'source_codec', 'channel_codec', 'channel_test', 'ber'):
                        if meta[field] != config[field]:
                            raise ValueError('网络帧参数与会话配置不一致：' + field)
                    filter_started = time.perf_counter()
                    recovered = fir.process(decoded)
                    filter_ms = (time.perf_counter() - filter_started) * 1000
                    sinks[0].write(decoded)
                    sinks[1].write(recovered)
                    underflowed = False
                    if player:
                        raw = np.clip(np.rint(recovered * 32768), -32768, 32767).astype('<i2').tobytes()
                        underflowed = bool(player.write(raw))
                    digest.update(packet[1:])
                    samples_total += decoded.size
                    wire_bytes += 4 + len(packet)
                    stats.update({'sequence': len(rows), 'samples': int(decoded.size),
                                  'recv_wait_ms': wait_ms, 'filter_ms': filter_ms,
                                  'processing_ms': (time.perf_counter() - processing_started) * 1000,
                                  'output_clipped_samples': int(np.count_nonzero(np.abs(recovered) > 1)),
                                  'playback_underflow': underflowed})
                    rows.append(stats)
                    send_control(connection, {'type': 'frame_ack', 'session_id': session_id,
                                 'sequence': meta['sequence']}, mac_key)
                    print('接收帧 {:04d}，{} 样本，认证通过'.format(meta['sequence'], decoded.size), flush=True)
    except Exception as exc:
        summary['error'] = str(exc)
        raise
    finally:
        cleanup_errors = close_resources([player] + sinks)
        if cleanup_errors:
            summary.update(status='incomplete', cleanup_errors=cleanup_errors)
        summary.update({'frames': len(rows), 'samples': samples_total,
                        'audio_framed_bytes': wire_bytes,
                        'wall_seconds': time.perf_counter() - started if started else None,
                        'cpu_seconds': time.process_time() - cpu_started,
                        'stage_total_ms': total_ms(rows),
                        'playback_underflows': sum(row['playback_underflow'] for row in rows)})
        if 'config' in summary:
            summary['audio_seconds'] = samples_total / summary['config']['sample_rate']
        save_rows(args.out / 'receive_frames.csv', rows)
        save_json(args.out / 'receive_summary.json', summary)
        if cleanup_errors:
            raise RuntimeError('资源关闭失败：' + '; '.join(cleanup_errors))
    summary['analysis'] = save_wav_analysis(args.out, ['decoded_noisy.wav', 'recovered.wav'])
    save_json(args.out / 'receive_summary.json', summary)
    print('接收完成，结果目录：' + str(args.out.resolve()))


def evaluate(args):
    status_checks = {}
    for label, path in [('sender', args.reference.parent / 'send_summary.json'),
                        ('receiver', args.received.parent / 'receive_summary.json')]:
        if path.exists():
            run = json.loads(path.read_text(encoding='utf-8'))
            if run.get('status') != 'complete':
                raise ValueError('禁止评价未完成的运行：' + str(path))
            status_checks[label] = 'complete'
        else:
            status_checks[label] = '无相邻运行摘要，仅进行离线文件评价'
    reference, fs = read_wav(args.reference)
    recovered, received_fs = read_wav(args.received)
    if fs != received_fs:
        raise ValueError('采样率不同，不能直接计算逐样本质量指标。')
    result = {'reference': str(args.reference.resolve()), 'received': str(args.received.resolve()),
              'run_status_checks': status_checks,
              'sample_rate': fs, 'reference_samples': int(reference.size),
              'received_samples': int(recovered.size),
              'against_clean_delay_aligned': metrics(reference, recovered, args.delay_samples)}
    # 同一 FIR 处理干净参考，区分滤波本身改变频带与传输/编码造成的误差。
    if args.taps:
        coefficients = design_bandpass(fs, args.low_hz, args.high_hz, args.taps)
        filtered_reference = FIRFilter(coefficients).process(reference)
        result['against_same_filter_clean'] = metrics(filtered_reference, recovered)
        result['same_filter_reference'] = {'low_hz': args.low_hz, 'high_hz': args.high_hz,
                                            'taps': args.taps}
    if args.decoded:
        decoded, decoded_fs = read_wav(args.decoded)
        if decoded_fs != fs:
            raise ValueError('decoded 文件采样率不一致。')
        result['decoded_noisy_against_clean'] = metrics(reference, decoded)
    result['note'] = '真实语音质量来自本次文件；干净参考须使用发送端保存的reference.wav（已乘headroom）。'
    save_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


def compare(args):
    rows = []
    for path in args.runs:
        run = json.loads(path.read_text(encoding='utf-8'))
        config = run.get('config', {})
        row = {'file': str(path), 'role': run.get('role'), 'status': run.get('status'),
               'os': run.get('environment', {}).get('os'),
               'machine': run.get('environment', {}).get('machine'),
               'session_id': run.get('session_id'),
               'sample_rate': config.get('sample_rate'), 'source': config.get('source_codec'),
               'channel': config.get('channel_codec'), 'channel_test': config.get('channel_test'),
               'samples': run.get('samples'), 'frames': run.get('frames'),
               'audio_seconds': run.get('audio_seconds'), 'wall_seconds': run.get('wall_seconds'),
               'cpu_seconds': run.get('cpu_seconds'), 'audio_framed_bytes': run.get('audio_framed_bytes'),
               'effective_session_bps': run.get('effective_session_bps')}
        row.update(run.get('stage_total_ms', {}))
        rows.append(row)
    save_rows(args.out, rows)
    print('已汇总到 ' + str(args.out.resolve()) + '；发送与接收角色耗时不可直接当作相同任务的平台性能对比。')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    key = commands.add_parser('keygen', help='生成两端共用的随机密钥文件')
    key.add_argument('--out', type=Path, default=Path('shared.key'))
    key.set_defaults(func=keygen)
    rx = commands.add_parser('receive', help='在接收设备先运行')
    rx.add_argument('--bind', default='0.0.0.0')
    rx.add_argument('--out', type=Path, default=Path('results/receiver'))
    rx.add_argument('--play', action='store_true')
    rx.add_argument('--output-device', help='sounddevice 输出设备编号或名称')
    rx.set_defaults(func=receive)
    tx = commands.add_parser('send', help='在另一设备运行发送端')
    tx.add_argument('--host', required=True, help='接收设备实际IP')
    source = tx.add_mutually_exclusive_group(required=True)
    source.add_argument('--wav', type=Path)
    source.add_argument('--mic', action='store_true')
    tx.add_argument('--seconds', type=float, default=5)
    tx.add_argument('--sample-rate', type=int, default=16000, help='仅麦克风；WAV使用文件本身采样率')
    tx.add_argument('--block-samples', type=int, default=1024, help='实现建议，不是课件阈值')
    tx.add_argument('--headroom', type=float, default=0.75)
    tx.add_argument('--tone-hz', type=float, default=5000)
    tx.add_argument('--tone-amplitude', type=float, default=0.2, help='设为0关闭人工正弦加噪')
    tx.add_argument('--low-hz', type=float, default=300)
    tx.add_argument('--high-hz', type=float, default=3400)
    tx.add_argument('--taps', type=int, default=129, help='自编带通FIR奇数抽头数')
    tx.add_argument('--source', choices=['dpcm8', 'pcm16'], default='dpcm8')
    tx.add_argument('--channel', choices=['hamming74', 'none'], default='hamming74')
    tx.add_argument('--channel-test', choices=['none', 'single', 'random'], default='none')
    tx.add_argument('--ber', type=float, default=0)
    tx.add_argument('--pace', action='store_true', help='WAV按原音频时长发送；吞吐测试不加此项')
    tx.add_argument('--input-device', help='sounddevice 输入设备编号或名称')
    tx.add_argument('--out', type=Path, default=Path('results/sender'))
    tx.set_defaults(func=send)
    for cmd in (rx, tx):
        cmd.add_argument('--port', type=int, default=5000)
        cmd.add_argument('--key', type=Path, default=Path('shared.key'))
        cmd.add_argument('--timeout', type=float, default=60, help='网络等待秒数，非课件时延标准')
    ev = commands.add_parser('evaluate', help='复制两端结果到同一设备后计算质量')
    ev.add_argument('--reference', type=Path, required=True)
    ev.add_argument('--received', type=Path, required=True)
    ev.add_argument('--decoded', type=Path, help='可选接收端decoded_noisy.wav')
    ev.add_argument('--delay-samples', type=int, default=64, help='129tap线性相位FIR的群延迟')
    ev.add_argument('--taps', type=int, default=129, help='设0不计算同滤波参考对比')
    ev.add_argument('--low-hz', type=float, default=300)
    ev.add_argument('--high-hz', type=float, default=3400)
    ev.add_argument('--out', type=Path, default=Path('results/evaluation.json'))
    ev.set_defaults(func=evaluate)
    cp = commands.add_parser('compare', help='把多次运行的性能JSON汇总成CSV')
    cp.add_argument('--runs', type=Path, nargs='+', required=True)
    cp.add_argument('--out', type=Path, default=Path('results/comparison.csv'))
    cp.set_defaults(func=compare)
    dev = commands.add_parser('audio-devices', help='列出本机音频设备')
    dev.set_defaults(func=lambda args: print(audio_module().query_devices()))
    return parser


def main():
    args = build_parser().parse_args()
    if hasattr(args, 'port') and not 1 <= args.port <= 65535:
        raise ValueError('端口必须为1～65535。')
    # sounddevice 的数字设备索引需为整数，设备名称则保留字符串。
    for field in ('input_device', 'output_device'):
        value = getattr(args, field, None)
        if value is not None and value.isdecimal():
            setattr(args, field, int(value))
    args.func(args)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, ConnectionError, wave.Error) as exc:
        print('运行失败：' + str(exc), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('用户中止运行，已保存的 incomplete 结果不能作为完整传输。', file=sys.stderr)
        sys.exit(130)
