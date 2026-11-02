"""TCP 字节流分帧和带 HMAC 的会话控制消息。Python 3.8+。"""
import hashlib
import hmac
import json
import socket
import struct

LENGTH = struct.Struct('!I')
MAX_PACKET_BYTES = 2 * 1024 * 1024  # 实现的资源保护上限，不是课件阈值


def recv_exact(connection, size):
    result = bytearray()
    while len(result) < size:
        block = connection.recv(size - len(result))
        if not block:
            raise ConnectionError('连接提前断开：没有收到完整消息或结束标记。')
        result.extend(block)
    return bytes(result)


def send_packet(connection, packet):
    if not 0 < len(packet) <= MAX_PACKET_BYTES:
        raise ValueError('消息长度超出程序允许范围。')
    connection.sendall(LENGTH.pack(len(packet)) + packet)
    return LENGTH.size + len(packet)


def recv_packet(connection):
    length, = LENGTH.unpack(recv_exact(connection, LENGTH.size))
    if not 0 < length <= MAX_PACKET_BYTES:
        raise ValueError('收到非法消息长度。')
    return recv_exact(connection, length)


def control_packet(message, mac_key):
    body = json.dumps(message, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')
    if len(body) > 16384:
        raise ValueError('控制消息过长。')
    tag = hmac.new(mac_key, b'EX5-control\x00' + body, hashlib.sha256).digest()
    return b'C' + tag + body


def parse_control(packet, mac_key, expected_type=None):
    if not packet.startswith(b'C') or not 33 < len(packet) <= 16417:
        raise ValueError('控制消息格式错误。')
    tag, body = packet[1:33], packet[33:]
    actual = hmac.new(mac_key, b'EX5-control\x00' + body, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, actual):
        raise ValueError('控制消息认证失败，请检查两端密钥。')
    try:
        message = json.loads(body.decode('utf-8'))
    except (UnicodeError, ValueError) as exc:
        raise ValueError('控制消息不是合法 JSON。') from exc
    if not isinstance(message, dict) or (expected_type and message.get('type') != expected_type):
        raise ValueError('控制消息类型不符合当前通信步骤。')
    return message


def send_control(connection, message, mac_key):
    return send_packet(connection, control_packet(message, mac_key))


def receive_control(connection, mac_key, expected_type):
    return parse_control(recv_packet(connection), mac_key, expected_type)


def configure_connection(connection, timeout):
    connection.settimeout(timeout)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
