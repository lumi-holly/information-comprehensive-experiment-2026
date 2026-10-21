from __future__ import annotations

import socket
import struct


LENGTH_HEADER = struct.Struct("!I")
MAX_PACKET_SIZE = 16 * 1024 * 1024


def recv_exact(connection: socket.socket, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            return None if not chunks else bytes(chunks)
        chunks.extend(chunk)
    return bytes(chunks)


def send_packet(connection: socket.socket, packet: bytes) -> None:
    connection.sendall(LENGTH_HEADER.pack(len(packet)) + packet)


def recv_packet(connection: socket.socket) -> bytes | None:
    length_data = recv_exact(connection, LENGTH_HEADER.size)
    if length_data is None:
        return None
    if len(length_data) != LENGTH_HEADER.size:
        raise ConnectionError("连接在接收帧长度时中断。")
    (length,) = LENGTH_HEADER.unpack(length_data)
    if length <= 0 or length > MAX_PACKET_SIZE:
        raise ValueError(f"非法网络帧长度：{length}")
    packet = recv_exact(connection, length)
    if packet is None or len(packet) != length:
        raise ConnectionError("连接在接收网络帧时中断。")
    return packet

