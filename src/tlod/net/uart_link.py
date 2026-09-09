"""Byte-stream framing for the UART link (Orange Pi <-> Raspberry Pi).

This is the "learning/testing" link, not the production one -- see
`docs/deployment.md` for why UDP over Ethernet is what actually ships.
It exists so the two-board split can be rehearsed over a direct 3-wire
serial connection when there is no network between the boards at all.

UART has no packet boundaries the way a UDP datagram does: bytes just
arrive, sometimes split across reads, sometimes with several frames
already queued. `encode_frame`/`FrameDecoder` recover the same framing
a datagram gives you for free -- magic bytes to find a candidate start,
a length prefix to know where it ends, and a checksum so a byte pattern
that happens to start with the magic can't be mistaken for a real frame.

Deliberately reuses `tlod.net.protocol.Packet` for the payload itself.
The wire format that already exists, is tested, and is known to fit in
one datagram does not need reinventing just because the transport
underneath it changed; only the framing around it does.
"""

from __future__ import annotations

import struct

MAGIC = b"\xa5\x5a"
_HEADER = struct.Struct(">2sH")  # magic, payload length (uint16, big-endian)
MAX_PAYLOAD = 2048


def encode_frame(payload: bytes) -> bytes:
    """Wrap `payload` for transmission: magic + length + payload + checksum."""
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload too large for one frame: {len(payload)} bytes")
    checksum = sum(payload) & 0xFF
    return _HEADER.pack(MAGIC, len(payload)) + payload + bytes([checksum])


class FrameDecoder:
    """Recovers framed payloads from a raw UART byte stream.

    Feed it whatever `serial.Serial.read()` returns, in whatever chunks
    it arrives in; it buffers internally and only emits a payload once
    the checksum agrees, so noise or a mid-frame connect cannot produce
    a decoded payload out of garbage. Not thread-safe -- feed it from
    one reader thread only, same as the UDP path only ever has one
    socket-reading thread.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        self._buf.extend(data)
        out: list[bytes] = []
        while True:
            idx = self._buf.find(MAGIC)
            if idx == -1:
                # Keep the last byte: it might be the first half of a
                # magic sequence split across two reads.
                if len(self._buf) > 1:
                    del self._buf[:-1]
                break
            if idx > 0:
                del self._buf[:idx]
            if len(self._buf) < _HEADER.size:
                break
            _, length = _HEADER.unpack_from(self._buf)
            if length > MAX_PAYLOAD:
                # Not a real header -- just two bytes that happened to
                # match the magic. Skip past them and keep scanning.
                del self._buf[:2]
                continue
            end = _HEADER.size + length + 1
            if len(self._buf) < end:
                break
            payload = bytes(self._buf[_HEADER.size : end - 1])
            checksum = self._buf[end - 1]
            del self._buf[:end]
            if (sum(payload) & 0xFF) != checksum:
                continue  # corrupt frame; already dropped, keep scanning
            out.append(payload)
        return out
