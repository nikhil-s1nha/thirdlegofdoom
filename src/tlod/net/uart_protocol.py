"""Line framing for the UART link.

`tlod.net.protocol` defines *what* crosses the wire (a `Packet`, JSON,
under 300 bytes); this defines *how* it is delimited when the wire is a
raw UART byte stream instead of UDP.

UDP gives two things away for free that a serial port does not:

  * **Message boundaries.** One `recvfrom()` returns exactly one
    datagram. A UART port is just a stream of bytes -- nothing marks
    where one message ends and the next begins, so this adds a newline
    delimiter. The JSON payloads here are numbers, short labels and
    single ASCII words; none of that can contain a literal `\\n`, so a
    plain split on it is enough. No length-prefix, no escaping.

  * **Two independent channels.** The UDP link uses separate sockets
    for detections (port 45800) and clock probes (port 45801), because
    a second socket is free. A UART link is one shared wire, so this
    multiplexes both onto it with an explicit "k" (kind) tag read
    before anything else about the line is interpreted.

Garbage on the wire -- a line torn by a mid-write reset, noise on a
longer cable run than intended -- decodes to `None` here rather than
raising. The caller counts it and moves on; a corrupted line is exactly
as harmless as a dropped UDP datagram, provided the caller treats it
that way.
"""

from __future__ import annotations

import json

from tlod.net.protocol import PROTOCOL_VERSION, Packet

# Real STS3215 control frames run at 1 Mbaud on this hardware; the link
# to another board doesn't share that bus, so it is free to pick its own
# rate. 115200 is the highest rate most USB-TTL adapters and SBC UARTs
# agree on without extra configuration, and a ~150-300 byte JSON line
# fits inside one 16 ms frame interval with room to spare even there.
DEFAULT_BAUD = 115200


def encode_data(packet: Packet) -> bytes:
    """Frame one perception snapshot as a line."""
    obj = {
        "k": "data",
        "v": PROTOCOL_VERSION,
        "seq": packet.seq,
        "t": round(packet.stamp, 6),
        "s": round(packet.sent, 6),
        "h": packet.hands,
        "o": packet.objects,
    }
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode()


def encode_clock(kind: str, **fields) -> bytes:
    """Frame a clock probe or reply. `kind` is "ping" or "pong"."""
    obj = {"k": kind, **fields}
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode()


def decode_line(line: bytes) -> tuple[str, dict] | None:
    """Split one received line into its kind tag and payload.

    Returns `None` for anything that isn't a JSON object carrying a
    string "k" -- malformed JSON, a stray newline, a half-written line
    read across a buffer boundary. Never raises.
    """
    try:
        obj = json.loads(line)
    except (ValueError, UnicodeDecodeError):
        return None
    kind = obj.get("k")
    if not isinstance(kind, str):
        return None
    return kind, obj


def data_packet(obj: dict) -> Packet | None:
    """Rebuild a `Packet` from a decoded "data" line's payload."""
    return Packet.from_dict(obj)
