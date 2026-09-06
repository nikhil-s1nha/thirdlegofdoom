"""Wire format between the vision board and the control board.

Deliberately small and dumb: one UDP datagram per detection, plain JSON,
under 300 bytes. Frames never cross the wire -- only what was found in
them. A 720p MJPEG stream is ~30 Mbit/s and would buy nothing, since the
control board has no use for pixels.

Why UDP rather than TCP. This is a latest-value stream, not a log. TCP
guarantees ordered delivery, which sounds desirable until a delayed
packet holds up the newer ones queued behind it -- head-of-line blocking
turns one hiccup into a stall. A dropped UDP datagram costs nothing here,
because another arrives in ~16 ms with fresher data. Sequence numbers
catch reordering; anything older than what we hold is discarded.

Coordinates are already in the **robot base frame**. The vision board
owns the camera calibration and does the projection, so the control board
needs no intrinsics, no extrinsics, and no knowledge of where the camera
is. Moving the camera then requires recalibrating exactly one machine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from tlod.types import HandObservation, Perception

PROTOCOL_VERSION = 1
MAX_DATAGRAM = 1400        # comfortably inside a 1500-byte MTU, no fragmentation


@dataclass(slots=True)
class Packet:
    """One perception snapshot, as it travels."""

    seq: int
    stamp: float                    # shutter time, in the SENDER's clock
    hands: list[dict]
    objects: list[dict]
    sent: float = 0.0               # send time, sender's clock, for offset estimation

    def encode(self) -> bytes:
        payload = {
            "v": PROTOCOL_VERSION,
            "seq": self.seq,
            "t": round(self.stamp, 6),
            "s": round(self.sent, 6),
            "h": self.hands,
            "o": self.objects,
        }
        return json.dumps(payload, separators=(",", ":")).encode()

    @staticmethod
    def decode(data: bytes) -> Packet | None:
        try:
            d = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return None
        if d.get("v") != PROTOCOL_VERSION:
            return None
        return Packet(
            seq=int(d.get("seq", 0)),
            stamp=float(d.get("t", 0.0)),
            hands=d.get("h", []),
            objects=d.get("o", []),
            sent=float(d.get("s", 0.0)),
        )


def _round3(v) -> list[float]:
    # Millimetre resolution is well past what the vision can support, and
    # keeps a datagram small enough never to fragment.
    return [round(float(x), 4) for x in np.asarray(v, float).reshape(-1)]


def encode_perception(perception: Perception, seq: int, sent: float) -> Packet:
    hands = []
    for h in perception.hands:
        entry = {"p": _round3(h.position), "c": round(float(h.confidence), 3)}
        if h.velocity is not None:
            entry["v"] = _round3(h.velocity)
        if h.handedness != "unknown":
            entry["s"] = h.handedness
        hands.append(entry)
    objects = [
        {"l": d.label, "p": _round3(d.position), "c": round(float(d.confidence), 3),
         "r": round(float(d.radius), 4)}
        for d in perception.objects
    ]
    return Packet(seq=seq, stamp=perception.stamp, hands=hands, objects=objects, sent=sent)


def decode_perception(packet: Packet, clock_offset: float = 0.0) -> Perception:
    """Rebuild a Perception, translating the sender's clock into ours.

    `clock_offset` is added to the shutter timestamp. Without it every age
    computed downstream is wrong by the difference between two machines'
    clocks -- and wrong silently, which is worse than wrong loudly.
    """
    from tlod.types import Detection

    stamp = packet.stamp + clock_offset
    hands = [
        HandObservation(
            position=np.array(h["p"], float),
            stamp=stamp,
            velocity=np.array(h["v"], float) if "v" in h else None,
            handedness=h.get("s", "unknown"),
            confidence=float(h.get("c", 1.0)),
        )
        for h in packet.hands
    ]
    objects = [
        Detection(
            label=o.get("l", "object"),
            position=np.array(o["p"], float),
            stamp=stamp,
            confidence=float(o.get("c", 1.0)),
            radius=float(o.get("r", 0.0)),
        )
        for o in packet.objects
    ]
    return Perception(stamp=stamp, hands=hands, objects=objects, frame=None)
