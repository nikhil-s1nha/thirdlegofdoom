"""Frame recovery for the UART link.

Pure framing logic, no serial port involved -- these exercise exactly
the property a byte stream (unlike a UDP datagram) can violate: bytes
arriving split across reads, several frames arriving in one read, and
noise that happens to start with the magic bytes.
"""

from tlod.net.uart_link import MAGIC, FrameDecoder, encode_frame


def test_round_trip_single_frame():
    decoder = FrameDecoder()
    frame = encode_frame(b"hello")
    assert decoder.feed(frame) == [b"hello"]


def test_frame_split_across_reads():
    decoder = FrameDecoder()
    frame = encode_frame(b"hello world")
    assert decoder.feed(frame[:3]) == []
    assert decoder.feed(frame[3:7]) == []
    assert decoder.feed(frame[7:]) == [b"hello world"]


def test_multiple_frames_in_one_read():
    decoder = FrameDecoder()
    data = encode_frame(b"one") + encode_frame(b"two") + encode_frame(b"three")
    assert decoder.feed(data) == [b"one", b"two", b"three"]


def test_leading_garbage_is_skipped():
    decoder = FrameDecoder()
    data = b"\x00\x01garbage" + encode_frame(b"payload")
    assert decoder.feed(data) == [b"payload"]


def test_corrupt_checksum_is_dropped_not_returned():
    decoder = FrameDecoder()
    frame = bytearray(encode_frame(b"payload"))
    frame[-1] ^= 0xFF  # flip the checksum byte
    assert decoder.feed(bytes(frame)) == []
    # The decoder should have resynced, not gotten stuck -- a good frame
    # right after the corrupt one is still recovered.
    assert decoder.feed(encode_frame(b"next")) == [b"next"]


def test_magic_bytes_inside_payload_do_not_confuse_framing():
    decoder = FrameDecoder()
    data = encode_frame(MAGIC + b"looks like a header but is just data")
    assert decoder.feed(data) == [MAGIC + b"looks like a header but is just data"]


def test_empty_payload_round_trips():
    decoder = FrameDecoder()
    assert decoder.feed(encode_frame(b"")) == [b""]
