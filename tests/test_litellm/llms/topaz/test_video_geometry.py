"""
NOL-519: source geometry read from the bytes the Topaz create leg already holds.

Topaz's free `POST /video/` estimate endpoint prices SUPPLIED GEOMETRY, which is
what makes it deterministic and off-queue - but it means the caller has to
describe the footage. This parser reads that description out of the source bytes
the create leg already downloaded for the upload PUT, so the quote costs no
extra network I/O.

The boxes here are synthesised rather than loaded from a fixture so every field
under test is set explicitly and the failure mode is obvious. Cross-checked
against real ffmpeg output while writing: a 640x360 24fps 13s clip parsed to
exactly 640x360 / 24.0 / 13.0 / 312 frames, a 1920x1080 29.97fps 5.005s clip to
1920x1080 / 29.97 / 5.005 / 150, and a 1080x1920 60fps 3s .mov to
1080x1920 / 60.0 / 3.0 / 180, each matching ffprobe exactly.
"""

import pytest

from litellm.llms.topaz.video_geometry import SourceGeometry, parse_video_geometry


def _box(box_type: bytes, payload: bytes) -> bytes:
    return (len(payload) + 8).to_bytes(4, "big") + box_type + payload


def _full_box(box_type: bytes, payload: bytes, version: int = 0) -> bytes:
    return _box(box_type, bytes((version,)) + b"\x00\x00\x00" + payload)


def _mvhd(timescale: int = 1000, duration: int = 13000) -> bytes:
    return _full_box("mvhd".encode(), bytes(8) + timescale.to_bytes(4, "big") + duration.to_bytes(4, "big"))


def _mdhd(timescale: int, duration: int, version: int = 0) -> bytes:
    if version == 1:
        payload = bytes(16) + timescale.to_bytes(4, "big") + duration.to_bytes(8, "big")
    else:
        payload = bytes(8) + timescale.to_bytes(4, "big") + duration.to_bytes(4, "big")
    return _full_box(b"mdhd", payload, version=version)


def _hdlr(handler: bytes = b"vide") -> bytes:
    return _full_box(b"hdlr", bytes(4) + handler + bytes(12))


def _tkhd(width: int, height: int, version: int = 0) -> bytes:
    header = bytes(20) if version == 0 else bytes(32)
    fixed = (width << 16).to_bytes(4, "big") + (height << 16).to_bytes(4, "big")
    return _full_box(b"tkhd", header + bytes(16) + bytes(36) + fixed, version=version)


def _stsd(width: int, height: int) -> bytes:
    entry = _box(b"avc1", bytes(6) + bytes(2) + bytes(16) + width.to_bytes(2, "big") + height.to_bytes(2, "big"))
    return _full_box(b"stsd", (1).to_bytes(4, "big") + entry)


def _stts(sample_count: int, delta: int = 1) -> bytes:
    table = sample_count.to_bytes(4, "big") + delta.to_bytes(4, "big")
    return _full_box(b"stts", (1).to_bytes(4, "big") + table)


def _mp4(
    coded: tuple[int, int] | None = (640, 360),
    display: tuple[int, int] = (640, 360),
    timescale: int = 24000,
    duration: int = 312000,
    samples: int = 312,
    handler: bytes = b"vide",
    mdhd_version: int = 0,
) -> bytes:
    """A minimal but structurally valid ISO-BMFF file carrying one track."""
    stbl_children = _stts(samples) + (_stsd(*coded) if coded is not None else b"")
    stbl = _box(b"stbl", stbl_children)
    minf = _box(b"minf", stbl)
    mdia = _box(b"mdia", _mdhd(timescale, duration, version=mdhd_version) + _hdlr(handler) + minf)
    trak = _box(b"trak", _tkhd(*display) + mdia)
    moov = _box(b"moov", _mvhd() + trak)
    return _box(b"ftyp", b"isom" + bytes(4) + b"isomiso2") + moov


class TestParsesRealGeometry:
    def test_reads_every_field(self):
        geometry = parse_video_geometry(_mp4())
        assert geometry == SourceGeometry(width=640, height=360, duration_seconds=13.0, frame_rate=24.0)

    def test_frame_count_is_what_topaz_bills_for(self):
        """Topaz bills per frame processed, so this is the quantity that prices the job."""
        assert parse_video_geometry(_mp4()).frame_count == 312

    def test_fractional_ntsc_rate_survives(self):
        """29.97fps is 30000/1001; rounding it to 30 would misprice every NTSC source."""
        geometry = parse_video_geometry(_mp4(timescale=30000, duration=150150, samples=150))
        assert geometry.frame_rate == pytest.approx(29.97, abs=0.01)
        assert geometry.duration_seconds == pytest.approx(5.005)
        assert geometry.frame_count == 150

    def test_portrait_dimensions_are_not_transposed(self):
        """A 9:16 source quoted as 16:9 prices a different job; NOL-478 measured a 3.3x swing."""
        geometry = parse_video_geometry(_mp4(coded=(1080, 1920), display=(1080, 1920)))
        assert (geometry.width, geometry.height) == (1080, 1920)

    def test_coded_dimensions_win_over_display_dimensions(self):
        """
        tkhd carries display dimensions, which anamorphic footage stretches away
        from the frames the engine actually processes. Topaz runs on the coded
        frames, so the sample entry is authoritative.
        """
        geometry = parse_video_geometry(_mp4(coded=(1440, 1080), display=(1920, 1080)))
        assert (geometry.width, geometry.height) == (1440, 1080)

    def test_falls_back_to_tkhd_when_there_is_no_sample_entry(self):
        geometry = parse_video_geometry(_mp4(coded=None, display=(1280, 720)))
        assert (geometry.width, geometry.height) == (1280, 720)

    def test_reads_64_bit_mdhd(self):
        """Version 1 widens the timestamps and moves every field after them."""
        geometry = parse_video_geometry(_mp4(mdhd_version=1))
        assert geometry.duration_seconds == pytest.approx(13.0)
        assert geometry.frame_rate == pytest.approx(24.0)


class TestRefusesWhatItCannotRead:
    """
    Every one of these must yield None, never a guess. A wrong number produces a
    plausible-looking quote that silently misprices; None records no cost, which
    is the state the NOL-535 ledger guard exists to catch.
    """

    def test_matroska_is_not_parsed(self):
        """mkv is EBML, not ISO-BMFF. It falls back to the caller's declared geometry."""
        assert parse_video_geometry(b"\x1a\x45\xdf\xa3" + bytes(64)) is None

    def test_empty_and_truncated_buffers(self):
        assert parse_video_geometry(b"") is None
        assert parse_video_geometry(_mp4()[:40]) is None

    def test_audio_only_file_has_no_video_track(self):
        assert parse_video_geometry(_mp4(handler=b"soun")) is None

    def test_missing_sample_table_yields_no_frame_rate(self):
        moov = _box(b"moov", _mvhd() + _box(b"trak", _tkhd(640, 360) + _box(b"mdia", _mdhd(24000, 312000) + _hdlr())))
        assert parse_video_geometry(moov) is None

    def test_zero_duration_is_refused(self):
        assert parse_video_geometry(_mp4(duration=0)) is None

    def test_implausible_frame_rate_is_refused(self):
        """A parse that latched onto the wrong box shows up as a nonsense rate."""
        assert parse_video_geometry(_mp4(samples=500_000)) is None
        assert parse_video_geometry(_mp4(samples=1, timescale=1000, duration=600_000)) is None

    def test_a_box_claiming_to_exceed_its_parent_stops_the_walk(self):
        """A size field is attacker-controlled data; overrunning it must not read past the buffer."""
        corrupt = bytearray(_mp4())
        moov_at = corrupt.index(b"moov") - 4
        corrupt[moov_at : moov_at + 4] = (len(corrupt) * 4).to_bytes(4, "big")
        assert parse_video_geometry(bytes(corrupt)) is None


class TestFrameCount:
    def test_never_zero(self):
        """A sub-frame clip still processes a frame; a zero would quote a job that does nothing."""
        assert SourceGeometry(width=16, height=16, duration_seconds=0.01, frame_rate=24.0).frame_count == 1

    def test_tracks_duration_and_rate(self):
        assert SourceGeometry(width=16, height=16, duration_seconds=10.0, frame_rate=60.0).frame_count == 600
