"""
Source geometry for Topaz's free cost-estimate endpoint (NOL-519).

Topaz's `POST /video/` prices SUPPLIED GEOMETRY rather than a queued job, which
is what makes it usable on the create leg: it is free, synchronous and off the
queue, so it answers deterministically instead of waiting on a status
transition whose timing is set by Topaz's queue depth (measured at 2.6s on one
restore and still pending at 68s on the next, with identical inputs).

The price of that determinism is that the caller must describe the footage.
This module reads that description out of the source bytes the create leg has
ALREADY downloaded in order to PUT them to the presigned upload URL, so the
quote costs no extra network I/O - only a parse of a buffer that is in memory
regardless.

Reading it here rather than accepting it from the client is deliberate. A
client-supplied geometry field is present only for the clients that send it, so
every other caller of the proxy would keep recording $0 - a PARTIALLY populated
COGS ledger, which is strictly worse than a cleanly empty one: it looks healthy
while understating by an unpredictable amount, and it permanently mutes the
NOL-535 guard, which only fires for a model that has NEVER recorded a cost.
An explicit override is still honored (see `_SOURCE_GEOMETRY_PARAMS` in
videos/transformation.py) for callers that already know their footage exactly,
and for containers this parser does not read.

Scope: ISO base media file format - `mp4` and `mov`, 2 of Topaz's 3 accepted
source containers, including the fragmented variant whose sample tables live in
`moof` boxes rather than the initialization `moov`. Matroska (`mkv`) is a
different container format entirely
(EBML) and is not parsed; it returns None and falls back to the override, which
is honest rather than guessed.

Stdlib only, and total: it never raises, because a bookkeeping read must not be
able to fail a job whose footage Topaz has already accepted.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

# ISO/IEC 14496-12 box header: a 32-bit size followed by a 4-character type.
_HEADER_BYTES = 8

# A `size` of 1 means the real size is a 64-bit value directly after the type;
# a `size` of 0 means the box runs to the end of its parent.
_SIZE_IS_64BIT = 1
_SIZE_TO_END = 0

# tkhd/mvhd/mdhd carry a 1-byte version and 3 flag bytes before their payload.
_FULL_BOX_PREFIX = 4

# Offsets from the end of a tkhd full-box prefix to its 16.16 fixed-point
# width. Version 0 stores 32-bit creation/modification/duration, version 1
# stores 64-bit, and both are followed by 16 reserved/layer/volume bytes and a
# 36-byte transform matrix.
_TKHD_DIMENSION_OFFSET = {  # mutable-ok: frozen constant lookup table, never mutated after definition
    0: 20 + 16 + 36,
    1: 32 + 16 + 36,
}

# Offsets from the end of a tkhd full-box prefix to its track_ID, which is what
# ties a track to the movie fragments carrying its samples.
_TKHD_TRACK_ID_OFFSET = {  # mutable-ok: frozen constant lookup table, never mutated after definition
    0: 8,
    1: 16,
}

# Offsets from the end of an mvhd/mdhd full-box prefix to its timescale. Both
# boxes share this layout, which is why one reader serves both.
_TIMESCALE_OFFSET = {  # mutable-ok: frozen constant lookup table, never mutated after definition
    0: 8,
    1: 16,
}

# tfhd/trun flag bits (ISO/IEC 14496-12 8.8.7 and 8.8.8). Both boxes are
# variable-length: every optional field present shifts the ones after it.
_TFHD_BASE_DATA_OFFSET = 0x000001
_TFHD_SAMPLE_DESCRIPTION_INDEX = 0x000002
_TFHD_DEFAULT_SAMPLE_DURATION = 0x000008

_TRUN_DATA_OFFSET = 0x000001
_TRUN_FIRST_SAMPLE_FLAGS = 0x000004
_TRUN_SAMPLE_DURATION = 0x000100
_TRUN_OPTIONAL_SAMPLE_FIELDS = (
    _TRUN_SAMPLE_DURATION,
    0x000200,  # sample_size
    0x000400,  # sample_flags
    0x000800,  # sample_composition_time_offset
)

# Width/height in a visual sample entry, measured from the entry's start: an
# 8-byte box header, 6 reserved bytes, a 2-byte data-reference index, then
# 16 bytes of pre-defined/reserved fields.
_SAMPLE_ENTRY_WIDTH_OFFSET = 8 + 6 + 2 + 16

# A 16.16 fixed-point value's fractional divisor.
_FIXED_POINT_16_16 = 65536.0

# Frame rates outside this range mean the parse latched onto the wrong box.
# Rejecting them keeps a misread from reaching Topaz as a plausible-looking
# quote request; the caller then records no cost, which is detectable.
_MIN_PLAUSIBLE_FPS = 1.0
_MAX_PLAUSIBLE_FPS = 480.0


@dataclass(frozen=True, slots=True)
class SourceGeometry:
    """The footage description Topaz's estimate endpoint prices."""

    width: int
    height: int
    duration_seconds: float
    frame_rate: float
    # The exact total read out of the sample tables, when it was read. Not part
    # of the geometry's identity, so it stays out of equality.
    sample_count: int | None = field(default=None, compare=False)

    @property
    def frame_count(self) -> int:
        """
        Frames Topaz's engine will process - the quantity it bills for.

        The parsed total wins over the derived product because duration * rate
        is not exactly representable: timescale 1000, duration 1025 and 25
        samples multiply out to 24.999999999999996, which would quote 24 frames
        of footage that has 25. Rounded rather than floored for the same reason
        when it has to be derived, and never below 1 - a sub-frame clip still
        costs a frame, and a zero would quote a job that processes nothing.
        """
        if self.sample_count is not None and self.sample_count >= 1:
            return self.sample_count
        return max(round(self.duration_seconds * self.frame_rate), 1)


def _read_uint(data: bytes, start: int, width: int) -> int:
    return int.from_bytes(data[start : start + width], "big")


def _boxes(data: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int, int]]:
    """
    Walk the boxes directly inside `data[start:end]`.

    Yields `(type, box_start, payload_start, box_end)`. The box start is carried
    because a sample entry's fields are measured from the start of its HEADER,
    which is not a fixed distance from its payload: a 64-bit size box puts eight
    extra bytes between them. Stops rather than raising the moment a header
    would run past `end`, so a truncated or non-ISO-BMFF buffer yields nothing.
    """
    offset = start
    while offset + _HEADER_BYTES <= end:
        size = _read_uint(data, offset, 4)
        box_type = data[offset + 4 : offset + _HEADER_BYTES]
        payload_start = offset + _HEADER_BYTES
        if size == _SIZE_IS_64BIT:
            if payload_start + 8 > end:
                return
            size = _read_uint(data, payload_start, 8)
            payload_start += 8
        elif size == _SIZE_TO_END:
            size = end - offset
        box_end = offset + size
        # A box smaller than its own header, or one claiming to extend past its
        # parent, means the buffer is not the structure we think it is.
        if size < _HEADER_BYTES or box_end > end or payload_start > box_end:
            return
        yield box_type, offset, payload_start, box_end
        offset = box_end


def _find_box(data: bytes, start: int, end: int, box_type: bytes) -> tuple[int, int] | None:
    for kind, _box_start, payload_start, payload_end in _boxes(data, start, end):
        if kind == box_type:
            return payload_start, payload_end
    return None


def _find_path(data: bytes, start: int, end: int, path: tuple[bytes, ...]) -> tuple[int, int] | None:
    """Descend a chain of nested box types, e.g. mdia -> minf -> stbl -> stts."""
    bounds = (start, end)
    for box_type in path:
        found = _find_box(data, bounds[0], bounds[1], box_type)
        if found is None:
            return None
        bounds = found
    return bounds


def _timescale_and_duration(data: bytes, start: int, end: int) -> tuple[int, int] | None:
    """
    Read the (timescale, duration) pair shared by the mvhd and mdhd layouts.

    Both are full boxes whose creation/modification times widen from 32 to 64
    bits at version 1, moving the timescale by a fixed amount and widening the
    duration that follows it.

    A zero duration is returned rather than refused: a fragmented file declares
    it there and carries the real one in its movie fragments.
    """
    if start + _FULL_BOX_PREFIX > end:
        return None
    version = data[start]
    timescale_offset = _TIMESCALE_OFFSET.get(version)
    if timescale_offset is None:
        return None
    duration_width = 8 if version == 1 else 4
    base = start + _FULL_BOX_PREFIX + timescale_offset
    if base + 4 + duration_width > end:
        return None
    timescale = _read_uint(data, base, 4)
    duration = _read_uint(data, base + 4, duration_width)
    if timescale <= 0:
        return None
    return timescale, duration


def _track_handler(data: bytes, trak_start: int, trak_end: int) -> bytes | None:
    """The four-character handler type of a track: `vide` for video."""
    hdlr = _find_path(data, trak_start, trak_end, (b"mdia", b"hdlr"))
    if hdlr is None:
        return None
    # A full-box prefix and a 4-byte pre_defined field precede the handler.
    handler_start = hdlr[0] + _FULL_BOX_PREFIX + 4
    if handler_start + 4 > hdlr[1]:
        return None
    return data[handler_start : handler_start + 4]


def _tkhd_dimensions(data: bytes, trak_start: int, trak_end: int) -> tuple[int, int] | None:
    """
    Display dimensions from tkhd, stored as 16.16 fixed point.

    These are the DISPLAY dimensions, so they already account for anamorphic
    pixel aspect ratios; the coded dimensions in the sample entry are preferred
    when available and this is the fallback.
    """
    tkhd = _find_box(data, trak_start, trak_end, b"tkhd")
    if tkhd is None:
        return None
    start, end = tkhd
    if start + _FULL_BOX_PREFIX > end:
        return None
    dimension_offset = _TKHD_DIMENSION_OFFSET.get(data[start])
    if dimension_offset is None:
        return None
    base = start + _FULL_BOX_PREFIX + dimension_offset
    if base + 8 > end:
        return None
    width = int(_read_uint(data, base, 4) / _FIXED_POINT_16_16)
    height = int(_read_uint(data, base + 4, 4) / _FIXED_POINT_16_16)
    if width <= 0 or height <= 0:
        return None
    return width, height


def _sample_entry_dimensions(data: bytes, trak_start: int, trak_end: int) -> tuple[int, int] | None:
    """Coded dimensions from the first visual sample entry in stsd."""
    stsd = _find_path(data, trak_start, trak_end, (b"mdia", b"minf", b"stbl", b"stsd"))
    if stsd is None:
        return None
    # Full-box prefix then a 4-byte entry count; the entries follow as boxes.
    entries_start = stsd[0] + _FULL_BOX_PREFIX + 4
    if entries_start > stsd[1]:
        return None
    for _kind, entry_start, _payload_start, _payload_end in _boxes(data, entries_start, stsd[1]):
        width_at = entry_start + _SAMPLE_ENTRY_WIDTH_OFFSET
        if width_at + 4 > stsd[1]:
            return None
        width = _read_uint(data, width_at, 2)
        height = _read_uint(data, width_at + 2, 2)
        if width > 0 and height > 0:
            return width, height
        return None
    return None


def _sample_count(data: bytes, trak_start: int, trak_end: int) -> int | None:
    """
    Total samples (video frames) from the time-to-sample table.

    stts is run-length encoded as [sample_count, sample_delta] pairs, so the
    frame total is the sum of the counts rather than the entry count.
    """
    stts = _find_path(data, trak_start, trak_end, (b"mdia", b"minf", b"stbl", b"stts"))
    if stts is None:
        return None
    start, end = stts
    table_start = start + _FULL_BOX_PREFIX + 4
    if table_start > end:
        return None
    entry_count = _read_uint(data, start + _FULL_BOX_PREFIX, 4)
    # Each entry is two 32-bit fields; a count that overruns the box means the
    # table is not what we think it is.
    if entry_count <= 0 or table_start + entry_count * 8 > end:
        return None
    total = sum(_read_uint(data, table_start + index * 8, 4) for index in range(entry_count))
    return total if total > 0 else None


def _track_id(data: bytes, trak_start: int, trak_end: int) -> int | None:
    tkhd = _find_box(data, trak_start, trak_end, b"tkhd")
    if tkhd is None:
        return None
    start, end = tkhd
    if start + _FULL_BOX_PREFIX > end:
        return None
    track_id_offset = _TKHD_TRACK_ID_OFFSET.get(data[start])
    if track_id_offset is None:
        return None
    base = start + _FULL_BOX_PREFIX + track_id_offset
    if base + 4 > end:
        return None
    return _read_uint(data, base, 4) or None


def _box_flags(data: bytes, payload_start: int) -> int:
    """The 24-bit flags field a full box carries after its version byte."""
    return _read_uint(data, payload_start + 1, 3)


def _tfhd_defaults(data: bytes, traf_start: int, traf_end: int) -> tuple[int, int] | None:
    """The fragment's (track_ID, default_sample_duration); the duration is 0 when absent."""
    tfhd = _find_box(data, traf_start, traf_end, b"tfhd")
    if tfhd is None:
        return None
    start, end = tfhd
    cursor = start + _FULL_BOX_PREFIX
    if cursor + 4 > end:
        return None
    flags = _box_flags(data, start)
    track_id = _read_uint(data, cursor, 4)
    cursor += 4
    if flags & _TFHD_BASE_DATA_OFFSET:
        cursor += 8
    if flags & _TFHD_SAMPLE_DESCRIPTION_INDEX:
        cursor += 4
    if not flags & _TFHD_DEFAULT_SAMPLE_DURATION:
        return track_id, 0
    if cursor + 4 > end:
        return None
    return track_id, _read_uint(data, cursor, 4)


def _trun_totals(data: bytes, start: int, end: int, default_duration: int) -> tuple[int, int] | None:
    """One run's (samples, duration in media timescale)."""
    cursor = start + _FULL_BOX_PREFIX
    if cursor + 4 > end:
        return None
    flags = _box_flags(data, start)
    sample_count = _read_uint(data, cursor, 4)
    cursor += 4
    if flags & _TRUN_DATA_OFFSET:
        cursor += 4
    if flags & _TRUN_FIRST_SAMPLE_FLAGS:
        cursor += 4
    per_sample = 4 * sum(1 for field_flag in _TRUN_OPTIONAL_SAMPLE_FIELDS if flags & field_flag)
    if cursor + sample_count * per_sample > end:
        return None
    if not flags & _TRUN_SAMPLE_DURATION:
        return sample_count, sample_count * default_duration
    # sample_duration is the first per-sample field when present.
    duration = sum(_read_uint(data, cursor + index * per_sample, 4) for index in range(sample_count))
    return sample_count, duration


def _fragment_timing(data: bytes, track_id: int) -> tuple[int, int] | None:
    """
    A track's (samples, duration in media timescale) summed over the movie
    fragments that follow the moov.

    A fragmented MP4 ships an INITIALIZATION moov: its stts is empty and its
    mdhd duration is commonly 0, because the samples live in the moof/traf/trun
    boxes after it. Without reading those, a perfectly valid supported file
    quotes nothing and the restore records $0.
    """
    samples = 0
    duration = 0
    for kind, _box_start, moof_start, moof_end in _boxes(data, 0, len(data)):
        if kind != b"moof":
            continue
        for traf_kind, _traf_box_start, traf_start, traf_end in _boxes(data, moof_start, moof_end):
            if traf_kind != b"traf":
                continue
            defaults = _tfhd_defaults(data, traf_start, traf_end)
            if defaults is None or defaults[0] != track_id:
                continue
            for run_kind, _run_box_start, run_start, run_end in _boxes(data, traf_start, traf_end):
                if run_kind != b"trun":
                    continue
                run = _trun_totals(data, run_start, run_end, defaults[1])
                if run is None:
                    return None
                samples += run[0]
                duration += run[1]
    if samples <= 0:
        return None
    return samples, duration


def _video_trak(data: bytes, moov_start: int, moov_end: int) -> tuple[int, int] | None:
    for kind, _box_start, payload_start, payload_end in _boxes(data, moov_start, moov_end):
        if kind == b"trak" and _track_handler(data, payload_start, payload_end) == b"vide":
            return payload_start, payload_end
    return None


def parse_video_geometry(data: bytes) -> SourceGeometry | None:
    """
    Read width, height, duration and frame rate from ISO-BMFF source bytes.

    Returns None for anything it cannot read with confidence - a non-ISO-BMFF
    container, a truncated buffer, a file with no video track, or a frame rate
    outside the plausible range. None means "no quote", which records no cost,
    which is exactly the state the NOL-535 guard is designed to catch. A wrong
    number would not be.
    """
    moov = _find_box(data, 0, len(data), b"moov")
    if moov is None:
        return None
    trak = _video_trak(data, moov[0], moov[1])
    if trak is None:
        return None

    dimensions = _sample_entry_dimensions(data, trak[0], trak[1]) or _tkhd_dimensions(data, trak[0], trak[1])
    if dimensions is None:
        return None

    media = _find_path(data, trak[0], trak[1], (b"mdia", b"mdhd"))
    if media is None:
        return None
    media_time = _timescale_and_duration(data, media[0], media[1])
    if media_time is None:
        return None
    timescale, media_duration = media_time

    samples = _sample_count(data, trak[0], trak[1])
    if samples is None or media_duration <= 0:
        # Either half can be missing on a fragmented file, so fill in only the
        # half the initialization moov did not answer.
        track_id = _track_id(data, trak[0], trak[1])
        fragments = _fragment_timing(data, track_id) if track_id is not None else None
        if fragments is None:
            return None
        samples = samples if samples is not None else fragments[0]
        media_duration = media_duration if media_duration > 0 else fragments[1]

    if media_duration <= 0:
        return None
    duration_seconds = media_duration / timescale
    frame_rate = samples / duration_seconds
    if frame_rate < _MIN_PLAUSIBLE_FPS or frame_rate > _MAX_PLAUSIBLE_FPS:
        return None

    return SourceGeometry(
        width=dimensions[0],
        height=dimensions[1],
        duration_seconds=duration_seconds,
        frame_rate=frame_rate,
        sample_count=samples,
    )
