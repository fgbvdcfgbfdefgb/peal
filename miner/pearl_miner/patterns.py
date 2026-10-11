"""
Periodic index patterns (port of ``zk_pow::api::proof::PeriodicPattern``).

A pattern is a 3-dimensional arithmetic progression
``a*stride0 + b*stride1 + c*stride2`` with ``a < len0`` etc., serialised to
exactly 6 bytes: for each dimension, ``(factor-1, length-1)`` where
``factor = stride / min_stride`` and ``min_stride`` is the product of all
previous ``stride * length``.
"""

from __future__ import annotations

NUM_DIMS = 3


class PeriodicPattern:
    __slots__ = ("shape",)

    def __init__(self, shape):
        assert len(shape) == NUM_DIMS
        self.shape = [(int(s), int(l)) for s, l in shape]

    # -- serialisation ------------------------------------------------------

    def to_bytes(self) -> bytes:
        out = bytearray(2 * NUM_DIMS)
        min_stride = 1
        for i, (stride, length) in enumerate(self.shape):
            factor = stride // min_stride
            out[2 * i] = (factor - 1) & 0xFF
            out[2 * i + 1] = (length - 1) & 0xFF
            min_stride = stride * length
        return bytes(out)

    @classmethod
    def from_bytes(cls, data: bytes) -> "PeriodicPattern":
        if len(data) != 2 * NUM_DIMS:
            raise ValueError(f"expected {2 * NUM_DIMS} bytes, got {len(data)}")
        shape = []
        min_stride = 1
        is_done = False
        for i in range(NUM_DIMS):
            factor = 1 + data[2 * i]
            length = 1 + data[2 * i + 1]
            if length == 1 or is_done:
                if not (factor == 1 and length == 1):
                    raise ValueError("non-canonical pattern representation")
                is_done = True
            elif factor <= 1 and min_stride != 1:
                raise ValueError("a single stride must not be broken")
            if min_stride > (1 << 24) // (factor * length):
                raise ValueError("pattern period must be <= 2^24")
            stride = factor * min_stride
            shape.append((stride, length))
            min_stride = stride * length
        result = cls(shape)
        if result.to_bytes() != bytes(data):
            raise ValueError("pattern does not round-trip")
        return result

    # -- semantics ----------------------------------------------------------

    def to_list(self) -> list[int]:
        res = [0]
        for stride, length in self.shape:
            new_res = []
            for i in range(length):
                for r in res:
                    new_res.append(r + i * stride)
            res = new_res
        return res

    def max(self) -> int:
        return max(self.to_list())

    def period(self) -> int:
        stride, length = self.shape[-1]
        return stride * length

    def size(self) -> int:
        out = 1
        for _, length in self.shape:
            out *= length
        return out

    def offset_is_valid(self, offset: int) -> bool:
        for stride, length in reversed(self.shape):
            offset %= stride * length
            if offset >= stride:
                return False
        return True

    def indices_with_offset(self, offset: int) -> list[int]:
        return [i + offset for i in self.to_list()]

    @classmethod
    def from_list(cls, pattern) -> "PeriodicPattern":
        p = [int(x) for x in pattern]
        if not p:
            raise ValueError("pattern cannot be empty")
        if any(a >= b for a, b in zip(p, p[1:])):
            raise ValueError("pattern must be sorted with no duplicates")
        if p[0] != 0:
            raise ValueError("pattern must start at 0")

        shape_vec = []
        while len(p) > 1:
            found = False
            for period in range(1, len(p)):
                if len(p) % period == 0:
                    s = p[period]
                    if all(p[i] + s == p[i + period] for i in range(len(p) - period)):
                        shape_vec.append((s, len(p) // period))
                        p = p[:period]
                        found = True
                        break
            if not found:
                raise ValueError("pattern is not periodic")
        shape_vec.reverse()
        period = shape_vec[-1][0] * shape_vec[-1][1] if shape_vec else 1
        while len(shape_vec) < NUM_DIMS:
            shape_vec.append((period, 1))
        result = cls(shape_vec)
        if result.to_list() != sorted(set(int(x) for x in pattern)):
            raise ValueError("constructed pattern is not valid")
        return result

    # -- dunder -------------------------------------------------------------

    def __eq__(self, other):
        return isinstance(other, PeriodicPattern) and self.shape == other.shape

    def __repr__(self):
        return f"PeriodicPattern({self.to_list()})"


def threads_partition(pattern: PeriodicPattern, total_dimension: int):
    """
    Partition ``range(total_dimension)`` into the pattern's index groups,
    one group per valid offset (port of ``threads_partition``).
    """
    period = pattern.period()
    if total_dimension % period != 0:
        raise ValueError(
            f"total_dimension ({total_dimension}) must be divisible by pattern "
            f"period ({period})")
    base = pattern.to_list()
    return [
        [offset + d for d in base]
        for offset in range(total_dimension)
        if pattern.offset_is_valid(offset)
    ]
