from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class Interval:
    start: int
    end: int
    support: int


def merge_timestamps(timestamps: Iterable[int], max_gap_sec: int) -> List[Interval]:
    """
    Merge sorted timestamps into contiguous intervals.
    If the gap between consecutive timestamps is <= max_gap_sec, they belong to the same interval.
    """
    ts = sorted(set(int(t) for t in timestamps))
    if not ts:
        return []

    intervals: List[Interval] = []
    start = prev = ts[0]
    support = 1

    for t in ts[1:]:
        if t - prev <= max_gap_sec:
            prev = t
            support += 1
        else:
            intervals.append(Interval(start=start, end=prev, support=support))
            start = prev = t
            support = 1

    intervals.append(Interval(start=start, end=prev, support=support))
    return intervals
