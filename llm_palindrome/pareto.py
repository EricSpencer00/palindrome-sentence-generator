"""Small utilities for selecting candidates without collapsing objectives."""
from __future__ import annotations

from collections.abc import Iterable, Mapping


def pareto_front(rows: Iterable[Mapping], maximize: tuple[str, ...]) -> list[Mapping]:
    """Return rows not dominated on every requested higher-is-better field."""
    items = list(rows)
    front = []
    for row in items:
        dominated = any(
            all(other[field] >= row[field] for field in maximize)
            and any(other[field] > row[field] for field in maximize)
            for other in items if other is not row
        )
        if not dominated:
            front.append(row)
    return front
