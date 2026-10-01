"""Small, validity-preserving source selection."""

from __future__ import annotations

from typing import Iterable

from .models import EvidenceBundle


def select_sources(
    bundle: EvidenceBundle,
    preferred_ids: Iterable[str],
    *,
    maximum: int = 5,
) -> tuple[str, ...]:
    if not 1 <= maximum <= 5:
        raise ValueError("maximum sources must be between 1 and 5")
    allowed = {item.unit_id for item in bundle.items}
    selected: list[str] = []
    for unit_id in preferred_ids:
        if unit_id in allowed and unit_id not in selected:
            selected.append(unit_id)
        if len(selected) == maximum:
            break
    return tuple(selected)

