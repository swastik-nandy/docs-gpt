from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator


# --------------- CONTEXT --------------------------------

_ACTIVE_TIMING_COLLECTOR: ContextVar["TimingCollector | None"] = ContextVar(
    "markdown_parse_timing_collector",
    default=None,
)


# --------------- SCHEMA --------------------------------

@dataclass
class TimingEvent:
    name: str
    duration_ms: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingCollector:
    events: list[TimingEvent] = field(default_factory=list)

    def add(
        self,
        name: str,
        duration_ms: float,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            TimingEvent(
                name=name,
                duration_ms=round(duration_ms, 3),
                meta=meta or {},
            )
        )

    def totals_by_name(self) -> dict[str, float]:
        totals: dict[str, float] = {}

        for event in self.events:
            totals[event.name] = round(
                totals.get(event.name, 0.0) + event.duration_ms,
                3,
            )

        return totals

    def counts_by_name(self) -> dict[str, int]:
        counts: dict[str, int] = {}

        for event in self.events:
            counts[event.name] = counts.get(event.name, 0) + 1

        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [
                {
                    "name": event.name,
                    "duration_ms": event.duration_ms,
                    "meta": event.meta,
                }
                for event in self.events
            ],
            "totals_ms": self.totals_by_name(),
            "counts": self.counts_by_name(),
        }


# --------------- ACTIVE COLLECTOR --------------------------------

def get_active_timing_collector() -> TimingCollector | None:
    return _ACTIVE_TIMING_COLLECTOR.get()


@contextmanager
def use_timing_collector(
    collector: TimingCollector,
) -> Iterator[TimingCollector]:
    token = _ACTIVE_TIMING_COLLECTOR.set(collector)

    try:
        yield collector
    finally:
        _ACTIVE_TIMING_COLLECTOR.reset(token)


# --------------- TIMER --------------------------------

@contextmanager
def timed(
    name: str,
    **meta: Any,
) -> Iterator[None]:
    collector = get_active_timing_collector()
    start = time.perf_counter()

    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0

        if collector is not None:
            collector.add(
                name=name,
                duration_ms=duration_ms,
                meta={key: value for key, value in meta.items() if value is not None},
            )


def time_call(
    name: str,
    fn: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    with timed(name):
        return fn(*args, **kwargs)
