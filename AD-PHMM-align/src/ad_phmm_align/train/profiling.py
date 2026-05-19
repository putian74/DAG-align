"""CPU and memory profiling contracts for training and data preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
import resource
import time
from typing import Mapping, Optional


@dataclass(frozen=True)
class ProfilingConfig:
    """Controls lightweight profiling captured during training."""

    enabled: bool = True
    record_cpu_time: bool = True
    record_peak_memory: bool = True
    record_device_memory: bool = False
    sample_every_steps: int = 1


@dataclass(frozen=True)
class ProfilingResult:
    """Profiling measurements attached to a data-prep or training step."""

    wall_seconds: float
    cpu_seconds: Optional[float] = None
    peak_rss_bytes: Optional[int] = None
    device_peak_bytes: Optional[int] = None
    metadata: Mapping[str, object] = field(default_factory=dict)


class ProfilingTimer:
    """Context manager for lightweight wall/CPU/RSS measurements."""

    def __init__(self, config: ProfilingConfig) -> None:
        self.config = config
        self._start_wall = 0.0
        self._start_cpu = 0.0
        self.result: Optional[ProfilingResult] = None

    def __enter__(self) -> "ProfilingTimer":
        self._start_wall = time.perf_counter()
        if self.config.record_cpu_time:
            self._start_cpu = time.process_time()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        wall_seconds = time.perf_counter() - self._start_wall
        cpu_seconds = None
        peak_rss_bytes = None
        if self.config.record_cpu_time:
            cpu_seconds = time.process_time() - self._start_cpu
        if self.config.record_peak_memory:
            peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.result = ProfilingResult(
            wall_seconds=wall_seconds,
            cpu_seconds=cpu_seconds,
            peak_rss_bytes=peak_rss_bytes,
        )

