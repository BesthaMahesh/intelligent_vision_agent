import time
from contextlib import contextmanager
from typing import Dict, Any, Optional
from backend.observability.logger import get_logger

log = get_logger("MetricsTracker")


class PerformanceTracker:
    def __init__(self):
        self.timings: Dict[str, float] = {}

    @contextmanager
    def measure(self, stage_name: str):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.timings[stage_name] = round(elapsed_ms, 2)
            log.debug(f"Stage [{stage_name}] completed in {elapsed_ms:.2f} ms")

    def get_timing(self, stage_name: str, default: float = 0.0) -> float:
        return self.timings.get(stage_name, default)

    def get_summary(self) -> Dict[str, float]:
        return dict(self.timings)
