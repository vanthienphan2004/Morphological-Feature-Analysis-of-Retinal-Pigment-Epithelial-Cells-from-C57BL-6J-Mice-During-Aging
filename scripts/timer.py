"""Small execution timer utilities.

Provides:
- start_timer() -> float
- stop_timer(start) -> (end, duration)
- Timer context manager (prints duration by default)
- timeit decorator

Use from other scripts:
    from scripts.timer import Timer
    with Timer('full run'):
        do_work()

Or the simple helpers:
    start = start_timer()
    ...
    end, duration = stop_timer(start)
"""
from __future__ import annotations

import time
import functools
from typing import Callable, Optional, Tuple


def start_timer() -> float:
    """Return a timestamp (float) representing start time."""
    return time.time()


def stop_timer(start: float) -> Tuple[float, float]:
    """Stop timer started with start_timer().

    Returns (end_time, duration_seconds).
    """
    end = time.time()
    return end, end - start


class Timer:
    """Context manager for measuring elapsed wall time.

    Example:
        with Timer('processing'):
            do_work()
    """

    def __init__(self, name: Optional[str] = None, logger: Callable[[str], None] = print):
        self.name = name
        self.logger = logger
        self.start: Optional[float] = None
        self.end: Optional[float] = None

    def __enter__(self) -> 'Timer':
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end = time.time()
        duration = self.end - (self.start or self.end)
        label = f" [{self.name}]" if self.name else ""
        self.logger(f"Timer{label}: {duration:.4f} s")

    @property
    def elapsed(self) -> float:
        """Return elapsed seconds so far (if running) or total duration after exit."""
        if self.start is None:
            return 0.0
        if self.end is None:
            return time.time() - self.start
        return self.end - self.start


def timeit(func: Callable) -> Callable:
    """Decorator to measure a function's execution time and print it.

    Usage:
        @timeit
        def work(...):
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end = time.time()
            print(f"{func.__name__} took {end - start:.4f} s")

    return wrapper


if __name__ == '__main__':
    # simple CLI demo
    print('Timer demo: sleeping 0.5s')
    s = start_timer()
    time.sleep(0.5)
    e, d = stop_timer(s)
    print(f'start={s:.6f} end={e:.6f} duration={d:.4f} s')

    print('Context manager demo:')
    with Timer('demo'):
        time.sleep(0.25)
