from __future__ import annotations

import sys
import time
from typing import Any, Callable, Optional, TextIO


class ProgressReporter:
    """Single-parent progress output that never contaminates stdout JSON.

    Interactive terminals use tqdm when it is available. Redirected/non-TTY
    execution emits durable periodic text lines instead, so progress remains
    useful in scheduler logs without ANSI control sequences.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        interval_seconds: float = 5.0,
        stream: Optional[TextIO] = None,
        clock: Callable[[], float] = time.monotonic,
        force_plain: bool = False,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("progress interval must be positive")
        self.enabled = enabled
        self.interval_seconds = float(interval_seconds)
        self.stream = stream if stream is not None else sys.stderr
        self.clock = clock
        self.force_plain = force_plain
        self._stage: Optional[str] = None
        self._label: Optional[str] = None
        self._total = 0
        self._completed = 0
        self._initial = 0
        self._started = 0.0
        self._last_emitted = 0.0
        self._last_emitted_completed: Optional[int] = None
        self._bar: Any = None

    def _interactive(self) -> bool:
        if not self.enabled or self.force_plain:
            return False
        try:
            return bool(self.stream.isatty())
        except (AttributeError, OSError):
            return False

    def start(
        self,
        stage: str,
        label: str,
        total: int,
        *,
        initial: int = 0,
    ) -> None:
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError("progress total must be a non-negative integer")
        if (
            isinstance(initial, bool)
            or not isinstance(initial, int)
            or initial < 0
            or initial > total
        ):
            raise ValueError("progress initial value must be between zero and total")
        self.finish()
        self._stage = stage
        self._label = label
        self._total = total
        self._completed = initial
        self._initial = initial
        self._started = self.clock()
        self._last_emitted = self._started
        self._last_emitted_completed = None
        self._bar = None
        if not self.enabled:
            return
        if self._interactive():
            try:
                from tqdm.auto import tqdm

                # Avoid tqdm's background monitor thread. Progress updates are
                # parent-driven and occur only after a measured SQL call ends.
                tqdm.monitor_interval = 0
                self._bar = tqdm(
                    total=total,
                    initial=initial,
                    desc=label,
                    unit="example",
                    dynamic_ncols=True,
                    file=self.stream,
                    leave=True,
                )
                return
            except Exception:
                self._bar = None
        self._emit_plain(force=True)

    def update(self, completed: int) -> None:
        if self._stage is None:
            return
        if (
            isinstance(completed, bool)
            or not isinstance(completed, int)
            or completed < self._completed
            or completed > self._total
        ):
            raise ValueError("progress updates must be monotonic and within total")
        previous = self._completed
        self._completed = completed
        if not self.enabled:
            return
        if self._bar is not None:
            delta = completed - previous
            if delta:
                try:
                    self._bar.update(delta)
                except Exception:
                    self._bar = None
                    self._emit_plain(force=True)
            return
        now = self.clock()
        should_emit = (
            completed == self._total
            or now - self._last_emitted >= self.interval_seconds
        )
        if should_emit and completed != self._last_emitted_completed:
            self._emit_plain(now=now)

    def increment(self, amount: int = 1) -> None:
        self.update(self._completed + amount)

    def message(self, message: str) -> None:
        if not self.enabled:
            return
        rendered = "[progress] %s" % message
        if self._bar is not None:
            try:
                self._bar.write(rendered, file=self.stream)
                return
            except Exception:
                pass
        try:
            print(rendered, file=self.stream, flush=True)
        except OSError:
            return

    def finish(self) -> None:
        if self._stage is None:
            return
        if self.enabled:
            if self._bar is not None:
                try:
                    self._bar.close()
                except Exception:
                    pass
            elif self._completed != self._last_emitted_completed:
                self._emit_plain(force=True)
        self._bar = None
        self._stage = None
        self._label = None

    def close(self) -> None:
        self.finish()

    def _emit_plain(
        self, *, now: Optional[float] = None, force: bool = False
    ) -> None:
        if not self.enabled or self._stage is None:
            return
        current = self.clock() if now is None else now
        elapsed = max(0.0, current - self._started)
        completed_since_start = self._completed - self._initial
        rate = completed_since_start / elapsed if elapsed > 0 else 0.0
        remaining = self._total - self._completed
        eta = remaining / rate if rate > 0 else None
        percent = (100.0 * self._completed / self._total) if self._total else 100.0
        eta_text = _duration(eta) if eta is not None else "--:--"
        line = (
            "[progress] %s: %d/%d (%.1f%%) elapsed=%s rate=%.2f/s eta=%s"
            % (
                self._label,
                self._completed,
                self._total,
                percent,
                _duration(elapsed),
                rate,
                eta_text,
            )
        )
        try:
            print(line, file=self.stream, flush=True)
        except OSError:
            return
        self._last_emitted = current
        self._last_emitted_completed = self._completed


def _duration(seconds: float) -> str:
    rounded = max(0, int(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return "%d:%02d:%02d" % (hours, minutes, secs)
    return "%02d:%02d" % (minutes, secs)
