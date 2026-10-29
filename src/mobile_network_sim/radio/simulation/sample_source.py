"""Bounded, ordered radio samples with a single owner of mutable radio state."""

from collections import deque
from dataclasses import dataclass, fields
from queue import Empty, Full, Queue
from threading import BoundedSemaphore, Condition, Event, Thread
from time import perf_counter

import numpy as np

from .simulation_env import RadioEnvironment, RadioStepResult


@dataclass(frozen=True)
class RadioSample:
    episode: int
    index: int
    result: RadioStepResult


def freeze_result(result: RadioStepResult, *, owned: bool = False) -> RadioStepResult:
    """Detach mutable inputs and publish read-only sample arrays.

    ``owned`` is internal: the built-in radio returns fresh geometry/LSF
    arrays and transfers their ownership. Only mobility arrays alias live
    state. Unknown producers and public callers keep defensive full copies.
    """
    values = {}
    for name in _RESULT_FIELDS:
        value = getattr(result, name)
        if not owned or name in {"ue_pos_xy", "ue_angles_rad"}:
            value = np.array(value, copy=True)
        value.flags.writeable = False
        values[name] = value
    return RadioStepResult(**values)


_RESULT_FIELDS = tuple(field.name for field in fields(RadioStepResult))


class RadioSampleSource:
    """Prefetch pure radio steps; RRC and actions stay on the consumer thread.

    Reset and close are consumer operations. They join the producer before
    touching radio state. No producer survives an episode or profile reset.
    Block mode preserves per-step random draws; it deliberately does not batch
    stochastic channel calls across the time axis.
    """

    def __init__(
        self, radio: RadioEnvironment, *, mode: str, capacity: int, horizon: int
    ):
        if mode not in {"block", "thread"} or capacity < 1 or horizon < 1:
            raise ValueError("Invalid radio source mode, capacity or horizon")
        self.radio, self.mode, self.capacity, self.horizon = (
            radio,
            mode,
            capacity,
            horizon,
        )
        self.episode = 0
        self.consumed = 0
        self.produced = 0
        self.wait_seconds = 0.0
        self._stop = Event()
        self._thread: Thread | None = None
        self._queue: Queue = Queue(maxsize=capacity)
        self._block: deque[RadioSample] = deque()
        self._ready = False
        self._slots = BoundedSemaphore(capacity)
        self._pause_condition = Condition()
        self._paused = False
        self._in_sample = False

    def reset(self, seed, **radio_reset_options) -> None:
        self.close()
        self.radio.reset(seed, **radio_reset_options)
        self.episode += 1
        self.consumed = self.produced = 0
        self.wait_seconds = 0.0
        self._queue = Queue(maxsize=self.capacity)
        self._block.clear()
        self._stop.clear()
        self._slots = BoundedSemaphore(self.capacity)
        self._ready = True
        # Start lazily: no background work before the consumer's first step.

    def _sample(self) -> RadioSample:
        sample = RadioSample(
            self.episode,
            self.produced,
            freeze_result(
                self.radio.step(), owned=type(self.radio) is RadioEnvironment
            ),
        )
        self.produced += 1
        return sample

    def _publish(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return True
            except Full:
                pass
        return False

    def _produce(self) -> None:
        try:
            while not self._stop.is_set() and self.produced < self.horizon:
                if not self._slots.acquire(timeout=0.05):
                    continue
                if self._stop.is_set():
                    break
                with self._pause_condition:
                    while self._paused and not self._stop.is_set():
                        self._pause_condition.wait()
                    if self._stop.is_set():
                        break
                    self._in_sample = True
                try:
                    if not self._publish(self._sample()):
                        break
                finally:
                    with self._pause_condition:
                        self._in_sample = False
                        self._pause_condition.notify_all()
        except Exception as error:  # noqa: BLE001 - transport producer errors to the caller
            self._publish(error)

    def step(self) -> RadioStepResult:
        if not self._ready:
            raise RuntimeError("Radio source must be reset before stepping")
        if self.consumed >= self.horizon:
            raise RuntimeError("Radio source episode exhausted")
        if self.mode == "block":
            if not self._block:
                self._block.extend(
                    self._sample()
                    for _ in range(min(self.capacity, self.horizon - self.produced))
                )
            sample = self._block.popleft()
        else:
            if self._thread is None:
                self._thread = Thread(
                    target=self._produce, name="radio-producer", daemon=True
                )
                self._thread.start()
            start = perf_counter()
            while True:
                try:
                    sample = self._queue.get(timeout=0.05)
                    break
                except Empty:
                    if not self._thread.is_alive():
                        raise RuntimeError(
                            "Radio producer stopped without a sample"
                        ) from None
            self.wait_seconds += perf_counter() - start
            self._slots.release()
            if isinstance(sample, Exception):
                self.close()
                raise sample
        if sample.episode != self.episode or sample.index != self.consumed:
            raise RuntimeError("Radio sample episode/index mismatch")
        self.consumed += 1
        return sample.result

    def pause(self) -> None:
        """Finish the in-flight sample, then park without discarding state/RNGs."""
        with self._pause_condition:
            self._paused = True
            while self._in_sample:
                self._pause_condition.wait()

    def resume(self) -> None:
        """Continue the same episode and ordered buffer after a pause."""
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()

    def close(self) -> None:
        self._stop.set()
        self.resume()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._block.clear()
        # Release all potentially large buffered arrays, including on early reset.
        self._queue = Queue(maxsize=self.capacity)
        self._ready = False
