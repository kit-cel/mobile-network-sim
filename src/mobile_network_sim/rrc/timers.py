"""Timers for RRC state machine."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class StepTimer:
    """Simple step-based timer."""

    active: bool = False
    remaining_steps: int = 0
    _step_ms: int | None = field(default=None, repr=False)

    def set_step_ms(self, step_ms: int) -> None:
        """
        Set the step duration in ms for remaining_ms compatibility.

        Parameters
        ----------
        step_ms : int
            Duration of one step in milliseconds. Must be positive.
        """
        if step_ms <= 0:
            raise ValueError("step_ms must be positive")
        self._step_ms = int(step_ms)

    def start(self, duration_steps: int) -> None:
        """
        Start the timer.

        Parameters
        ----------
        duration_steps : int
            Duration of the timer in steps. Must be non-negative. Zero leaves
            the timer inactive; the caller must handle immediate expiry itself.
        """
        if duration_steps < 0:
            raise ValueError("duration_steps must be non-negative")
        if self.active:
            raise RuntimeError("Timer already active")
        self.active = duration_steps > 0
        self.remaining_steps = duration_steps

    def cancel(self) -> None:
        """
        Cancel the timer iif it is active.

        Raises
        ------
        RuntimeError
            If the timer is not active.
        """
        if not self.active:
            raise RuntimeError("Timer not active")
        self.active = False
        self.remaining_steps = 0

    def cancel_if_active(self) -> bool:
        """
        Cancel the timer if it is active.

        Returns
        -------
        bool
            True if the timer was active and is now canceled, False if it was
            not active.
        """
        if not self.active:
            return False
        self.active = False
        self.remaining_steps = 0
        return True

    def clear(self) -> None:
        """Clear the timer, making it inactive and resetting remaining steps to 0."""
        self.active = False
        self.remaining_steps = 0

    def tick(self) -> bool:
        """
        Advance the timer by one step.

        Returns
        -------
        bool
            True if the timer has expired, False otherwise, or if the timer was not active.
        """
        if not self.active:
            return False

        self.remaining_steps -= 1
        if self.remaining_steps <= 0:
            self.active = False
            self.remaining_steps = 0
            return True
        return False

    @property
    def remaining_ms(self) -> int | None:
        """
        Get the remaining time in milliseconds.

        Returns
        -------
        int | None
            Remaining time in ms if the timer is active and step_ms is set, None otherwise.
        """
        if not self.active:
            return None
        if self._step_ms is None:
            return None
        return self.remaining_steps * self._step_ms


@dataclass(slots=True)
class StepTimers:
    """Collection of step-based timers for one UE."""

    t310: StepTimer
    ttt: StepTimer
    prep: StepTimer
    exec: StepTimer
    rlf_recovery: StepTimer

    @classmethod
    def create(cls, *, step_ms: int) -> "StepTimers":
        """
        Create a new StepTimers instance with all timers initialized and step_ms set.

        Parameters
        ----------
        step_ms : int
            Duration of one step in milliseconds. Must be positive.

        Returns
        -------
        StepTimers
            A new instance of StepTimers with all timers initialized.
        """
        timers = cls(
            t310=StepTimer(),
            ttt=StepTimer(),
            prep=StepTimer(),
            exec=StepTimer(),
            rlf_recovery=StepTimer(),
        )
        for timer in (
            timers.t310,
            timers.ttt,
            timers.prep,
            timers.exec,
            timers.rlf_recovery,
        ):
            timer.set_step_ms(step_ms)
        return timers
