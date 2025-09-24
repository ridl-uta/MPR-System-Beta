"""Load shedding helper for overload handling."""

from __future__ import annotations

from typing import Optional, Union

EventType = Union[None, str, tuple]


class LoadShedder:
    """Dynamically shed load once an overload is detected."""

    def __init__(
        self,
        threshold_w: float,
        sample_period_s: float,
        *,
        margin_w: float = 0.0,
        activation_delay_s: float = 0.0,
        max_reduction_w: Optional[float] = None,
    ) -> None:
        self.threshold_w = float(threshold_w)
        self.margin_w = max(0.0, float(margin_w))
        self.target_w = max(0.0, self.threshold_w - self.margin_w)
        self.sample_period = max(1e-6, float(sample_period_s))
        self.delay_samples = max(0, int(round(activation_delay_s / self.sample_period)))
        self.max_reduction = (
            None if max_reduction_w is None else max(0.0, float(max_reduction_w))
        )

        self._active = False
        self._armed = False
        self._pending_samples = 0

    @property
    def active(self) -> bool:
        return self._active

    def adjust(self, watts: float) -> float:
        watts = float(watts)

        if self._armed and self._pending_samples > 0:
            self._pending_samples -= 1
            if self._pending_samples == 0:
                self._armed = False
                self._active = True
            return watts

        if not self._active:
            return watts

        excess = watts - self.target_w
        if excess <= 0:
            return watts

        reduction = excess
        if self.max_reduction is not None:
            reduction = min(reduction, self.max_reduction)

        return max(0.0, watts - reduction)

    def handle_event(self, event: EventType) -> None:
        label: Optional[str]
        if isinstance(event, tuple):
            label = event[0]
        else:
            label = event

        if label is None:
            return

        if label == 'OVERLOAD_START':
            if self.delay_samples > 0:
                self._armed = True
                self._pending_samples = self.delay_samples
                self._active = False
            else:
                self._active = True
                self._armed = False
        elif label == 'OVERLOAD_HANDLED':
            # remain active while handled state persists
            if not self._armed and self.delay_samples > 0:
                self._active = True
        elif label == 'OVERLOAD_END':
            self._active = False
            self._armed = False
            self._pending_samples = 0
        elif label and label.startswith('RAMP'):
            return
        elif label == 'SPIKE_WARNING':
            return
        else:
            # Unknown terminal event -> disable shedding
            self._active = False
            self._armed = False
            self._pending_samples = 0
