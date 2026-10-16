"""Process schedule: torch segments, inputs, torch position and scan path.

A schedule is a list of segments with constant current and constant travel
speed (0 for a dwell). A segment may start at a new position; the torch moves
there with the arc off. Travel is along x: towards +x, unless a segment gives
direction -1 (back-and-forth passes). Inputs are piecewise constant and
right-continuous: the value that starts at a segment start applies from that
time on, the convention of the 2D code. After the last segment the torch is
off and stationary.

Configurations give either the S0 form (change_times, currents, speeds,
t_heat_end, x0) or an explicit list `segments` of
{current, speed, duration, x_start (optional; default: previous end),
direction (optional; +1 or -1, default +1)}.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analytic import Segment


@dataclass
class Seg:
    t0: float
    t1: float
    x0: float
    x1: float
    current: float     # A
    speed: float       # m/s, magnitude; 0 for a dwell
    direction: int = 1  # +1 towards +x, -1 towards -x


class ProcessSchedule:
    def __init__(self, segments, voltage, eta, y0, z_top=0.0):
        self.segments = list(segments)
        self.voltage, self.eta, self.y0, self.z_top = float(voltage), float(eta), float(y0), float(z_top)
        self._t0 = np.array([s.t0 for s in self.segments])
        self._t_end = self.segments[-1].t1
        self._I = np.array([s.current for s in self.segments])
        self._u = np.array([s.speed for s in self.segments])
        self._v = np.array([s.direction * s.speed for s in self.segments])     # signed x velocity
        self._x0 = np.array([s.x0 for s in self.segments])
        travel, d = [], 1                     # dwells keep the direction of the previous pass
        for s in self.segments:
            d = s.direction if s.speed > 0 else d
            travel.append(d)
        self._travel = np.array(travel)

    @property
    def t_heat_end(self) -> float:
        """End of the last segment with the arc on."""
        on = [s.t1 for s in self.segments if s.current > 0]
        return max(on) if on else 0.0

    # -- inputs at time t (scalars or arrays) --------------------------------
    def _idx(self, t):
        t = np.asarray(t, dtype=float)
        i = np.clip(np.searchsorted(self._t0, t, side="right") - 1, 0, len(self.segments) - 1)
        active = (t >= self._t0[0]) & (t < self._t_end - 1e-12)
        return i, active, t

    def current(self, t):
        i, active, _ = self._idx(t)
        return np.where(active, self._I[i], 0.0)

    def speed(self, t):
        i, active, _ = self._idx(t)
        return np.where(active, self._u[i], 0.0)

    def incident_power(self, t):
        return self.voltage * self.current(t)

    def absorbed_power(self, t):
        return self.eta * self.incident_power(t)

    def position(self, t):
        i, active, t = self._idx(t)
        x = self._x0[i] + self._v[i] * (t - self._t0[i])
        return np.where(t >= self._t_end - 1e-12, self.segments[-1].x1,
                        np.where(t < self._t0[0], self._x0[0], x))

    def direction(self, t):
        """Travel direction (+1 or -1) at time t; a dwell keeps the previous pass's direction."""
        i, _, _ = self._idx(t)
        return self._travel[i]

    def absorbed_energy(self, t):
        """Absorbed energy delivered up to time t (J), exact for the schedule."""
        return sum(self.eta * self.voltage * s.current * max(0.0, min(s.t1, t) - s.t0) for s in self.segments)

    # -- exports ---------------------------------------------------------------
    def scan_path_text(self, t_end: float) -> str:
        """AdditiveFOAM scan path: mode x y z power parameter (header ignored).

        A zero-duration spot row places the torch at the start of the path and
        after every jump. Lines use mode 0 (target point and speed, in either
        direction); dwells use mode 1 (duration).
        """
        y, z = self.y0, self.z_top
        rows = ["Mode    X    Y    Z    Power    Param"]
        prev = None
        for s in self.segments:
            x0, x1 = float(s.x0), float(s.x1)
            if prev is None or abs(x0 - prev) > 1e-12:
                rows.append(f"1    {x0!r}    {y!r}    {z!r}    0    0")
            P = float(self.voltage * s.current)
            if s.speed > 0:
                rows.append(f"0    {x1!r}    {y!r}    {z!r}    {P!r}    {float(s.speed)!r}")
            else:
                rows.append(f"1    {x0!r}    {y!r}    {z!r}    {P!r}    {float(s.t1 - s.t0)!r}")
            prev = x1
        if t_end > self._t_end + 1e-9:
            rows.append(f"1    {float(self.segments[-1].x1)!r}    {y!r}    {z!r}    0    {float(t_end - self._t_end)!r}")
        return "\n".join(rows) + "\n"

    def analytic_segments(self) -> list:
        return [Segment(t0=s.t0, t1=s.t1, Q=self.eta * self.voltage * s.current,
                        x0=s.x0, x1=s.x1, y=self.y0)
                for s in self.segments if s.current > 0]

    @classmethod
    def from_config(cls, cfg: dict) -> "ProcessSchedule":
        s = cfg["schedule"]
        segs = []
        if s.get("segments"):
            t, x = 0.0, None
            for spec in s["segments"]:
                x0 = float(spec["x_start"]) if spec.get("x_start") is not None else x
                if x0 is None:
                    raise ValueError("the first schedule segment needs x_start")
                d = int(spec.get("direction", 1))
                if d not in (1, -1):
                    raise ValueError(f"segment direction must be +1 or -1, got {d}")
                dur, u = float(spec["duration"]), float(spec.get("speed", 0.0))
                segs.append(Seg(t, t + dur, x0, x0 + d * u * dur, float(spec["current"]), u, d))
                t, x = t + dur, x0 + d * u * dur
        else:
            ct = [float(v) for v in s["change_times"]] + [float(s["t_heat_end"])]
            x = float(s["x0"])
            for i in range(len(s["change_times"])):
                u = float(s["speeds"][i])
                x1 = x + u * (ct[i + 1] - ct[i])
                segs.append(Seg(ct[i], ct[i + 1], x, x1, float(s["currents"][i]), u))
                x = x1
        return cls(segs, s["voltage"], cfg["source"]["eta"], s["y0"])
