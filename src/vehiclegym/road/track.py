"""Lightweight geometry and packaged track resources, independent of plotting."""

from dataclasses import dataclass
from importlib.resources import files
import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class Track:
    s: np.ndarray
    xy: np.ndarray
    heading: np.ndarray
    curvature: np.ndarray
    half_width: float = 7.0
    name: str = "Synthetic circuit section"

    def __post_init__(self):
        if (
            self.s.ndim != 1
            or len(self.s) < 4
            or not np.isfinite(self.s).all()
            or np.any(np.diff(self.s) <= 0)
        ):
            raise ValueError("Track distance must increase and contain at least four samples")
        if self.xy.shape != (len(self.s), 2) or not np.isfinite(self.xy).all():
            raise ValueError("Track coordinates must be finite with shape (samples, 2)")
        if (
            self.heading.shape != self.s.shape
            or self.curvature.shape != self.s.shape
            or not np.isfinite(self.heading).all()
            or not np.isfinite(self.curvature).all()
        ):
            raise ValueError("Heading and curvature must match the distance grid")
        if not np.isfinite(self.half_width) or self.half_width <= 0:
            raise ValueError("Track width must be positive")

    @classmethod
    def synthetic(cls, bend=0.003, length=1800.0):
        s = np.linspace(-100, length, int((length + 100) / 2) + 1)
        heading = bend * 70 * np.sin(s / 70)
        ds = np.diff(s)
        xy = np.column_stack(
            (
                np.r_[0, np.cumsum(ds * np.cos((heading[1:] + heading[:-1]) / 2))],
                np.r_[0, np.cumsum(ds * np.sin((heading[1:] + heading[:-1]) / 2))],
            )
        )
        xy -= [np.interp(0, s, xy[:, 0]), np.interp(0, s, xy[:, 1])]
        return cls(s, xy, heading, bend * np.cos(s / 70))

    @classmethod
    def from_csv(cls, name="Spielberg"):
        if name not in cls.available_tracks():
            raise ValueError(f"Unknown track {name!r}")
        with files("data").joinpath("racetracks", name + ".csv").open() as stream:
            raw = np.loadtxt(stream, delimiter=",")
        xy = raw[:, :2]
        xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-5]]
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        spline = CubicSpline(s, xy)
        d1, d2 = spline(s, 1), spline(s, 2)
        heading = np.unwrap(np.arctan2(d1[:, 1], d1[:, 0]))
        curvature = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.linalg.norm(d1, axis=1) ** 3
        return cls(s, xy, heading, curvature, float(np.min(raw[:, 2:4])), name)

    @staticmethod
    def available_tracks():
        return sorted(
            p.name[:-4]
            for p in files("data").joinpath("racetracks").iterdir()
            if p.name.endswith(".csv")
        )

    def position(self, s, n=0.0):
        angle = np.interp(s, self.s, self.heading)
        x = np.interp(s, self.s, self.xy[:, 0]) - np.sin(angle) * n
        y = np.interp(s, self.s, self.xy[:, 1]) + np.cos(angle) * n
        return np.stack((x, y), axis=-1)

    def export(self, start=0.0, end=600.0, samples=601):
        s = np.linspace(start, end, samples)
        return {
            "s": s.tolist(),
            "xy": self.position(s).tolist(),
            "left": self.position(s, self.half_width).tolist(),
            "right": self.position(s, -self.half_width).tolist(),
            "curvature": np.interp(s, self.s, self.curvature).tolist(),
            "name": self.name,
        }
