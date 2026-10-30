"""Periodic, arc-length parameterized geometry from the packaged racing circuits."""

from importlib.resources import files
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from .track import Track


class Circuit(Track):
    @staticmethod
    def available_tracks():
        return ["Oval", *Track.available_tracks()]

    @classmethod
    def load(cls, name="IMS", spacing=2.0, smoothing_m=6.0):
        if name == "Oval":
            # Analytic ellipse, explicitly synthetic: about 650 m per lap.
            theta = np.linspace(-np.pi / 2, 3 * np.pi / 2, 720, endpoint=False)
            raw = np.column_stack(
                (125 * np.cos(theta), 80 * np.sin(theta), np.full(720, 7.5), np.full(720, 7.5))
            )
        elif name not in Track.available_tracks():
            raise ValueError(f"Unknown packaged circuit: {name}")
        else:
            with files("data").joinpath("racetracks", name + ".csv").open() as stream:
                raw = np.loadtxt(stream, delimiter=",")
        keep = np.r_[True, np.linalg.norm(np.diff(raw[:, :2], axis=0), axis=1) > 1e-4]
        raw = raw[keep]
        xy = np.vstack((raw[:, :2], raw[0, :2]))
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        grid = np.linspace(0, s[-1], int(s[-1] / spacing) + 1)
        points = CubicSpline(s, xy, bc_type="periodic")(grid)[:-1]
        # A six-metre smoothing scale removes survey sampling noise, not corners.
        points = gaussian_filter1d(points, smoothing_m / spacing, axis=0, mode="wrap")
        points = np.vstack((points, points[0]))
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        length = arc[-1]
        base_s = np.linspace(0, length, len(arc))
        curve = CubicSpline(arc, points, bc_type="periodic")
        base_xy = curve(base_s)
        d1, d2 = curve(base_s, 1), curve(base_s, 2)
        heading = np.unwrap(np.arctan2(d1[:, 1], d1[:, 0]))
        curvature = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.linalg.norm(d1, axis=1) ** 3
        extended_s = np.concatenate([base_s[:-1] + lap * length for lap in range(-1, 4)])
        extended_xy = np.tile(base_xy[:-1], (5, 1))
        winding = round((heading[-1] - heading[0]) / (2 * np.pi)) * 2 * np.pi
        extended_heading = np.concatenate([heading[:-1] + lap * winding for lap in range(-1, 4)])
        result = cls(
            extended_s,
            extended_xy,
            extended_heading,
            np.tile(curvature[:-1], 5),
            float(raw[:, 2:4].min()),
            name,
        )
        result.length, result.base_s = float(length), base_s
        result.base_xy, result.base_heading = base_xy, heading
        result.base_curvature = curvature
        result.smoothing_m = smoothing_m
        result.width_right = np.interp(base_s / length * s[-1], s, np.r_[raw[:, 2], raw[0, 2]])
        result.width_left = np.interp(base_s / length * s[-1], s, np.r_[raw[:, 3], raw[0, 3]])
        return result

    def position(self, s, n=0.0):
        wrapped = np.asarray(s) % self.length
        angle = np.interp(wrapped, self.base_s, self.base_heading)
        x = np.interp(wrapped, self.base_s, self.base_xy[:, 0]) - np.sin(angle) * n
        y = np.interp(wrapped, self.base_s, self.base_xy[:, 1]) + np.cos(angle) * n
        return np.stack((x, y), axis=-1)

    def nearby(self, state, opponents):
        values = np.array(opponents, copy=True)
        values[:, 0] += np.round((state[0] - values[:, 0]) / self.length) * self.length
        return values

    def curvature_at(self, s):
        return np.interp(np.asarray(s) % self.length, self.base_s, self.base_curvature)

    def heading_at(self, s):
        return np.interp(np.asarray(s) % self.length, self.base_s, self.base_heading)

    def export_circuit(self):
        s = self.base_s
        return dict(
            name=self.name,
            length=self.length,
            s=s.tolist(),
            xy=self.base_xy.tolist(),
            left=self.position(s, self.width_left).tolist(),
            right=self.position(s, -self.width_right).tolist(),
            curvature=self.base_curvature.tolist(),
            half_width=self.half_width,
            smoothing_m=self.smoothing_m,
            source="Analytic ellipse: semi-axes 125 m and 80 m"
            if self.name == "Oval"
            else "Packaged vehicle_gym racetrack survey",
        )
