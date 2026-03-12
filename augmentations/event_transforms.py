from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RandomFlipLR:
    sensor_size: Tuple
    p: float = 0.5

    def __post_init__(self):
        assert 0 <= self.p <= 1

    def __call__(self, events):
        if np.random.rand() > self.p:
            return events

        events = events.copy()
        events["x"] = self.sensor_size[0] - 1 - events["x"]
        return events


class Rotate:
    def __init__(self, sensor_size, angle, remove_outliers=True):
        self.sensor_size = sensor_size
        self.angle = angle
        self.remove_outliers = remove_outliers
        self.is_random = isinstance(angle, (tuple, list))

        if not self.is_random:
            self.angle_rad = np.deg2rad(angle)
            self.cos_angle = np.cos(self.angle_rad)
            self.sin_angle = np.sin(self.angle_rad)

        self.center_x = sensor_size[0] / 2
        self.center_y = sensor_size[1] / 2

    def __call__(self, events):
        # Work in-place if possible
        x = events["x"].astype(np.float32, copy=False)
        y = events["y"].astype(np.float32, copy=False)

        if self.is_random:
            angle_rad = np.deg2rad(np.random.uniform(self.angle[0], self.angle[1]))
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)
        else:
            cos_angle = self.cos_angle
            sin_angle = self.sin_angle

        # Center, rotate, translate back (in-place)
        x -= self.center_x
        y -= self.center_y

        x_rot = x * cos_angle - y * sin_angle + self.center_x
        y_rot = x * sin_angle + y * cos_angle + self.center_y

        if self.remove_outliers:
            valid_mask = (
                    (x_rot >= 0) & (x_rot < self.sensor_size[0]) &
                    (y_rot >= 0) & (y_rot < self.sensor_size[1])
            )
            events = events[valid_mask]
            x_rot = x_rot[valid_mask]
            y_rot = y_rot[valid_mask]

        events["x"] = x_rot.astype(events["x"].dtype)
        events["y"] = y_rot.astype(events["y"].dtype)

        return events


class ShearX:
    def __init__(self, sensor_size, shear_range):
        self.sensor_size = sensor_size
        self.shear_range = shear_range

    def __call__(self, events):
        shear_factor = np.random.uniform(self.shear_range[0], self.shear_range[1])

        # Compute new x values without intermediate array
        x_new = events['x'] + (shear_factor * events['y']).astype(events['x'].dtype)

        # Filter and update in one step
        valid_mask = (x_new >= 0) & (x_new < self.sensor_size[0])
        events = events[valid_mask]
        events['x'] = x_new[valid_mask]

        return events


class Rolling:
    def __init__(self, sensor_size, max_shift):
        self.sensor_size = sensor_size
        self.max_shift_x = int(max_shift * sensor_size[0])
        self.max_shift_y = int(max_shift * sensor_size[1])

    def __call__(self, events):
        shift_x = np.random.randint(-self.max_shift_x, self.max_shift_x + 1)
        shift_y = np.random.randint(-self.max_shift_y, self.max_shift_y + 1)

        # Direct modulo operation without intermediate variables
        events['x'] = (events['x'] + shift_x) % self.sensor_size[0]
        events['y'] = (events['y'] + shift_y) % self.sensor_size[1]

        return events


class DropEventByArea:
    def __init__(self, sensor_size, area_ratio):
        self.sensor_size = sensor_size
        self.area_ratio = area_ratio

    def __call__(self, events: np.ndarray):
        """Drops events located in a randomly chosen box area."""
        assert "x" and "t" and "y" and "p" in events.dtype.names
        assert (type(self.area_ratio) == float and 0.0 <= self.area_ratio < 1.0) or (
                type(self.area_ratio) is tuple
                and len(self.area_ratio) == 2
                and all(0 <= val < 1.0 for val in self.area_ratio)
        )

        # select ratio
        if isinstance(self.area_ratio, tuple):
            area_ratio = np.random.uniform(self.area_ratio[0], self.area_ratio[1])
        else:
            area_ratio = self.area_ratio

        # select area
        cut_w = int(self.sensor_size[0] * area_ratio)
        cut_h = int(self.sensor_size[1] * area_ratio)
        bbx1 = np.random.randint(0, self.sensor_size[0] - cut_w + 1)
        bby1 = np.random.randint(0, self.sensor_size[1] - cut_h + 1)
        bbx2 = bbx1 + cut_w
        bby2 = bby1 + cut_h

        # filter events (single boolean mask)
        mask = (events["x"] >= bbx1) & (events["x"] < bbx2) & \
               (events["y"] >= bby1) & (events["y"] < bby2)

        return events[~mask]


class RandomCropTime:
    def __init__(self, sensor_size, crop_ratio):
        self.sensor_size = sensor_size
        self.crop_ratio = crop_ratio

    def __call__(self, events: np.ndarray):
        """Crops events in the time dimension by a random ratio."""
        assert "t" in events.dtype.names

        if isinstance(self.crop_ratio, tuple):
            crop_ratio = np.random.uniform(self.crop_ratio[0], self.crop_ratio[1])
        else:
            crop_ratio = self.crop_ratio

        total_time = events["t"][-1] - events["t"][0]
        crop_time = total_time * crop_ratio

        start_time = np.random.uniform(events["t"][0], events["t"][-1] - crop_time)
        end_time = start_time + crop_time

        # Filter events based on time
        valid_mask = (events["t"] >= start_time) & (events["t"] < end_time)
        return events[valid_mask]


class RandomApply:
    def __init__(self, transform, p):
        self.transform = transform
        self.p = p

    def __call__(self, events):
        if np.random.rand() < self.p:
            return self.transform(events)
        return events

