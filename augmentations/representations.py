from dataclasses import dataclass

import numpy as np
import torch
from numba import jit


@jit(nopython=True)
def _accumulate_events_jit(frames, p, y, x, start_idx, end_idx, frame_idx):
    """JIT-compiled event accumulation for a single time bin."""
    for i in range(start_idx, end_idx):
        frames[frame_idx, p[i], y[i], x[i]] += 1


@jit(nopython=True)
def _accumulate_voxel_grid_merged_jit(voxel_grid, xs, ys, pols, tis, vals_left, vals_right, n_time_bins):
    """JIT-compiled voxel grid accumulation with merged polarities."""
    for i in range(len(xs)):
        pol_weight = 1.0 if pols[i] == 1 else -1.0
        ti = tis[i]

        # Left bin
        if ti < n_time_bins:
            voxel_grid[ti, 0, ys[i], xs[i]] += pol_weight * vals_left[i]

        # Right bin
        if ti + 1 < n_time_bins:
            voxel_grid[ti + 1, 0, ys[i], xs[i]] += pol_weight * vals_right[i]


@jit(nopython=True)
def _accumulate_voxel_grid_separate_jit(voxel_grid, xs, ys, pols, tis, vals_left, vals_right, n_time_bins):
    """JIT-compiled voxel grid accumulation with separate polarities."""
    for i in range(len(xs)):
        pol = pols[i]
        ti = tis[i]

        # Left bin
        if ti < n_time_bins:
            voxel_grid[ti, pol, ys[i], xs[i]] += vals_left[i]

        # Right bin
        if ti + 1 < n_time_bins:
            voxel_grid[ti + 1, pol, ys[i], xs[i]] += vals_right[i]


@dataclass
class ToFrame:
    """
    Accumulate events to frames with fixed number of time bins.

    Parameters:
        sensor_size (tuple): size of the sensor [W, H, P] where P=2 for polarities
        n_time_bins (int): number of time bins to slice events
    """
    sensor_size: tuple  # (W, H, P)
    n_time_bins: int

    def __post_init__(self):
        assert len(self.sensor_size) == 3, "sensor_size must be (W, H, P)"
        assert self.sensor_size[2] == 2, "Polarity dimension must be 2"
        # Cache frame shape
        self.frame_shape = self.sensor_size[::-1]  # (P, H, W)
        self.P, self.H, self.W = self.frame_shape

    def get_slice_metadata(self, events: np.ndarray) -> np.ndarray:
        """Get time bin boundaries for slicing events.

        Returns:
            np.ndarray of shape (n_time_bins, 2) with [start_idx, end_idx] per bin
        """
        times = events["t"]

        # Single linspace call for bin edges
        edges = np.linspace(times[0], times[-1], self.n_time_bins + 1)

        # Single searchsorted call for all edges
        indices = np.searchsorted(times, edges, side='left')

        # Stack into (n_time_bins, 2) array
        return np.column_stack([indices[:-1], indices[1:]])

    def __call__(self, events: np.ndarray) -> np.ndarray:
        """
        Convert events to frames.

        Parameters:
            events: structured array with fields 'x', 'y', 't', 'p'

        Returns:
            frames: array of shape (n_time_bins, P, H, W)
        """
        assert "x" in events.dtype.names and "y" in events.dtype.names
        assert "t" in events.dtype.names and "p" in events.dtype.names

        # Handle empty events
        if len(events) == 0:
            return np.zeros((self.n_time_bins, *self.frame_shape), dtype=np.int16)

        # Handle single polarity case
        p_values = events["p"].astype(np.int32)
        if self.sensor_size[2] == 1:
            if np.any(p_values != p_values[0]):
                raise ValueError(
                    "Single polarity sensor, but events contain both polarities."
                )
            p_values = np.zeros(len(events), dtype=np.int32)

        # Pre-extract coordinates (avoid repeated structured array access)
        x = events["x"].astype(np.int32)
        y = events["y"].astype(np.int32)

        # Get slice metadata (now returns ndarray)
        metadata = self.get_slice_metadata(events)

        # Allocate output frames
        frames = np.zeros((len(metadata), *self.frame_shape), dtype=np.int16)

        # Accumulate events - iterate over rows
        for i in range(len(metadata)):
            start, end = metadata[i]
            if start < end:
                _accumulate_events_jit(
                    frames, p_values, y, x, start, end, i
                )

        return frames


@dataclass
class ToVoxelGrid:
    """
    Build a voxel grid with bilinear interpolation in the time domain.

    Parameters:
        sensor_size (tuple): size of the sensor [W, H, P] where P=2 for polarities
        n_time_bins (int): number of time bins
        merge_polarities (bool): if True, merge polarities into single channel (T, 1, H, W),
                                 if False, keep separate channels (T, 2, H, W)
    """
    sensor_size: tuple  # (W, H, P)
    n_time_bins: int
    merge_polarities: bool = False

    def __post_init__(self):
        assert len(self.sensor_size) == 3, "sensor_size must be (W, H, P)"
        assert self.sensor_size[2] == 2, "Polarity dimension must be 2"
        self.W, self.H, self.P = self.sensor_size

    def __call__(self, events: np.ndarray) -> np.ndarray:
        """
        Convert events to voxel grid.

        Parameters:
            events: structured array with fields 'x', 'y', 't', 'p'

        Returns:
            voxel_grid: array of shape (n_time_bins, C, H, W) where C=1 or 2
        """
        assert "x" in events.dtype.names and "y" in events.dtype.names
        assert "t" in events.dtype.names and "p" in events.dtype.names

        # Handle empty events
        if len(events) == 0:
            n_channels = 1 if self.merge_polarities else 2
            return np.zeros((self.n_time_bins, n_channels, self.H, self.W), dtype=np.float32)

        # Pre-extract and convert coordinates
        xs = events["x"].astype(np.int32)
        ys = events["y"].astype(np.int32)
        pols = events["p"].astype(np.int32)

        # Normalize timestamps to [0, n_time_bins]
        ts = events["t"].astype(np.float64)
        ts_norm = self.n_time_bins * (ts - ts[0]) / (ts[-1] - ts[0])

        # Compute temporal indices and interpolation weights
        tis = ts_norm.astype(np.int32)
        dts = ts_norm - tis
        vals_left = (1.0 - dts).astype(np.float32)
        vals_right = dts.astype(np.float32)

        # Allocate output
        n_channels = 1 if self.merge_polarities else 2
        voxel_grid = np.zeros((self.n_time_bins, n_channels, self.H, self.W), dtype=np.float32)

        # Use JIT-compiled accumulation
        if self.merge_polarities:
            _accumulate_voxel_grid_merged_jit(
                voxel_grid, xs, ys, pols, tis, vals_left, vals_right, self.n_time_bins
            )
        else:
            _accumulate_voxel_grid_separate_jit(
                voxel_grid, xs, ys, pols, tis, vals_left, vals_right, self.n_time_bins
            )

        return voxel_grid


class ToTensor:
    """
    Transform that converts a numpy array to a PyTorch tensor.

    The resulting tensor will have float32 data type.
    """
    def __init__(self, normalize: bool = True):
        """
        Initialize the ToTensor transform.

        Args:
            normalize (bool): If True, normalize the tensor values to the range [0, 1].
        """
        self.normalize = normalize

    def __call__(self, x):
        """
        Convert numpy array to normalized PyTorch tensor.

        Args:
            x (numpy.ndarray): Input numpy array to convert.

        Returns:
            torch.Tensor: Float tensor converted from the input array, normalized to [0, 1].
        """
        tensor = torch.from_numpy(x).float()

        if not self.normalize:
            return tensor

        min_val = tensor.min()
        max_val = tensor.max()

        # Avoid division by zero if all values are the same
        if max_val - min_val > 0:
            tensor = (tensor - min_val) / (max_val - min_val)

        return tensor
