from typing import Optional, Callable, Sequence

import numpy as np
from torch.utils.data import Dataset


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    This is a tonic-friendly version of torch.utils.data.Subset that properly
    handles transform, target_transform, and transforms attributes, which are
    commonly used in tonic datasets.

    Parameters:
        dataset: The full dataset
        indices: Indices in the whole set selected for subset
        transform: Optional transform to override the dataset's transform
        target_transform: Optional target_transform to override the dataset's target_transform
        transforms: Optional transforms to override the dataset's transforms
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.indices = indices

        # Handle transforms - either use provided or inherit from dataset
        self._transform = transform
        self._target_transform = target_transform
        self._transforms = transforms

    @property
    def transform(self):
        """Get transform - returns custom if set, otherwise falls back to dataset's transform."""
        if self._transform is not None:
            return self._transform
        return getattr(self.dataset, 'transform', None)

    @transform.setter
    def transform(self, value):
        """Set transform for this subset."""
        self._transform = value

    @property
    def target_transform(self):
        """Get target_transform - returns custom if set, otherwise falls back to dataset's target_transform."""
        if self._target_transform is not None:
            return self._target_transform
        return getattr(self.dataset, 'target_transform', None)

    @target_transform.setter
    def target_transform(self, value):
        """Set target_transform for this subset."""
        self._target_transform = value

    @property
    def transforms(self):
        """Get transforms - returns custom if set, otherwise falls back to dataset's transforms."""
        if self._transforms is not None:
            return self._transforms
        return getattr(self.dataset, 'transforms', None)

    @transforms.setter
    def transforms(self, value):
        """Set transforms for this subset."""
        self._transforms = value

    def __getitem__(self, idx):
        """Get item at index."""
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        """Get length of subset."""
        return len(self.indices)

    def __getattr__(self, name):
        """
        Forward attribute access to the underlying dataset if not found in Subset.
        This allows the subset to behave like the original dataset for attributes
        like 'targets', 'data', 'sensor_size', etc.
        """
        # Avoid infinite recursion for private attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Special handling for common dataset attributes
        if name in ('targets', 'data', 'classes', 'class_to_idx'):
            attr = getattr(self.dataset, name, None)
            if attr is not None and hasattr(attr, '__getitem__'):
                # If it's indexable, return the subset
                try:
                    return [attr[i] for i in self.indices]
                except (TypeError, KeyError):
                    return attr
            return attr

        # For other attributes, just forward to the dataset
        return getattr(self.dataset, name)

def create_subset(dataset, samples_per_class: float, seed: Optional[int] = None, samples_mode: str = 'percent'):
    """Create a subset with a specific percentage or absolute number of samples per class.

    Args:
        dataset: Dataset to create a subset from.
        samples_per_class: Either a percentage (0.0–1.0) when samples_mode='percent',
                           or an absolute integer count when samples_mode='count'.
        seed: Random seed for reproducibility.
        samples_mode: 'percent' to use fraction of each class, 'count' for absolute number.
    """
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))

    if samples_mode == 'percent':
        if samples_per_class >= 1.0:
            return dataset
        indices = []
        for class_idx in range(num_classes):
            class_indices = np.where(targets == class_idx)[0]
            n_samples = max(1, int(len(class_indices) * samples_per_class))
            selected = rng.choice(class_indices, size=n_samples, replace=False)
            indices.extend(selected.tolist())
    elif samples_mode == 'count':
        n_samples_per_class = max(1, int(samples_per_class))
        indices = []
        for class_idx in range(num_classes):
            class_indices = np.where(targets == class_idx)[0]
            n_samples = min(n_samples_per_class, len(class_indices))
            selected = rng.choice(class_indices, size=n_samples, replace=False)
            indices.extend(selected.tolist())
    else:
        raise ValueError(f"Unknown samples_mode: '{samples_mode}'. Choose 'percent' or 'count'.")

    return Subset(dataset, indices)


