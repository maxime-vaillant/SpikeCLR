import torch


class RandomPolarityAverage:
    """Randomly average the two polarity channels."""

    def __init__(self, p=0.5):
        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            avg = img.mean(dim=1, keepdim=True)
            return avg.repeat(1, 2, 1, 1)
        return img


class PolarityJitter:
    def __init__(self, contrast=0., brightness=0.):
        self.contrast = self._check_input(contrast, 'contrast')
        self.brightness = self._check_input(brightness, 'brightness')

    @staticmethod
    def _check_input(value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def adjust_contrast(img, contrast_factor):
        """Adjust contrast by scaling around the mean intensity."""
        mean_intensity = img.mean(dim=1, keepdim=True)
        return (img - mean_intensity) * contrast_factor + mean_intensity

    @staticmethod
    def adjust_brightness(img, brightness_factor):
        """Adjust brightness by scaling the intensity."""
        return img * brightness_factor

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"contrast={self.contrast}, "
                f"brightness={self.brightness})")

    def __call__(self, img):
        transforms = []

        if self.contrast is not None:
            contrast_factor = torch.empty(1).uniform_(self.contrast[0], self.contrast[1]).item()
            transforms.append(lambda x: self.adjust_contrast(x, contrast_factor))

        if self.brightness is not None:
            brightness_factor = torch.empty(1).uniform_(self.brightness[0], self.brightness[1]).item()
            transforms.append(lambda x: self.adjust_brightness(x, brightness_factor))

        # Apply transforms in random order
        fn_idx = torch.randperm(len(transforms))
        for idx in fn_idx:
            img = transforms[idx](img)

        return img
