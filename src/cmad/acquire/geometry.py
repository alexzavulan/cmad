from __future__ import annotations

from torchvision.transforms import v2


def resize_crop(
    res: int, resize: int | None = None, interpolation: str = "bilinear"
) -> v2.Compose:
    """Short side -> `resize` (default `res`), then centre crop to `res`.

    The single definition of the image geometry: images use bilinear, ground-truth masks
    must go through the same call with interpolation="nearest" so the two stay aligned.
    """
    mode = v2.InterpolationMode(interpolation)
    return v2.Compose(
        [
            v2.Resize(
                resize or res,
                interpolation=mode,
                antialias=mode != v2.InterpolationMode.NEAREST,
            ),
            v2.CenterCrop(res),
        ]
    )
