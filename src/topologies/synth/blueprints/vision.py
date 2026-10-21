"""Vision blueprints: procedural rendering + domain randomization, exact labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from topologies.synth.blueprints._base import blueprint, default_splits, spec_kwargs
from topologies.synth.errors import unknown
from topologies.synth.spec import Spec


def _randomize(randomize: Any) -> Any:
    from topologies.synth.lib.image import Randomize

    if randomize is True:
        return Randomize()
    if randomize in (None, False):
        return None
    if isinstance(randomize, Mapping):
        return Randomize(**randomize)
    return randomize


@blueprint(example={"kind": "shapes", "classes": ["circle", "square", "triangle"],
                    "image_size": 32, "size": 600})
def image_classification(name: str, *, kind: str = "shapes",
                         classes: Sequence[str] | str | None = None, image_size: int = 32,
                         channels: int | None = None, randomize: Any = True,
                         size: int = 5000, splits: Any = None, **spec: Any) -> Any:
    """Procedural images with exact class labels: ``kind="shapes"`` (geometric shapes) or
    ``"glyphs"`` (characters, ``classes`` = a string of characters); domain
    randomization (backgrounds, lighting, blur, noise, affine) varies everything but the
    label."""
    from topologies.synth.lib.image import Glyphs, Shapes

    if kind == "shapes":
        gen: Any = Shapes(size=image_size, classes=tuple(classes or ("circle", "square",
                                                                     "triangle")),
                          channels=channels or 3, randomize=_randomize(randomize))
        labels = list(gen.classes)
    elif kind == "glyphs":
        chars = "".join(classes) if classes else "0123456789"
        gen = Glyphs(chars=chars, size=image_size, channels=channels or 1,
                     randomize=_randomize(randomize))
        labels = list(chars)
    else:
        raise unknown("image kind", kind, ["shapes", "glyphs"], stage="blueprint")
    return Spec(name, generator=gen, attributes={"label": labels}, size=size,
                splits=splits if splits is not None else default_splits("label"),
                **spec_kwargs(spec))


@blueprint(example={"classes": ["circle", "square", "triangle"], "image_size": 64,
                    "size": 300})
def detection(name: str, *, classes: Sequence[str] = ("circle", "square", "triangle"),
              image_size: int = 64, objects: tuple[int, int] = (1, 4), channels: int = 3,
              randomize: Any = True, size: int = 5000, splits: Any = None,
              **spec: Any) -> Any:
    """Scenes of several objects with exact boxes and masks (object detection)."""
    from topologies.synth.lib.image import Scenes

    gen = Scenes(size=image_size, classes=tuple(classes), n_objects=tuple(objects),
                 channels=channels, randomize=_randomize(randomize))
    return Spec(name, generator=gen, size=size,
                splits=splits if splits is not None else default_splits(),
                **spec_kwargs(spec))


@blueprint(example={"texts": None, "height": 16, "width": 128, "size": 500})
def ocr_lines(name: str, *, texts: Sequence[str] | None = None, height: int = 16,
              width: int = 128, chars: tuple[int, int] = (3, 16), alphabet: str | None = None,
              randomize: Any = True, size: int = 5000, splits: Any = None,
              **spec: Any) -> Any:
    """Rendered text lines with exact transcriptions (OCR): random strings over
    ``alphabet`` or lines drawn from ``texts``."""
    from topologies.synth.lib.image import TextLines

    kw: dict[str, Any] = {"alphabet": alphabet} if alphabet else {}
    gen = TextLines(height=height, width=width, min_chars=int(chars[0]),
                    max_chars=int(chars[1]), randomize=_randomize(randomize),
                    texts=list(texts) if texts else None, **kw)
    return Spec(name, generator=gen, size=size,
                splits=splits if splits is not None else default_splits(),
                **spec_kwargs(spec))


__all__ = ["detection", "image_classification", "ocr_lines"]
