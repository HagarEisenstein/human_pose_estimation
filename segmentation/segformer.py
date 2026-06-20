"""
SegFormer-based human-parsing segmentor.

Wraps the pretrained ``mattmdjaga/segformer_b2_clothes`` checkpoint from
HuggingFace Hub behind the SegmentationModel interface.  Source labels
(18 ATR clothes-parsing classes) are remapped to canonical ``Part`` labels
via SEGFORMER_CLOTHES_TO_PART.

Usage
-----
    from segmentation.segformer import SegFormerSegmentor

    seg = SegFormerSegmentor()                # downloads weights on first call
    mask = seg.predict(sample)                # (H, W) uint8 with canonical Parts

Requirements
------------
    pip install "transformers>=4.36" "Pillow>=10.0"

The first call downloads ~100 MB to the HuggingFace cache directory.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pose.parts import SEGFORMER_CLOTHES_TO_PART, remap_mask
from segmentation.base import SegmentationModel

DEFAULT_CHECKPOINT = "mattmdjaga/segformer_b2_clothes"


class SegFormerSegmentor(SegmentationModel):
    """Wrap a HuggingFace SegFormer human-parsing model.

    Args:
        checkpoint: HuggingFace model id or local path.  Defaults to
            ``mattmdjaga/segformer_b2_clothes``.
        device:     ``"cpu"``, ``"cuda"``, or ``"cuda:N"``.  Default ``"cpu"``.
        remap:      Source-label-to-Part mapping.  Defaults to
            ``SEGFORMER_CLOTHES_TO_PART``.  Override only if you load a
            differently-trained SegFormer checkpoint.
        model:      Optional preloaded model instance — used by tests to inject
            a mock without touching HuggingFace.  When supplied, ``processor``
            must also be supplied.
        processor:  Optional preloaded image processor (see ``model``).

    The model and processor are loaded lazily the first time ``predict()``
    is called, so importing this module is cheap and never touches the
    network on its own.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str = "cpu",
        remap: dict[int, Any] | None = None,
        *,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.remap = remap if remap is not None else SEGFORMER_CLOTHES_TO_PART
        self._model = model
        self._processor = processor

    def _ensure_loaded(self) -> None:
        """Load weights + processor on first use.  No-op after that."""
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch  # noqa: F401  (used implicitly by transformers)
            from transformers import (
                AutoImageProcessor,
                SegformerForSemanticSegmentation,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SegFormerSegmentor requires the optional 'transformers' "
                "extra.  Install with:  pip install 'transformers>=4.36'"
            ) from exc

        self._processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self._model = SegformerForSemanticSegmentation.from_pretrained(
            self.checkpoint
        ).to(self.device).eval()

    def predict(self, sample) -> np.ndarray:
        """Run SegFormer on sample.image and return a canonical (H, W) mask."""
        import torch
        from PIL import Image

        self._ensure_loaded()

        pil = Image.fromarray(sample.image)  # sample.image is RGB uint8
        inputs = self._processor(images=pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self._model(**inputs).logits  # (1, C, h, w) at low res

        # Up-sample logits back to the original image size, then argmax.
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=(sample.height, sample.width),
            mode="bilinear",
            align_corners=False,
        )
        source_mask = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.int32)

        return remap_mask(source_mask, self.remap)
