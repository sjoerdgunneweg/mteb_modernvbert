from __future__ import annotations

import logging
from functools import partial

import torch
from transformers.utils.import_utils import is_flash_attn_2_available

from mteb.models.colpali_models import COLPALI_TRAINING_DATA, ColPaliEngineWrapper
from mteb.requires_package import (
    requires_package,
)

logger = logging.getLogger(__name__)

class SparseModernVBertMLMWrapper(ColPaliEngineWrapper):
    """Wrapper for SparseModernVBertMLM model."""

    def __init__(
        self,
        model_name: str = "SparseModernVBERT/sparsemodernvbertmlm",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import SparseModernVBertMLM, SparseModernVBertMLMProcessor

        super().__init__(
            model_name=model_name,
            model_class=SparseModernVBertMLM,
            processor_class=SparseModernVBertMLMProcessor,
            revision=revision,
            device=device,
            **kwargs,
        )

        if "torch_dtype" in kwargs:
            self.mdl.to(kwargs["torch_dtype"])