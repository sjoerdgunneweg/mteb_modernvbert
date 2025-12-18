from __future__ import annotations
import logging

from mteb.models.model_implementations.colpali_models import COLPALI_TRAINING_DATA, ColPaliEngineWrapper
from mteb._requires_package import (
    requires_package,
)

logger = logging.getLogger(__name__)

class ColModernVBertWrapper(ColPaliEngineWrapper):
    """Wrapper for ColModernVBert model."""

    def __init__(
        self,
        model_name: str = "SmolVEncoder/colvbert-modernbert_base-vidore",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

        super().__init__(
            model_name=model_name,
            model_class=ColModernVBert,
            processor_class=ColModernVBertProcessor,
            revision=revision,
            device=device,
            **kwargs,
        )

        if "torch_dtype" in kwargs:
            self.mdl.to(kwargs["torch_dtype"])

class BiModernVBertWrapper(ColPaliEngineWrapper):
    """Wrapper for BiVBert model."""

    def __init__(
        self,
        model_name: str = "SmolVEncoder/bivbert-slbert_210",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import BiModernVBert, BiModernVBertProcessor

        super().__init__(
            model_name=model_name,
            model_class=BiModernVBert,
            processor_class=BiModernVBertProcessor,
            revision=revision,
            device=device,
            **kwargs,
        )

        if "torch_dtype" in kwargs:
            self.mdl.to(kwargs["torch_dtype"])