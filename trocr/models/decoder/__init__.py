"""Text-decoder configuration helpers."""

from .model import TrOCRForCausalLM
from .reinit_token import configure_decoder

__all__ = ["TrOCRForCausalLM", "configure_decoder"]
