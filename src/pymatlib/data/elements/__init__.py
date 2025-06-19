"""Chemical element data and definitions."""

from .element_data import (
    element_map,
    CARBON, NITROGEN, ALUMINIUM, SILICON, PHOSPHORUS, SULFUR,
    TITANIUM, VANADIUM, CHROMIUM, MANGANESE, IRON, NICKEL, MOLYBDENUM
)

__all__ = [
    "element_map",
    # Individual elements for direct access
    "CARBON", "NITROGEN", "ALUMINIUM", "SILICON", "PHOSPHORUS", "SULFUR",
    "TITANIUM", "VANADIUM", "CHROMIUM", "MANGANESE", "IRON", "NICKEL", "MOLYBDENUM"
]
