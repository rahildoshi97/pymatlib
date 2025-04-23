import sympy as sp
from typing import Dict


class SymbolRegistry:
    """Registry for SymPy symbols to ensure uniqueness."""
    _symbols = {}

    @classmethod
    def get(cls, name: str) -> sp.Symbol:
        """Get or create a symbol with the given name."""
        if name not in cls._symbols:
            cls._symbols[name] = sp.Symbol(name)
        return cls._symbols[name]

    @classmethod
    def get_all(cls) -> Dict[str, sp.Symbol]:
        """Get all registered symbols."""
        return cls._symbols.copy()

    @classmethod
    def clear(cls) -> None:
        """Clear all registered symbols."""
        cls._symbols.clear()