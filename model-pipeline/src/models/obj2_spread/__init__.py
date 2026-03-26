"""
obj2_spread — Fire spread simulation models.

Public API
----------
Cell2FireSpread : primary spread model (Cell2Fire C++ wrapper)
Cell2FireError  : exception raised on simulation failure
"""
from .cell2fire_spread import Cell2FireSpread
from .exceptions import Cell2FireError, Cell2FireNotInstalledError

__all__ = [
    "Cell2FireSpread",
    "Cell2FireError",
    "Cell2FireNotInstalledError",
]
