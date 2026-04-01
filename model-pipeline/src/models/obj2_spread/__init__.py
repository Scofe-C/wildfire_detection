"""
obj2_spread — Fire spread simulation models.

Public API
----------
PythonFireSpreadSimulator : pure-Python deterministic spread simulator (primary)
Cell2FireSpread           : Cell2Fire C++ wrapper (legacy, kept for reference)
Cell2FireError            : exception raised on Cell2Fire simulation failure
"""
from .fire_spread_simulator import PythonFireSpreadSimulator
from .cell2fire_spread import Cell2FireSpread
from .exceptions import Cell2FireError, Cell2FireNotInstalledError

__all__ = [
    "PythonFireSpreadSimulator",
    "Cell2FireSpread",
    "Cell2FireError",
    "Cell2FireNotInstalledError",
]
