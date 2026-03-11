"""TIGER audio separation models (vendored from github.com/JusperLee/TIGER)"""

from .tiger import TIGER
from .tiger_dnr import TIGERDNR

__all__ = ["TIGER", "TIGERDNR"]
