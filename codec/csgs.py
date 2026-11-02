"""Compatibility alias: `codec.csgs` is the old name of `codec.falt` (Faltwerk format, formerly CSGS).

Binds this module name to the very same module object, so `from codec.csgs import X`, `codec.csgs.X` and monkeypatches
behave exactly as with `codec.falt`. Nothing else lives here; new code imports codec.falt.
"""
import sys

from . import falt

sys.modules[__name__] = falt
