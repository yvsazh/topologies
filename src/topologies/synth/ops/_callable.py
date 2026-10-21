"""Callable modules: ``synth.ops.verify(fn)`` and ``synth.ops.verify.round_trip(...)``.

A submodule calls ``make_callable(__name__, "verify")`` at import time; the
module object then forwards calls to that function. (A plain namespace object in
the package would be replaced by the submodule as soon as it is imported.)
"""

from __future__ import annotations

import sys
import types
from typing import Any


class CallableModule(types.ModuleType):
    __synth_call__: str = ""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return getattr(self, self.__synth_call__)(*args, **kwargs)


def make_callable(module_name: str, function: str) -> None:
    mod = sys.modules[module_name]
    cls = type(f"Callable_{module_name.rsplit('.', 1)[-1]}", (CallableModule,),
               {"__synth_call__": function})
    mod.__class__ = cls


__all__ = ["CallableModule", "make_callable"]
