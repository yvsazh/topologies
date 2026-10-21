import importlib.util
from pathlib import Path

_source = Path(__file__).parents[1] / "layers" / "han.py"
_spec = importlib.util.spec_from_file_location("han_layer", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
HAN = _module.HAN

__all__ = ["HAN"]
