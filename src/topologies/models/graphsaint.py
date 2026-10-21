import importlib.util
from pathlib import Path

_source = Path(__file__).parents[1] / "layers" / "graphsaint.py"
_spec = importlib.util.spec_from_file_location("graphsaint_layer", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

GraphSAINT = _module.GraphSAINT

__all__ = ["GraphSAINT"]
