"""Auto-discovery for metrics subpackage."""
import importlib
import pkgutil
from pathlib import Path

from src.metrics.registry import METRICS

package_dir = Path(__file__).parent
for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
    if not module_name.startswith("_") and module_name != "registry":
        importlib.import_module(f"src.metrics.{module_name}")

__all__ = ["METRICS"]
