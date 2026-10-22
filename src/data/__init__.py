"""Auto-discovery for data subpackage."""
import importlib
import pkgutil
from pathlib import Path

from src.data.registry import DATASETS, DEGRADATIONS

# Auto-discover all modules in this package (skip _* and registry)
package_dir = Path(__file__).parent
for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
    if not module_name.startswith("_") and module_name != "registry":
        importlib.import_module(f"src.data.{module_name}")

__all__ = ["DATASETS", "DEGRADATIONS"]
