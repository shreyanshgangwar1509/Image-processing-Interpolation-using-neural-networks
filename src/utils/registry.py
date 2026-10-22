"""Generic registry implementation."""
from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")


class Registry:
    """Registry class for mapping string names to classes or functions."""

    def __init__(self, name: str) -> None:
        self.name: str = name
        self._module_dict: Dict[str, Any] = {}

    def register(self, name: Optional[Any] = None) -> Any:
        """Decorator to register a class or function.

        Can be used as @registry.register, @registry.register(), or @registry.register("custom_name").
        """
        if callable(name):
            obj = name
            reg_name = getattr(obj, "__name__", str(obj))
            if reg_name in self._module_dict:
                raise KeyError(
                    f"[{self.name}] '{reg_name}' is already registered in '{self.name}' registry!"
                )
            self._module_dict[reg_name] = obj
            return obj

        def _register(obj: T) -> T:
            reg_name = name if name is not None else getattr(obj, "__name__", str(obj))
            if reg_name in self._module_dict:
                raise KeyError(
                    f"[{self.name}] '{reg_name}' is already registered in '{self.name}' registry!"
                )
            self._module_dict[reg_name] = obj
            return obj

        return _register

    def get(self, name: str) -> Any:
        """Retrieves a registered item by name. Raises KeyError if not found."""
        if name not in self._module_dict:
            avail = ", ".join(sorted(self._module_dict.keys())) or "none"
            raise KeyError(
                f"[{self.name}] Unknown name '{name}'. Available in '{self.name}': [{avail}]"
            )
        return self._module_dict[name]

    def build(self, name: str, **kwargs: Any) -> Any:
        """Builds/instantiates an object from the registry using provided kwargs."""
        obj = self.get(name)
        return obj(**kwargs)

    def names(self) -> List[str]:
        """Returns a sorted list of registered component names."""
        return sorted(list(self._module_dict.keys()))
