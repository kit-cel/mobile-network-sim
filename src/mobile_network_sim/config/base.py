"""Strict, serializable configuration models shared by radio and RRC."""

from __future__ import annotations

from abc import ABC
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, TypeVar, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt
import yaml

# Type aliases for strict types with additional constraints
StrictNonNegativeFloat = Annotated[StrictFloat, Field(ge=0.0)]
StrictNonNegativeInt = Annotated[StrictInt, Field(ge=0)]
StrictPositiveFloat = Annotated[StrictFloat, Field(gt=0.0)]
StrictPositiveInt = Annotated[StrictInt, Field(gt=0)]
StrictProbability = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]

# Type variable for BaseConfig and its subclasses
T = TypeVar("T", bound="BaseConfig")


def _serializable(value: Any) -> Any:
    """
    Serialization helper for nested config objects.
    This is necessary to safely export configs as YAML files in the next setp.
    """
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, BaseModel):
        return _serializable(value.model_dump(mode="python", serialize_as_any=True))
    if is_dataclass(value) and not isinstance(value, type):
        return _serializable(asdict(cast(Any, value)))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serializable(item) for item in value]
    return value


class BaseConfig(BaseModel, ABC):
    """Base for strict YAML-backed simulator configuration.

    Attributes
    ----------
    seed : int
        Non-negative component seed. It is excluded from serialized output
        because the enclosing simulation configuration supplies it.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    seed: StrictNonNegativeInt = Field(
        default=42,
        exclude=True,
        description="Internal component seed inherited from the simulation seed.",
    )

    @classmethod
    def from_yaml(cls: type[T], path: str | Path) -> T:
        """
        Load a configuration object from a YAML file.

        Parameters
        ----------
        path : str | Path
            Path to the YAML file.

        Returns
        -------
        T
            An instance of the configuration class, e.g., `RadioConfig` or `RRCConfig`.
        """
        with Path(path).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Expected a YAML mapping in {path!s}")
        return cls.model_validate(data)

    def export_as_yaml(self, path: str | Path) -> None:
        """
        Export the configuration object to a YAML file.

        Parameters
        ----------
        path : str | Path
            Path to the output YAML file.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.to_dict(), stream, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration to JSON/YAML-compatible values.

        Returns
        -------
        dict of str to Any
            Recursively serialized configuration fields.
        """
        return _serializable(self.model_dump(mode="python", serialize_as_any=True))

    def update_recursively(self, updates: dict[str, Any]) -> bool:
        """
        Recursively update the configuration object with values from a dictionary.

        Parameters
        ----------
        updates : dict[str, Any]
            A dictionary containing the updates to apply.

        Returns
        -------
        bool
            True if any values were updated, False otherwise.

        Examples
        --------
        >>> from mobile_network_sim.config import BaseConfig
        >>> class ExampleConfig(BaseConfig):
        ...     value: int = 1
        >>> config = ExampleConfig()
        >>> config.update_recursively({"value": 2})
        True
        """
        changed = False
        for name in type(self).model_fields:
            value = getattr(self, name)
            if name in updates and value != updates[name]:
                setattr(self, name, updates[name])
                changed = True
                continue
            if isinstance(value, BaseConfig):
                changed = value.update_recursively(updates) or changed
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, BaseConfig):
                        changed = item.update_recursively(updates) or changed
        return changed

    @property
    def fingerprint(self) -> str:
        """Return a stable short fingerprint of the serialized configuration.

        Returns
        -------
        str
            First 16 hexadecimal digits of a SHA-256 digest.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
