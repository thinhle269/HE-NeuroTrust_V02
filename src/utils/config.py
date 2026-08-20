"""YAML configuration loader with attribute-style access."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class AttrDict(dict):
    """Dictionary that allows attribute-style access for nested dicts."""

    def __init__(self, data: Dict[str, Any]):
        super().__init__()
        for k, v in data.items():
            self[k] = self._convert(v)

    @classmethod
    def _convert(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return cls(v)
        if isinstance(v, list):
            return [cls._convert(item) for item in v]
        return v

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_plain(self) -> Dict[str, Any]:
        def _back(v: Any) -> Any:
            if isinstance(v, AttrDict):
                return {k: _back(val) for k, val in v.items()}
            if isinstance(v, list):
                return [_back(x) for x in v]
            return v

        return {k: _back(v) for k, v in self.items()}


def load_config(path: str | Path) -> AttrDict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    return AttrDict(data)
