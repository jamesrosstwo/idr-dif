import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Union, Dict

import pandas as pd
import torch


class DataCache(ABC):
    def __init__(self):
        self._entries: Dict[str, List[Any]] = dict()

    def add_entry(self, key: str, val: Any):
        if key not in self._entries:
            self._entries[key] = list()
        self._entries[key].append(val)

    @abstractmethod
    def peek_all(self):
        pass

    def pop_all(self) -> Dict[str, Any]:
        out = self.peek_all()
        self._entries = dict()
        return out


class TensorStackCache(DataCache):
    def __init__(self, cuda=True):
        self._stack_fn = lambda v: torch.stack(v)
        if cuda:
            self._stack_fn = lambda v: torch.stack(v).cuda()
        super().__init__()

    def add_entry(self, key: str, val: torch.Tensor):
        assert isinstance(val, torch.Tensor)
        super().add_entry(key, val)

    def peek_all(self) -> Dict[str, torch.Tensor]:
        return {k: self._stack_fn(v) for k, v in self._entries.items()}


class AvgCache(DataCache):
    def add_entry(self, key: str, val: Union[float, int, torch.Tensor]):
        assert isinstance(val, float) or isinstance(val, int) or isinstance(val, torch.Tensor)
        super().add_entry(key, val)

    def peek_all(self):
        return {k: sum(v) / len(v) for k, v in self._entries.items()}

class ExpStorage:
    def __init__(self, out_location: Path, entries=None):
        self.loc: Path = out_location
        self._entries = dict() if entries is None else entries
        self.metadata = dict()
        self._cache = AvgCache()

    @classmethod
    def load(cls, load_path: Path):
        with open(str(load_path), "rb") as handle:
            return cls(load_path, pickle.load(handle))

    @property
    def keys(self):
        return list(self._entries.keys())

    def cache(self, key: str, val: float):
        self._cache.add_entry(key, val)

    def store(self, key: str, entry: Any):
        if key not in self._entries:
            self._entries[key] = list()
        self._entries[key].append(entry)

    def get_all(self, *keys) -> Union[List[Any], Dict[str, List[Any]]]:
        assert len(keys) > 0
        assert all([key in self._entries for key in keys])
        if len(keys) == 1:
            return self._entries[keys[0]]
        return {key: self._entries[key] for key in keys}

    def get_latest(self, *keys):
        if len(keys) == 1:
            return self.get_all(*keys)[-1]
        return {k: v[-1] for k, v in self.get_all(*keys).items()}

    def get_latest_with_default(self, key, default=None):
        if key in self.keys:
            return self._entries[key][-1]
        return default

    def to_df(self, *keys) -> pd.DataFrame:
        return pd.DataFrame.from_dict({k: self._entries[k] for k in keys})

    def delete_after(self, idx: int, *keys):
        assert all([key in self._entries for key in keys])
        for key in keys:
            self._entries[key] = self._entries[key][:idx]

    def pop_cache(self):
        for k, v in self._cache.pop_all().items():
            self.store(k, v)

    def peek_cache(self):
        for k, v in self._cache.peek_all().items():
            self.store(k, v)

    def change_path(self, new_path: Path):
        self.loc = new_path

    def save(self):
        with open(str(self.loc), "wb") as handle:
            pickle.dump(self._entries, handle)
