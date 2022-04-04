import pickle
from pathlib import Path
from typing import Any, List, Union, Dict

import pandas as pd


class ExpStorage:
    def __init__(self, out_location: Path, entries=None):
        self.loc: Path = out_location
        self._entries = dict() if entries is None else entries
        self.metadata = dict()

    @classmethod
    def load(cls, load_path: Path):
        with open(str(load_path), "rb") as handle:
            return cls(load_path, pickle.load(handle))

    @property
    def keys(self):
        return list(self._entries.keys())

    def add_entry(self, key: str, entry: Any):
        if key not in self._entries:
            self._entries[key] = list()
        self._entries[key].append(entry)

    def save(self):
        with open(str(self.loc), "wb") as handle:
            pickle.dump(self._entries, handle)

    def get_all(self, *keys) -> Union[List[Any], Dict[str, List[Any]]]:
        assert len(keys) > 0
        assert all([key in self._entries for key in keys])
        if len(keys) == 1:
            return self._entries[keys[0]]
        return {key: self._entries[key] for key in keys}

    def get_latest(self, *keys):
        if len(keys) == 1:
            return self.get_all(*keys)[-1]
        return {k: v[-1] for k, v in self.get_all(keys).items()}

    def get_latest_with_default(self, key, default=None):
        if key in self.keys:
            return self._entries[key][-1]
        return default

    def to_df(self, *keys) -> pd.DataFrame:
        return pd.DataFrame.from_dict({k: self._entries[k] for k in keys})

    def change_path(self, new_path: Path):
        self.loc = new_path

    def delete_after(self, idx: int, *keys):
        assert all([key in self._entries for key in keys])
        for key in keys:
            self._entries[key] = self._entries[key][:idx]
