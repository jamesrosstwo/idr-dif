import pickle
from pathlib import Path
from typing import Any

import pandas as pd


class ExpStorage:
    def __init__(self, out_location: Path, entries=None):
        self.loc: Path = out_location
        self._entries = dict() if entries is None else entries
        self.metadata = dict()

    def add_entry(self, key: str, entry: Any):
        if key not in self._entries:
            self._entries[key] = list()
        self._entries[key].append(entry)

    def save(self):
        with open(str(self.loc), "wb") as handle:
            pickle.dump(self._entries, handle)

    def get_latest(self, key: str):
        assert key in self._entries
        return self._entries[key][-1]

    def to_df(self, *keys) -> pd.DataFrame:
        return pd.DataFrame.from_dict({k: self._entries[k] for k in keys})

    @classmethod
    def load(cls, load_path: Path):
        with open(str(load_path), "rb") as handle:
            return cls(load_path, pickle.load(handle))
