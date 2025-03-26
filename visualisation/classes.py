from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

@dataclass
class Run:
    name: str
    df: pd.DataFrame
    binary: str = ""
    meta: str = ""
    tag: str = ""

    def concat(self, other_run):
        self.df = pd.concat([self.df, other_run.df])
        return self

    def merge(self, other_run):
        _vals = np.array([self.df.values, other_run.df.values])
        self.df = pd.DataFrame(np.mean(_vals, axis=0))
        return self

    def set_name(self, name: str):
        self.name = name
        return self
