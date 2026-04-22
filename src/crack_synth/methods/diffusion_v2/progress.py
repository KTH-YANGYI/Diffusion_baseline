from __future__ import annotations

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(iterable, **_: object):
        return iterable


tqdm = _tqdm
