"""
SigFeatX - _progress.py
========================
Optional tqdm wrapper. Falls back to a no-op when tqdm is not installed.

Internal helper used by FeatureAggregator.extract_batch and related methods.
"""

from typing import Iterable, Optional


def progress_iter(iterable: Iterable, *, total: Optional[int] = None,
                  desc: str = "Extracting", enabled: bool = True):
    """
    Wrap an iterable with tqdm if available and ``enabled=True``.

    Falls back to the original iterable otherwise so callers stay clean.
    """
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, leave=False)
    except ImportError:
        return iterable


class ProgressBar:
    """
    Context-managed progress bar for cases where we receive results via
    ``as_completed`` rather than a clean iterable.
    """

    def __init__(self, total: int, desc: str = "Extracting", enabled: bool = True):
        self.total = total
        self.desc = desc
        self.enabled = enabled
        self._bar = None
        self._fallback_count = 0

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            from tqdm import tqdm
            self._bar = tqdm(total=self.total, desc=self.desc, leave=False)
        except ImportError:
            self._bar = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._bar is not None:
            self._bar.close()
        elif self.enabled and self._fallback_count:
            # Move past the carriage-return line
            print()

    def update(self, n: int = 1) -> None:
        if not self.enabled:
            return
        if self._bar is not None:
            self._bar.update(n)
        else:
            self._fallback_count += n
            print(f"\r  {self.desc}: {self._fallback_count}/{self.total}",
                  end="", flush=True)