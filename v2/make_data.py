"""Generate and cache every v2 dataset split (deterministic seeds; see ejepa/data.py SPLITS)."""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ejepa.data import SPLITS, load_split  # noqa: E402

if __name__ == "__main__":
    names = sys.argv[1:] or list(SPLITS)
    for name in names:
        t0 = time.time()
        ep = load_split(name, HERE / "cache")
        print(f"{name:14s} n={ep.n:5d} lam={ep.nuisance:g} regime={ep.regime} "
              f"omega=[{ep.omega.min():.0f},{ep.omega.max():.0f}] {time.time() - t0:.0f}s", flush=True)
