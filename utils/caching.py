from pathlib import Path
from joblib import Memory

# Configure caching
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
memory = Memory(CACHE_DIR, verbose=0)