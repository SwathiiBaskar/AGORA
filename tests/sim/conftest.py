import os
import tempfile
from pathlib import Path

# Hypothesis caches data in a ``.hypothesis`` directory under the working directory by
# default, even with its example database disabled. Keep it out of the repository.
os.environ.setdefault(
    "HYPOTHESIS_STORAGE_DIRECTORY", str(Path(tempfile.gettempdir()) / "civbox-hypothesis")
)
