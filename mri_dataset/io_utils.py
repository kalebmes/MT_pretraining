# Path resolution and nifti loading helpers.

from pathlib import Path
from typing import Any, Optional, Union

import nibabel as nib
import numpy as np


def resolve_path(p: Any, root_dir: Optional[Path]) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        raise ValueError("Found missing path in CSV; image_path and mask_path are required.")
    p = str(p)
    if root_dir is not None and not Path(p).is_absolute():
        return str(root_dir / p)
    return p


def load_nifti(path: Union[str, Path]) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
