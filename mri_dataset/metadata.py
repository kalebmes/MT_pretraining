# Age/sex metadata parsing, isolated so both dataset classes share one code path.

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# Accepted string encodings for sex, normalized to lowercase before lookup.
SEX_MAPPING = {"f": 0, "m": 1, "female": 0, "male": 1}

def build_column_lookup(columns) -> Dict[str, str]:
    # Map lowercase column name -> actual column name, for case-insensitive access.
    return {c.lower(): c for c in columns}


def resolve_column(col: Optional[str], lookup: Dict[str, str]) -> Optional[str]:
    if col is None:
        return None
    return lookup.get(col.lower())


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def parse_age_sex(
    row: Dict[str, Any],
    age_col: Optional[str],
    sex_col: Optional[str],
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """
    Extract (age, sex) as float32 scalar tensors from a CSV row.
    Returns (None, None) if either column is absent or the value is missing/NaN.
    Raises ValueError on an unrecognized string sex value.
    """
    if age_col is None or sex_col is None:
        return None, None
    if age_col not in row or sex_col not in row:
        return None, None

    age_info = row[age_col]
    sex_info = row[sex_col]

    if _is_missing(age_info) or _is_missing(sex_info):
        return None, None

    if isinstance(sex_info, str):
        key = sex_info.strip().lower()
        if key not in SEX_MAPPING:
            raise ValueError(
                f"Unknown sex value {sex_info!r}. Expected one of: {sorted(set(SEX_MAPPING))}"
            )
        sex_info = SEX_MAPPING[key]

    age = torch.tensor(float(age_info), dtype=torch.float32)
    sex = torch.tensor(float(sex_info), dtype=torch.float32)
    return age, sex
