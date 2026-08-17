from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from torch.utils.data import Dataset

from .io_utils import load_nifti, resolve_path
from .metadata import build_column_lookup, parse_age_sex, resolve_column


class ParseDataset(Dataset):
    
    # Loads image/mask pairs from a CSV file, with optional age and sex metadata

    def __init__(
        self,
        dataset_path: Union[str, Path],
        *,
        image_col: str = "image_path",
        mask_col: str = "mask_path",
        age_col: Optional[str] = "age",
        sex_col: Optional[str] = "sex",
        root_dir: Optional[Union[str, Path]] = None,
    ):
        self.dataset_path = str(dataset_path) # path to CSV file
        self.image_col = image_col
        self.mask_col = mask_col
        self.root_dir = Path(root_dir) if root_dir is not None else None

        df = pd.read_csv(self.dataset_path)
        lookup = build_column_lookup(df.columns)
        self.age_col = resolve_column(age_col, lookup)
        self.sex_col = resolve_column(sex_col, lookup)
        self.rows = df.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_data(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        img_path = resolve_path(row[self.image_col], self.root_dir)
        msk_path = resolve_path(row[self.mask_col], self.root_dir)
        age, sex = parse_age_sex(row, self.age_col, self.sex_col)

        img = load_nifti(img_path)
        msk = load_nifti(msk_path)
        if img.shape != msk.shape:
            raise ValueError(f"Shape mismatch: {img.shape} vs {msk.shape}")

        # add channel dimension -> (1, Z, Y, X)
        img = img[None, ...]
        msk = msk[None, ...]

        return {
            "MRI": img,
            "MASK": msk,
            "image_path": img_path,
            "mask_path": msk_path,
            "age_info": age,
            "sex_info": sex,
            "row": row,
        }
