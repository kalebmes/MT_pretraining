# Supervised fine-tuning dataset for AD/MCI and MCI/CN classifications.

from pathlib import Path
from typing import Dict, Optional, Union

import torch

from .base import ParseDataset
from .normalization import masked_standardize


class DOWNSTREAMT1DATASET(ParseDataset):
    def __init__(
        self,
        csv_path: Union[str, Path],
        *,
        label_col: str,
        image_col: str = "image_path",
        mask_col: str = "mask_path",
        age_col: Optional[str] = "age",
        sex_col: Optional[str] = "sex",
        root_dir: Optional[Union[str, Path]] = None,
        label_mapping: Optional[Dict[str, int]] = None,
        label_dtype: torch.dtype = torch.float32,
        eps: float = 1e-6,
    ):
        super().__init__(
            csv_path,
            image_col=image_col,
            mask_col=mask_col,
            age_col=age_col,
            sex_col=sex_col,
            root_dir=root_dir,
        )
        self.label_col = label_col
        self.label_dtype = label_dtype
        self.eps = eps
        self.label_mapping = label_mapping

        if len(self.rows) > 0 and self.label_col not in self.rows[0]:
            raise ValueError(
                f"label_col={label_col!r} not found in CSV. Available keys: {list(self.rows[0].keys())}"
            )

    def __getitem__(self, idx: int):
        sample = self._load_data(idx)

        img = torch.from_numpy(sample["MRI"]).float().unsqueeze(0)  # [1, 1, D, H, W]
        msk = torch.from_numpy(sample["MASK"]).float().unsqueeze(0)  # [1, 1, D, H, W]
        img = masked_standardize(img, msk, eps=self.eps).squeeze(0)

        raw_y = sample["row"][self.label_col]
        if isinstance(raw_y, str):
            if self.label_mapping is None or raw_y not in self.label_mapping:
                raise ValueError(
                    f"Unknown label {raw_y!r} in column {self.label_col!r}. "
                    f"Expected one of: {list((self.label_mapping or {}).keys())}"
                )
            raw_y = self.label_mapping[raw_y]

        y = torch.tensor(raw_y, dtype=self.label_dtype)

        return {"MRI": img, "task_label": y, "age": sample["age_info"], "sex": sample["sex_info"]}
