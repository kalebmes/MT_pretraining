# SimCLR-style two-view contrastive dataset with age/sex auxiliary targets.

from pathlib import Path
from typing import Any, Dict, Optional, Union

from torch import Tensor

from .base import ParseDataset
from .normalization import masked_standardize

# max number of augmentation retries before PRETRAINT1DATASET gives up on a view
MAX_VIEW_RETRIES = 20

class PRETRAINT1DATASET(ParseDataset):
    def __init__(
        self,
        csv_path: Union[str, Path],
        transforms,
        *,
        image_col: str = "image_path",
        mask_col: str = "mask_path",
        age_col: Optional[str] = "age",
        sex_col: Optional[str] = "sex",
        root_dir: Optional[Union[str, Path]] = None,
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
        self.transform = transforms[0]  # spatial augmentations
        self.intensity_transform = transforms[1]  # intensity augmentations
        self.eps = eps

    def _process_view(self, raw_dict: Dict[str, Any]) -> Tensor:
        for _ in range(MAX_VIEW_RETRIES):
            d = self.transform({"MRI": raw_dict["MRI"], "MASK": raw_dict["MASK"]})
            img = d["MRI"].float().unsqueeze(0)  # [1, C, Z, Y, X]
            msk = d["MASK"].float().unsqueeze(0)

            img = masked_standardize(img, msk, eps=self.eps).squeeze(0)  # [C, Z, Y, X]
            img = self.intensity_transform({"MRI": img})["MRI"]

            bad = (
                img.isnan().any() or img.isinf().any()
                or msk.isnan().any() or msk.isinf().any()
                or msk.abs().max() == 0 or img.abs().max() == 0
            )
            if not bad:
                return img

        raise RuntimeError(
            f"Failed to produce a valid augmented view after {MAX_VIEW_RETRIES} attempts "
            f"for image_path={raw_dict.get('image_path')!r}. Likely a bad mask/crop combination."
        )

    def __getitem__(self, idx: int):
        sample = self._load_data(idx)
        x1 = self._process_view(sample)
        x2 = self._process_view(sample)
        return x1, x2, sample["age_info"], sample["sex_info"]
