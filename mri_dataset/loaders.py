import torch

from .downstream_dataset import DOWNSTREAMT1DATASET


def _make_downstream_loader(
    csv_path: str,
    *,
    label_col: str,
    label_mapping: dict,
    label_dtype: torch.dtype,
    root_dir,
    image_col: str,
    mask_col: str,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    pin_mem: bool,
) -> torch.utils.data.DataLoader:
    dataset = DOWNSTREAMT1DATASET(
        csv_path,
        label_col=label_col,
        label_dtype=label_dtype,
        root_dir=root_dir,
        image_col=image_col,
        mask_col=mask_col,
        label_mapping=label_mapping,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=num_workers > 0,
    )



def make_downstream_loaders(
    *,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    batch_size: int,
    label_mapping: dict,
    num_workers: int = 8,
    pin_mem: bool = True,
    drop_last: bool = True,
    label_dtype: torch.dtype = torch.float32,
    root_dir=None,
    label_col: str = "label",
    image_col: str = "image_path",
    mask_col: str = "mask_path",
):
    common = dict(
        label_col=label_col,
        label_mapping=label_mapping,
        label_dtype=label_dtype,
        root_dir=root_dir,
        image_col=image_col,
        mask_col=mask_col,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=pin_mem,
    )

    train_loader = _make_downstream_loader(train_csv, shuffle=True, drop_last=drop_last, **common)
    val_loader = _make_downstream_loader(val_csv, shuffle=False, drop_last=False, **common)
    test_loader = _make_downstream_loader(test_csv, shuffle=False, drop_last=False, **common)

    return train_loader, val_loader, test_loader
