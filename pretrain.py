# Pretraining script adapted from Kaczmarek et al. (2025) 3D T1 SimCLR implementation. https://github.com/emilykaczmarek/3D-Neuro-SimCLR/
# modified to include age prediction + sex classification auxiliary heads and auxiliary loss terms.

import os
import numpy as np
import random
import torch
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from mri_dataset.pretrain_dataset import PRETRAINT1DATASET

from monai.transforms import (
    Compose,
    RandSpatialCropd,
    Resized,
    RandFlipd,
    RandRotated,
    RandShiftIntensityd,
    RandAdjustContrastd,
)

# for reproducibility
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args, train_loader, model, criterion, optimizer):
    loss_epoch = 0
    contrastive_loss_epoch = 0
    age_loss_epoch = 0
    sex_loss_epoch = 0
    try:
        scaler = torch.amp.GradScaler()
    except Exception as e:
        scaler = torch.cuda.amp.GradScaler()

    mae_loss = torch.nn.L1Loss()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    for step, (x_i, x_j, x_age, x_sex) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        
        x_i = x_i.to(args.device, non_blocking=True)
        x_j = x_j.to(args.device, non_blocking=True)
        ages = x_age.to(args.device, non_blocking=True).float()
        sexes = x_sex.to(args.device, non_blocking=True).float()

        with torch.amp.autocast(args.device.type):
            z_i, z_j, pred_age, pred_sex = model(x_i, x_j)
        
        with torch.amp.autocast(args.device.type):   
            contrastive_loss = criterion(z_i, z_j)
            age_loss = mae_loss(pred_age.squeeze(), ages)
            sex_loss = bce_loss(pred_sex.squeeze(), sexes)
            # combined loss
            loss = contrastive_loss + args.age_weight * age_loss + args.sex_weight * sex_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.rank == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}] Contrastive Loss: {contrastive_loss.item():.4f} Age Loss: {age_loss.item():.4f} Sex Loss: {sex_loss.item():.4f} Combined Loss: {loss.item():.4f}")

        loss_epoch += loss.item()
        contrastive_loss_epoch += contrastive_loss.item()
        age_loss_epoch += age_loss.item()
        sex_loss_epoch+= sex_loss.item()
    
    return loss_epoch, contrastive_loss_epoch, age_loss_epoch, sex_loss_epoch


def replace_relu_inplace(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.ReLU) and child.inplace:
            setattr(module, name, torch.nn.ReLU(inplace=False))
        else:
            replace_relu_inplace(child)

def save_model(args, model, optimizer, best_metric, latest=False, best=False):
    if latest:
        save_path = os.path.join(args.save_model_path, "latest.tar")
    elif best:
        save_path = os.path.join(args.save_model_path, f"best.tar")
    else:
        save_path = os.path.join(args.save_model_path, f"epoch_{args.current_epoch}.tar")

    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    
    checkpoint = {
        'epoch': args.current_epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
    }
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
        print(f"Created directory {args.save_model_path} for saving checkpoints.")
    torch.save(checkpoint, save_path)

def load_checkpoint(args, model, optimizer=None, path=None, best_metric=None):
    path = path or os.path.join(args.save_model_path, "latest.tar")
    ckpt = torch.load(path, map_location=args.device)

    # Remove 'module.' prefix if present (for DP/DDP models)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)

    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if best_metric is not None and "best_metric" in ckpt:
        args.best_metric = ckpt["best_metric"]
    args.current_epoch = ckpt.get("epoch", 0)
    args.start_epoch = ckpt.get("epoch", 0)
    if args.rank == 0:
        print(f"Resumed from checkpoint at epoch {args.current_epoch}")


def main(local_rank: int, args):

    # Get rank and world size from torchrun env vars
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Set device for this process
    torch.cuda.set_device(local_rank)
    args.device = torch.device(f"cuda:{local_rank}")

    args.world_size = world_size
    args.rank = rank
    args.local_rank = local_rank

    dist.init_process_group(backend="nccl", init_method="env://")
    seed_everything(args.seed)

    spatial_transform = Compose([
        RandSpatialCropd(keys=["MRI", "MASK"], roi_size=(30, 40, 40), random_center=True, random_size=True),
        Resized(keys=["MRI", "MASK"], spatial_size=(96, 96, 96), mode=["trilinear", "nearest"]),
        RandFlipd(keys=["MRI", "MASK"], prob=0.5, spatial_axis=[2]),
        RandRotated(keys=["MRI", "MASK"], range_x=0.785, prob=0.5, mode=["trilinear", "nearest"]),
        ])
    intensity_transform = Compose([
        RandShiftIntensityd(keys=["MRI"], offsets=0.5, prob=0.8),
        RandAdjustContrastd(keys=["MRI"], gamma=(0.5, 1.5), prob=0.8),
        ])

    train_ds = PRETRAINT1DATASET(args.data_file, 
                                 transforms=[spatial_transform, intensity_transform],
                                 image_col=args.image_col,
                                 mask_col=args.mask_col,
                                 age_col=args.age_col,
                                 sex_col=args.sex_col,)

    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=train_ds,
            num_replicas=world_size,
            rank=rank, shuffle=True)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=False,
        num_workers=args.num_workers,
    )

    # Our ResNet Model
    encoder = get_resnet(args.resnet, pretrained=False)
    model = SimCLR(encoder, args.projection_dim, encoder.fc.in_features)
    replace_relu_inplace(model)
    model = model.to(args.device)

    # optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

    # Load checkpoint if requested
    if args.reload:
        ckpt_path = os.path.join(args.save_model_path, "latest.tar")
        if os.path.exists(ckpt_path):
            load_checkpoint(args, model, optimizer)
        else:
            if args.rank == 0:
                print("Reload requested, but latest.tar not found — starting fresh.")


    # DDP / DP
    if args.world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )

    min_comb_loss = float("inf") if not hasattr(args, 'best_metric') else args.best_metric
    args.current_epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch, contrastive_loss_epoch, age_loss_epoch, sex_loss_epoch = train(args, train_loader, model, criterion, optimizer)

        if args.rank == 0 and epoch % 100 == 0:
            save_model(args, model, optimizer, min_comb_loss, latest=False, best=False)

        if args.rank == 0:
            # Calculate average losses
            num_steps = len(train_loader)
            avg_loss = loss_epoch / num_steps
            avg_contrastive = contrastive_loss_epoch / num_steps
            avg_age = age_loss_epoch / num_steps
            avg_sex = sex_loss_epoch / num_steps
            print('#' * 50)
            print(f"Epoch [{epoch}/{args.epochs}], Contrastive Loss: {avg_contrastive}, Age Loss: {avg_age}, Sex Loss: {avg_sex}, Combined Loss: {avg_loss}")
            print('#' * 50)
            args.current_epoch += 1

        # Save best checkpoint based on combined loss
        if args.rank == 0 and avg_loss < min_comb_loss:
            min_comb_loss = avg_loss
            print(f"New best model found at epoch {epoch} with combined loss {avg_loss:.4f}. Saving checkpoint.")
            save_model(args, model, optimizer, min_comb_loss, latest=False, best=True)
        
        # Save checkpoint every epoch by overwriting latest.pth
        if args.rank == 0:
            save_model(args, model, optimizer, min_comb_loss, latest=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    parser.add_argument('--data_file', type=str, default=None,
                    help='Path to .csv file containing list of training samples and metadata')
    parser.add_argument('--save_model_path', type=str, default=None,
                    help='Directory to save checkpoint_##.tar files')
    parser.add_argument('--age_weight', type=float, default=0.2,
                    help='Weight for age prediction auxiliary loss')
    parser.add_argument('--sex_weight', type=float, default=0.3,
                    help='Weight for sex prediction auxiliary loss')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Starting epoch for training')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Total number of epochs for training')
    parser.add_argument('--resnet', type=str, default='resnet18',
                        help='ResNet architecture to use (e.g., resnet18, resnet50)')
    parser.add_argument('--projection_dim', type=int, default=32,
                        help='Dimensionality of the projection head output')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature parameter for NT-Xent loss')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    parser.add_argument('--image_col', type=str, default='image_path',
                        help='Column name in CSV for image paths')
    parser.add_argument('--mask_col', type=str, default='mask_path',
                        help='Column name in CSV for mask paths')
    parser.add_argument('--age_col', type=str, default='age',
                        help='Column name in CSV for age')
    parser.add_argument('--sex_col', type=str, default='sex',
                        help='Column name in CSV for sex')
    parser.add_argument('--reload', action='store_true',
                    help='Whether to reload from latest checkpoint in save_model_path')
    
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "LOCAL_RANK not found in environment. "
            "This script should be launched with torchrun, e.g.\n"
            "  torchrun --nproc_per_node=4 --nnodes=2 ... main.py"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    main(local_rank, args)