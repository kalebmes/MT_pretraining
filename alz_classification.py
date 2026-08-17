# Fine-tuning/Linear-probing script adapted from Kaczmarek et al. (2025) 3D T1 SimCLR implementation. https://github.com/emilykaczmarek/3D-Neuro-SimCLR/
# Modifications: added early fusion and late fusion of age and sex features with the encoder.

import os
import csv
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

from simclr.modules import get_resnet
from simclr.modules.identity import Identity
from mri_dataset.loaders import make_downstream_loaders

# for reproducibility
def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_metrics_to_csv(csv_path, epoch, train_loss, val_loss, test_loss, val_metrics, test_metrics):
    metric_names = ["acc", "auc"]
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "val_loss", "test_loss",
            *[f"val_{k}" for k in metric_names],
            *[f"test_{k}" for k in metric_names],
        ])
        if not file_exists:
            writer.writeheader()
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
        }
        for m in metric_names:
            row[f"val_{m}"] = val_metrics.get(m, float("nan"))
            row[f"test_{m}"] = test_metrics.get(m, float("nan"))
        writer.writerow(row)

def save_ckpt(path, model, optimizer, epoch, best_metric=None):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )

def init_model_and_optimizer(
    model,
    device,
    lr, 
    finetune,
    pretrained_ckpt: str | None = None,
):
    start_epoch = 0
    
    # Load pretrained weights if provided. This corresponds to fine-tuning or linear probing from a pretrained encoder.
    if pretrained_ckpt and os.path.exists(pretrained_ckpt):
        ckpt = torch.load(pretrained_ckpt, map_location=device)
        # Remove DDP prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in ckpt['model_state_dict'].items()}
        
        # extract only encoder weights, ignore head weights
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                encoder_state_dict[k] = v
        
        model.load_state_dict(encoder_state_dict, strict=False)
        start_epoch = 0
        print(f"Loaded encoder pretrained weights from: {pretrained_ckpt}")

    # if no pretrained weights are provided, we will train from scratch. This is the baseline supervised training mode.
    # early fusion (channel-aware) or late fusion (encoder-aware) of age and sex features are also handled here.
    else:
        print("No checkpoint found. Starting from scratch.")
        if args.age_sex_channel_aware:
            print('[EARLY FUSION] Adding Sex and Age as additional channels to the encoder input.')
        elif args.age_sex_encoder_aware:
            print('[LATE FUSION] Concatenating Age and Sex Features to the encoder output before the head.')
    
    if not finetune:
        for p in model.encoder.parameters():
            p.requires_grad = False
        model.encoder.eval()  # set encoder to eval mode for linear probing
        print('Linear probing mode: encoder frozen, only head will be trained.')

    if finetune:
        if not pretrained_ckpt:
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Here we assign a different learning rate for the encoder and for the head 
        # Because, during finetuning, we want a smaller lr for the encoder
        else:    
            optimizer = torch.optim.Adam([
                {"params": model.encoder.parameters(), "lr": lr * 0.1},
                {"params": model.head.parameters(), "lr": lr}
            ], weight_decay=1e-4)
    
    # if linear probing, encoder is frozen, so we only optimize the head parameters
    else:
        optimizer = torch.optim.Adam(model.head.parameters(), lr=lr, weight_decay=1e-4)

    return model, optimizer, start_epoch

class LinearHeadModel(nn.Module):
    def __init__(self, encoder, n_features, num_classes=1, age_sex_channel_aware=False, age_sex_encoder_aware=False):
        super().__init__()
        self.encoder = encoder
        self.age_sex_channel_aware = age_sex_channel_aware
        self.age_sex_encoder_aware = age_sex_encoder_aware
        self.head = nn.Linear(n_features, num_classes)
        if self.age_sex_channel_aware:
            # Replace the first convolutional layer to accommodate the additional channels
            old_conv = self.encoder.conv1
            self.encoder.conv1 = nn.Conv3d(
                in_channels=3, # 1 (MRI) + 2 (age and sex as additional channels)
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
        
        elif self.age_sex_encoder_aware:
            self.age_sex_encoder = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, n_features),
            )
            self.head = nn.Linear(n_features * 2, num_classes)
        
    def forward(self, x, age=None, sex=None):
        if self.age_sex_channel_aware:
            # x: (B, 1, D, H, W)
            # age, sex: (B,)
            B, _, D, H, W = x.shape
            
            # tile age and sex to match x spatial dimensions
            a = age.view(B, 1, 1, 1, 1).expand(-1, -1, D, H, W)
            s = sex.view(B, 1, 1, 1, 1).expand(-1, -1, D, H, W)
            
            # then concatenate them along the channel dimension with the original image: (B, 3, D, H, W)
            x_combined = torch.cat([x, a, s], dim=1)
            
            x_out = self.encoder(x_combined)
            return self.head(x_out)
        
        elif self.age_sex_encoder_aware:
            x = self.encoder(x)  # (B, n_features)
            a = age.view(age.size(0), 1)  # (B, 1)
            s = sex.view(sex.size(0), 1)  # (B, 1)
            age_sex_concat = torch.cat([a, s], dim=1)  # (B, 2)
            age_sex_out = self.age_sex_encoder(age_sex_concat) # (B, n_features)
            final_concat = torch.cat([x, age_sex_out], dim=1)  # (B, n_features * 2)
            return self.head(final_concat)
        
        else:
            x = self.encoder(x)
            return self.head(x)


def train(model, loader, criterion, optimizer, device, args):
    model.train()
    total_loss = 0
    nan_count = 0

    for step, batch in enumerate(loader):
        x, y = batch['MRI'].to(device), batch['task_label'].to(device)
        age, sex = batch['age'].to(device), batch['sex'].to(device)
        
        y = y.float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        out = model(x, age, sex)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step [{step}/{len(loader)}] Train Loss: {loss.item():.4f}")

        total_loss += loss.item()
    
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN losses in this epoch!")
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, args):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            x, y = batch['MRI'].to(device), batch['task_label'].to(device)
            age, sex = batch['age'].to(device), batch['sex'].to(device)
            
            y = y.float().unsqueeze(1)

            output = model(x, age, sex)
            loss = criterion(output, y)
            total_loss += loss.item()

            all_preds.append(output.detach().cpu())
            all_labels.append(y.detach().cpu())

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    metrics = {}

    probs = torch.sigmoid(preds).squeeze(1).numpy()
    y_true = labels.squeeze(1).numpy()
    y_hat = (probs > 0.5).astype(np.int64)
    metrics["acc"] = float((y_hat == y_true).mean())
    try:
        metrics["auc"] = float(roc_auc_score(y_true, probs))
    except Exception:
        metrics["auc"] = float("nan")

    return total_loss / max(1,len(loader)), metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    seed_everything(args.seed)

    categories = args.categories.split("_")
    label_mapping = {cat: idx for idx, cat in enumerate(categories)}
    
    train_loader, val_loader, test_loader = make_downstream_loaders(
        train_csv=args.train_file,
        val_csv=args.val_file,
        test_csv=args.test_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_col=args.label_col,
        label_mapping = label_mapping,
    )

    encoder = get_resnet("resnet18", pretrained=False)
    n_features = encoder.fc.in_features 
    encoder.fc = Identity()
    model = LinearHeadModel(encoder, 
                            n_features, 
                            age_sex_channel_aware=args.age_sex_channel_aware,
                            age_sex_encoder_aware=args.age_sex_encoder_aware,
                            ).to(device)
    is_finetune = args.lp_or_ft == 'ft'
    # Create paths and csv for checkpointing latest and best fine-tuned/linear probed models
    eval_dir = os.path.join(args.save_path, "evaluation")
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    ckpt_path = os.path.join(eval_dir, f"{args.seed}_ft_{is_finetune}_latest.pth.tar")
    csv_path =  os.path.join(eval_dir, f"{args.seed}_ft_{is_finetune}_metrics.csv")
    best_path = os.path.join(eval_dir, f"{args.seed}_ft_{is_finetune}_best.pth.tar")
    pretrained_path = args.simclr_ckpt

    model, optimizer, start_epoch = init_model_and_optimizer(model=model, 
                                                                               device=device, 
                                                                               lr=args.lr, 
                                                                               finetune=is_finetune, 
                                                                               pretrained_ckpt=pretrained_path)
    

    criterion = nn.BCEWithLogitsLoss()
    best_metric = float("inf")  # we will use validation loss as the main metric, since it's more stable than metrics like AUC in low-data regimes

    for epoch in range(start_epoch, args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, args)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, args)
        test_loss, test_metrics = evaluate(model, test_loader, criterion, device, args)

        # Log metrics per epoch
        log_metrics_to_csv(csv_path, epoch, train_loss, val_loss, test_loss, val_metrics, test_metrics)        
        print(f"Epoch {epoch+1}/{args.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | test loss {test_loss:.4f} | val metrics {val_metrics} | test metrics {test_metrics}")
        
        if val_loss < best_metric:
            best_metric = val_loss
            save_ckpt(best_path, model, optimizer, epoch + 1, best_metric=best_metric)
        
        # Save latest ckpt
        save_ckpt(ckpt_path, model, optimizer, epoch + 1, best_metric=best_metric)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--categories', type=str, default=None, choices=['AD_MCI', 'MCI_CN'], help='Categories for the classification task.')
    parser.add_argument('--train_file', type=str, default=None,)
    parser.add_argument('--val_file', type=str, default=None,)
    parser.add_argument('--test_file', type=str, default=None,)
    parser.add_argument('--save_path', type=str, default=None,
                    help='Directory to save checkpoint_##.tar files')
    parser.add_argument('--simclr_ckpt', type=str, default=None,
                    help='Directory to load pretrained SSL checkpoint')
    parser.add_argument('--age_sex_channel_aware', action='store_true',)
    parser.add_argument('--age_sex_encoder_aware', action='store_true',)
    parser.add_argument('--lp_or_ft', type=str, default='ft', choices=['lp', 'ft'],)
    parser.add_argument('--seed', type=int, default=0, help='Initialization seed)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=100, help='Total number of epochs for training')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--label_col', type=str, default='Group', help='Column name in CSV for labels')
    parser.add_argument('--image_col', type=str, default='image_path', help='Column name in CSV for image paths')
    parser.add_argument('--mask_col', type=str, default='mask_path', help='Column name in CSV for mask paths')
    parser.add_argument('--age_col', type=str, default='Age', help='Column name in CSV for age')
    parser.add_argument('--sex_col', type=str, default='Sex', help='Column name in CSV for sex')
    
    args = parser.parse_args()
    main(args)
