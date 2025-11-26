import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from dataset import JawBoneSegmentationDataset
from unet_model import UNet2D


def dice_loss(pred, target, smooth=1.0):
    # pred: probabilities (after sigmoid), target: {0,1}
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(imgs)
        probs = torch.sigmoid(logits)

        bce = F.binary_cross_entropy(probs, masks)
        dsc = dice_loss(probs, masks)
        loss = 0.5 * bce + 0.5 * dsc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(imgs)
        probs = torch.sigmoid(logits)

        bce = F.binary_cross_entropy(probs, masks)
        dsc = dice_loss(probs, masks)
        loss = 0.5 * bce + 0.5 * dsc

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="../Dataset")
    parser.add_argument("--csv_path", type=str, default="../Dataset/metadata_labeled.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = JawBoneSegmentationDataset(
        csv_path=args.csv_path,
        root_dir=args.dataset_root,
        transform=None,
    )

    # Handle the case where there are no labeled samples yet
    if len(dataset) == 0:
        raise RuntimeError(
            "No labeled samples found in metadata_labeled.csv. "
            "Add at least one labeled frame before training."
        )

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = UNet2D(in_channels=1, out_channels=1, base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_path = os.path.join(args.out_dir, "unet_best.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = eval_one_epoch(model, val_loader, device)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved new best model to {best_path}")


if __name__ == "__main__":
    main()
