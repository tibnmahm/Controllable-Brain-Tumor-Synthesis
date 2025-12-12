# train_segmenter.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss
from tqdm import tqdm
import torch.amp  # Updated AMP import

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset: Load only high-quality slices ---
class RealTumorDataset(Dataset):
    def __init__(self, root="2d_data", max_samples=2500):
        print("Loading dataset into RAM...")
        image_files = sorted([
            os.path.join(root, "images", f) 
            for f in os.listdir(os.path.join(root, "images")) 
            if f.endswith(".npy")
        ])
        mask_files = sorted([
            os.path.join(root, "masks", f) 
            for f in os.listdir(os.path.join(root, "masks")) 
            if f.endswith(".npy")
        ])
        
        images = []
        masks = []
        for img_path, mask_path in zip(image_files, mask_files):
            mask = np.load(mask_path)
            # Keep all tumor slices + 10% healthy slices
            if mask.sum() > 0 or np.random.rand() < 0.1:
                images.append(np.load(img_path).astype(np.float32))
                masks.append(mask.astype(np.float32))
        
        # Randomly select up to max_samples
        if len(images) > max_samples:
            indices = np.random.choice(len(images), max_samples, replace=False)
            self.images = [images[i] for i in indices]
            self.masks = [masks[i] for i in indices]
        else:
            self.images = images
            self.masks = masks
        
        print(f"✅ Loaded {len(self.images)} high-quality samples into RAM.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx][None]  # (1, H, W)
        mask = self.masks[idx]        # (H, W)
        return torch.tensor(img), torch.tensor(mask)

# --- Model: Compact U-Net ---
segmenter = BasicUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    features=(32, 64, 128, 256, 512, 512),  # 6 values, reduced last layer
    dropout=0.0,
).to(device)

# --- Loss & Optimizer ---
dice_loss = DiceLoss(sigmoid=True)
optimizer = optim.Adam(segmenter.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cuda")  # Updated AMP

# --- Data ---
dataset = RealTumorDataset(max_samples=2500)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset, batch_size=4, shuffle=True,
    num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=4, shuffle=False,
    num_workers=4, pin_memory=True
)

# --- Training Loop ---
os.makedirs("segmenter_checkpoints", exist_ok=True)
best_val_loss = float('inf')

for epoch in range(20):
    segmenter.train()
    train_loss = 0
    for img, mask in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        with torch.amp.autocast("cuda"):  # Updated AMP
            pred = segmenter(img)
            loss = dice_loss(pred, mask.unsqueeze(1))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    # Validation
    segmenter.eval()
    val_loss = 0
    with torch.no_grad():
        for img, mask in val_loader:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                pred = segmenter(img)
                loss = dice_loss(pred, mask.unsqueeze(1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(segmenter.state_dict(), "segmenter.pth")
        print("✅ Saved best segmenter.pth")

print("🎉 Training complete! Ready for evaluation.")