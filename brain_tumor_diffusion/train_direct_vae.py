# train_direct_vae.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Reuse CustomVAE from train_vae.py ---
class CustomVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

# --- Dataset ---
class RealTumorDataset(Dataset):
    def __init__(self, root="2d_data"):
        self.image_files = sorted([os.path.join(root, "images", f) for f in os.listdir(os.path.join(root, "images")) if f.endswith(".npy")])
        self.mask_files = sorted([os.path.join(root, "masks", f) for f in os.listdir(os.path.join(root, "masks")) if f.endswith(".npy")])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = np.load(self.image_files[idx]).astype(np.float32)[None]  # (1, H, W)
        mask = np.load(self.mask_files[idx]).astype(np.float32)[None]  # (1, H, W)
        return torch.tensor(img), torch.tensor(mask)

# --- Model & Training Setup ---
vae = CustomVAE().to(device)
recon_loss = nn.MSELoss()
dice_loss = DiceLoss(sigmoid=True)
optimizer = optim.Adam(vae.parameters(), lr=1e-4)

dataset = RealTumorDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# --- Training ---
os.makedirs("direct_vae_samples", exist_ok=True)
for epoch in range(30):  # 30 epochs is sufficient for VAE
    epoch_loss = 0
    for img, mask in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        img, mask = img.to(device), mask.to(device)
        recon = vae(img)

        loss_img = recon_loss(recon, img)
        loss_mask = dice_loss(recon, mask)
        loss = loss_img + loss_mask

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.4f}")

    # --- Save samples every 5 epochs ---
    if (epoch + 1) % 5 == 0:
        vae.eval()
        with torch.no_grad():
            sample_img, sample_mask = next(iter(DataLoader(dataset, batch_size=4, shuffle=True)))
            sample_img = sample_img.to(device)
            recon_img = vae(sample_img)

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(sample_img[0, 0].cpu(), cmap='gray')
            axs[0].set_title('Input')
            axs[0].axis('off')
            axs[1].imshow(recon_img[0, 0].cpu(), cmap='gray')
            axs[1].set_title('Reconstruction')
            axs[1].axis('off')
            axs[2].imshow(sample_mask[0, 0], cmap='gray')
            axs[2].set_title('Tumor Mask')
            axs[2].axis('off')
            plt.tight_layout()
            plt.savefig(f"direct_vae_samples/epoch_{epoch+1}.png", dpi=150)
            plt.close()
        vae.train()

torch.save(vae.state_dict(), "direct_vae.pth")
print("✅ Direct VAE saved as direct_vae.pth")

# --- Generate final synthetic set (100 samples) ---
# For Direct VAE, "synthesis" = reconstruction of real images
# But to compare fairly, we'll generate from real images and save alongside mask
print("Generating final Direct VAE evaluation set...")
os.makedirs("synthetic_final_vae/images", exist_ok=True)
os.makedirs("synthetic_final_vae/masks", exist_ok=True)

sample_dataset = RealTumorDataset()
sample_loader = DataLoader(sample_dataset, batch_size=100, shuffle=True)
real_img, real_mask = next(iter(sample_loader))

with torch.no_grad():
    vae.eval()
    recon_img = vae(real_img.to(device))

for i in range(100):
    np.save(f"synthetic_final_vae/images/img_{i:03d}.npy", recon_img[i, 0].cpu().numpy())
    np.save(f"synthetic_final_vae/masks/mask_{i:03d}.npy", real_mask[i, 0].numpy())

print("✅ Final Direct VAE dataset saved to 'synthetic_final_vae/'")
print("📸 Use this folder as input to evaluate_synthetic.py")
