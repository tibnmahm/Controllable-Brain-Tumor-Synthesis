# train_vae_gan.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
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

# --- Dataset: Use real images + masks ---
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

# --- Generator: Latent Mask -> Latent Image -> Image ---
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 256, 3, stride=1, padding=1)  # ← Output 256 channels, no Sigmoid
        )

    def forward(self, x):
        return self.model(x)

# --- Discriminator: Latent Image + Latent Mask -> Real/Fake ---
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Input: (512, 16, 16) — latent image + latent mask
            nn.Conv2d(512, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# --- Model & Training Setup ---
vae = CustomVAE().to(device)
vae.load_state_dict(torch.load("vae.pth", map_location=device, weights_only=False))
vae.eval()

generator = Generator().to(device)
discriminator = Discriminator().to(device)

# --- Losses ---
bce_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

# --- Optimizers ---
g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# --- Data ---
dataset = RealTumorDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# --- Training ---
os.makedirs("vae_gan_samples", exist_ok=True)
loss_history_g = []
loss_history_d = []

for epoch in range(50):  # 50 epochs is usually enough
    g_epoch_loss = 0
    d_epoch_loss = 0
    for img, mask in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        img, mask = img.to(device), mask.to(device)

        with torch.no_grad():
            z_mask = vae.encode(mask)      # (B, 256, 16, 16)
            real_latent = vae.encode(img)  # (B, 256, 16, 16)

        real_labels = torch.ones((img.size(0), 1), device=device)
        fake_labels = torch.zeros((img.size(0), 1), device=device)

        # --- Train Discriminator ---
        fake_latent = generator(z_mask)  # ← Fresh forward pass
        fake_input = torch.cat([fake_latent, z_mask], dim=1)
        real_input = torch.cat([real_latent, z_mask], dim=1)

        d_real_loss = bce_loss(discriminator(real_input), real_labels)
        d_fake_loss = bce_loss(discriminator(fake_input), fake_labels)
        d_loss = d_real_loss + d_fake_loss

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # --- Train Generator (recompute fake_latent to avoid graph reuse) ---
        fake_latent = generator(z_mask)  # ← Re-run generator forward pass
        fake_input = torch.cat([fake_latent, z_mask], dim=1)

        g_adv_loss = bce_loss(discriminator(fake_input), real_labels)
        g_l1_loss = l1_loss(fake_latent, real_latent)
        g_loss = g_adv_loss + 100 * g_l1_loss

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        g_epoch_loss += g_loss.item()

    avg_g_loss = g_epoch_loss / len(dataloader)
    avg_d_loss = d_epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}, G Loss: {avg_g_loss:.6f}, D Loss: {avg_d_loss:.6f}")
    loss_history_g.append(avg_g_loss)
    loss_history_d.append(avg_d_loss)

    # --- Save samples every 5 epochs ---
    if (epoch + 1) % 5 == 0:
        _, real_mask = next(iter(dataloader))
        cond = real_mask[:4].to(device)
        with torch.no_grad():
            z_cond = vae.encode(cond)  # (4, 256, 16, 16)
            syn_latent = generator(z_cond)  # (4, 256, 16, 16)
            syn_img = vae.decode(syn_latent)  # (4, 1, 256, 256)

        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        for i in range(4):
            axs[0, i].imshow(cond[i, 0].cpu(), cmap='gray')
            axs[0, i].set_title(f'Condition Mask {i+1}')
            axs[0, i].axis('off')
            axs[1, i].imshow(syn_img[i, 0].cpu(), cmap='gray')
            axs[1, i].set_title(f'Synthetic Image {i+1}')
            axs[1, i].axis('off')
        plt.tight_layout()
        plt.savefig(f"vae_gan_samples/epoch_{epoch+1}.png", dpi=150)
        plt.close()

torch.save(generator.state_dict(), "generator_vae_gan.pth")
print("✅ VAE+GAN generator saved as generator_vae_gan.pth")

# --- Plot loss curves ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history_g, label='Generator Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator Loss Curve')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(loss_history_d, label='Discriminator Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Discriminator Loss Curve')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('vae_gan_loss_curve.png', dpi=150)
plt.close()
print("📊 VAE+GAN loss curves saved as 'vae_gan_loss_curve.png'")

# --- Final evaluation set (100 samples) ---
print("Generating final synthetic evaluation set...")
os.makedirs("synthetic_final_vae_gan/images", exist_ok=True)
os.makedirs("synthetic_final_vae_gan/masks", exist_ok=True)

_, final_masks = next(iter(DataLoader(RealTumorDataset(), batch_size=100, shuffle=True)))
cond_final = final_masks.to(device)
with torch.no_grad():
    z_cond_final = vae.encode(cond_final)
    syn_latent_final = generator(z_cond_final)
    img_final = vae.decode(syn_latent_final)

for i in range(100):
    np.save(f"synthetic_final_vae_gan/images/img_{i:03d}.npy", img_final[i, 0].cpu().numpy())
    np.save(f"synthetic_final_vae_gan/masks/mask_{i:03d}.npy", cond_final[i, 0].cpu().numpy())

print("✅ Final VAE+GAN synthetic dataset saved to 'synthetic_final_vae_gan/'")
print("📸 Use this folder as input to evaluate_synthetic.py")
