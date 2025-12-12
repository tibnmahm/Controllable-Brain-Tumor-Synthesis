# train_diffusion.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from monai.networks.nets import DiffusionModelUNet
from tqdm import tqdm
import torch.nn.functional as F
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

# --- Dataset: Safe CPU return ---
class LatentDataset(Dataset):
    def __init__(self, vae_path="vae.pth", data_root="2d_data"):
        self.image_files = sorted([os.path.join(data_root, "images", f) for f in os.listdir(os.path.join(data_root, "images")) if f.endswith(".npy")])
        self.mask_files = sorted([os.path.join(data_root, "masks", f) for f in os.listdir(os.path.join(data_root, "masks")) if f.endswith(".npy")])

        self.vae = CustomVAE().to(device)
        self.vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=False))
        self.vae.eval()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = torch.tensor(np.load(self.image_files[idx]).astype(np.float32))[None, None]
        mask = torch.tensor(np.load(self.mask_files[idx]).astype(np.float32))[None, None]

        with torch.no_grad():
            img = img.to(device)
            z = self.vae.encode(img)  # (1, 256, 16, 16)
            mask_low = F.interpolate(mask.to(device), size=(16, 16), mode="nearest")
            # Return CPU tensors for DataLoader safety
            z = z.cpu()
            mask_low = mask_low.cpu()
        return z.squeeze(0), mask_low.squeeze(0)

# --- Model & Training Setup ---
unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=257,
    out_channels=256,
    channels=(64, 128, 256),
    attention_levels=(False, True, True),
    num_res_blocks=2,
).to(device)

num_steps = 1000
betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
alphas_cumprod = torch.cumprod(1 - betas, dim=0)

dataset = LatentDataset()
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
optimizer = optim.Adam(unet.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)

def ddpm_sample(model, cond, shape, steps=50):
    model.eval()
    z = torch.randn(shape, device=device)
    with torch.no_grad():
        for t in reversed(range(steps)):
            t_batch = torch.full((shape[0],), t, device=device)
            alpha_t = alphas_cumprod[t].sqrt()
            sigma_t = (1 - alphas_cumprod[t]).sqrt()
            input_cat = torch.cat([z, cond], dim=1)
            noise_pred = model(input_cat, timesteps=t_batch.float())
            z = (z - sigma_t * noise_pred) / alpha_t
    return z

vae = CustomVAE().to(device)
vae.load_state_dict(torch.load("vae.pth", map_location=device, weights_only=False))
vae.eval()

print("🔍 Checking shapes from dataloader...")
for z, mask in dataloader:
    print("Latent shape:", z.shape)   # Expected: (B, 256, 16, 16)
    print("Mask shape:", mask.shape)  # Expected: (B, 1, 16, 16)
    break
print("✅ Shapes verified. Starting training...\n")

# --- Training ---
os.makedirs("diffusion_samples", exist_ok=True)
unet.train()
loss_history = []
for epoch in range(400):
    epoch_loss = 0
    for z, mask in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        z, mask = z.to(device), mask.to(device)

        t = torch.randint(0, num_steps, (z.shape[0],), device=device)
        noise = torch.randn_like(z)
        alpha_t = alphas_cumprod[t].view(-1,1,1,1).sqrt()
        sigma_t = (1 - alphas_cumprod[t]).view(-1,1,1,1).sqrt()
        z_t = alpha_t * z + sigma_t * noise

        model_input = torch.cat([z_t, mask], dim=1)
        noise_pred = unet(model_input, timesteps=t.float())
        loss = nn.MSELoss()(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    # print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.6f}")
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")
    loss_history.append(avg_loss)
    scheduler.step()

    # --- Save visualizations + .npy files ---
    if (epoch + 1) % 5 == 0:
        _, real_mask = next(iter(dataloader))
        cond = real_mask[:4].to(device)
        z_gen = ddpm_sample(unet, cond, shape=(4, 256, 16, 16), steps=50)

        with torch.no_grad():
            syn_img = vae.decode(z_gen)

        # Visual plot
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        for i in range(4):
            axs[0, i].imshow(cond[i, 0].cpu(), cmap='gray')
            axs[0, i].set_title(f'Condition Mask {i+1}')
            axs[0, i].axis('off')
            axs[1, i].imshow(syn_img[i, 0].cpu(), cmap='gray')
            axs[1, i].set_title(f'Synthetic Image {i+1}')
            axs[1, i].axis('off')
        plt.tight_layout()
        plt.savefig(f"diffusion_samples/epoch_{epoch+1}.png", dpi=600)
        plt.close()

        # Save .npy for evaluation
        os.makedirs("synthetic_eval/images", exist_ok=True)
        os.makedirs("synthetic_eval/masks", exist_ok=True)
        cond_full = F.interpolate(cond, size=(256, 256), mode="nearest")
        for i in range(4):
            np.save(f"synthetic_eval/images/img_e{epoch+1}_s{i}.npy", syn_img[i, 0].cpu().numpy())
            np.save(f"synthetic_eval/masks/mask_e{epoch+1}_s{i}.npy", cond_full[i, 0].cpu().numpy())

torch.save(unet.state_dict(), "diffusion.pth")
print("✅ Diffusion model saved as diffusion.pth")

# --- Plot and save loss curve ---
plt.figure(figsize=(10, 6))
plt.plot(loss_history, color='blue', linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Diffusion Model Training Loss Curve (400 Epochs)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=600)
plt.close()
print("📊 Training loss curve saved as 'loss_curve.png'")

# --- Final evaluation set (100 samples) ---
print("Generating final synthetic evaluation set...")
os.makedirs("synthetic_final_vae_diffusion/images", exist_ok=True)
os.makedirs("synthetic_final_vae_diffusio/masks", exist_ok=True)

_, final_masks = next(iter(DataLoader(LatentDataset(), batch_size=100, shuffle=True)))
cond_final = final_masks.to(device)
z_final = ddpm_sample(unet, cond_final, shape=(100, 256, 16, 16), steps=50)

with torch.no_grad():
    img_final = vae.decode(z_final)
mask_final = F.interpolate(cond_final, size=(256, 256), mode="nearest")

for i in range(100):
    np.save(f"synthetic_final_vae_diffusio/images/img_{i:03d}.npy", img_final[i, 0].cpu().numpy())
    np.save(f"synthetic_final_vae_diffusio/masks/mask_{i:03d}.npy", mask_final[i, 0].cpu().numpy())

print("✅ Final synthetic dataset saved to 'synthetic_final_vae_diffusio/'")
print("📸 Use this folder as input to evaluate_synthetic.py")