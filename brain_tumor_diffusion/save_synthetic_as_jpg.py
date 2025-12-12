# save_synthetic_as_jpg.py
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# --- Directories ---
# input_img_dir = "synthetic_eval/images"      # or "synthetic_final/images"
# input_mask_dir = "synthetic_eval/masks"      # or "synthetic_final/masks"
# output_dir = "synthetic_jpg"
input_img_dir = "synthetic_final/images"      # or "synthetic_final/images"
input_mask_dir = "synthetic_final/masks"      # or "synthetic_final/masks"
output_dir = "synthetic_final_vae_diffusion_jpg"

os.makedirs(output_dir, exist_ok=True)

# --- Get all .npy files ---
img_files = sorted(glob(os.path.join(input_img_dir, "*.npy")))
mask_files = sorted(glob(os.path.join(input_mask_dir, "*.npy")))

assert len(img_files) == len(mask_files), "Mismatch in image/mask counts!"

# --- Convert to JPG ---
for img_path, mask_path in zip(img_files, mask_files):
    # Load
    img = np.load(img_path)      # (256, 256), range [0, 1]
    mask = np.load(mask_path)    # (256, 256), binary

    # Normalize to [0, 255] for JPG
    img_jpg = (img * 255).astype(np.uint8)
    mask_jpg = (mask * 255).astype(np.uint8)

    # Save as JPG
    base_name = os.path.basename(img_path).replace(".npy", "")
    plt.imsave(os.path.join(output_dir, f"{base_name}_img.jpg"), img_jpg, cmap='gray')
    plt.imsave(os.path.join(output_dir, f"{base_name}_mask.jpg"), mask_jpg, cmap='gray')

print(f"✅ Saved {len(img_files)} image-mask pairs as JPGs in '{output_dir}/'")
