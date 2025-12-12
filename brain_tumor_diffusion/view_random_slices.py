# view_random_slices.py
import os
import numpy as np
import matplotlib.pyplot as plt
import random

# --- Settings ---
data_root = "2d_data"
num_samples = 4  # number of random slices to show
seed = 14

# --- Get file lists ---
image_files = sorted([
    os.path.join(data_root, "images", f)
    for f in os.listdir(os.path.join(data_root, "images"))
    if f.endswith(".npy")
])
mask_files = sorted([
    os.path.join(data_root, "masks", f)
    for f in os.listdir(os.path.join(data_root, "masks"))
    if f.endswith(".npy")
])

assert len(image_files) == len(mask_files), "Mismatch in image/mask counts!"
print(f"Found {len(image_files)} slices.")

# --- Pick random indices ---
random.seed(seed)
indices = random.sample(range(len(image_files)), num_samples)

# --- Plot ---
fig, axs = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
for i, idx in enumerate(indices):
    img = np.load(image_files[idx])
    mask = np.load(mask_files[idx])

    # Image
    axs[0, i].imshow(img, cmap='gray')
    axs[0, i].set_title(f"Image {idx}")
    axs[0, i].axis('off')

    # Mask
    axs[1, i].imshow(mask, cmap='gray')
    axs[1, i].set_title(f"Mask {idx}")
    axs[1, i].axis('off')

plt.tight_layout()
plt.savefig("sample_slices.png", dpi=150)
plt.show()
print("✅ Saved sample_slices.png")
