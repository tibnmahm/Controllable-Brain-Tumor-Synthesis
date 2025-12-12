# extract_2d.py
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.transforms import (
    ScaleIntensityRanged,
    ResizeWithPadOrCropd,
    Spacingd,
    EnsureChannelFirstd,
    Compose
)
import torch

# --- Paths ---
data_root = "Task01_BrainTumour"
images_dir = os.path.join(data_root, "imagesTr")
labels_dir = os.path.join(data_root, "labelsTr")
output_dir = "2d_data"
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

# --- Get file lists ---
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".nii.gz")])
label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")])

# --- Transforms for 2D slice ---
# Note: We apply spacing + resize AFTER loading full volume
def process_volume(img_path, lbl_path):
    # Load 4D image (H, W, D, 4) and 3D label (H, W, D)
    img_nii = nib.load(img_path)
    lbl_nii = nib.load(lbl_path)

    img = img_nii.get_fdata()  # (H, W, D, 4)
    lbl = lbl_nii.get_fdata()  # (H, W, D)

    # Reorder to (4, H, W, D)
    img = np.transpose(img, (3, 0, 1, 2))
    flair = img[3]  # (H, W, D)

    # Binary tumor mask
    tumor_mask = (lbl > 0).astype(np.float32)

    # Add channel dim: (1, H, W, D)
    data = {
        "image": flair[None],
        "label": tumor_mask[None],
        "image_meta_dict": {"affine": img_nii.affine},
        "label_meta_dict": {"affine": lbl_nii.affine},
    }

    # Apply transforms
    try:
        tfm = Compose([
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(256, 256, -1)),
            ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True)
        ])
        out = tfm(data)
    except Exception as e:
        print(f"Skipping {img_path} due to error: {e}")
        return []

    img_proc = out["image"][0]  # (256, 256, D)
    mask_proc = out["label"][0]  # (256, 256, D)

    slices = []
    D = img_proc.shape[-1]
    start, end = max(0, D//2 - 40), min(D, D//2 + 40)
    for z in range(start, end):
        img_slice = img_proc[..., z]
        mask_slice = mask_proc[..., z]
        if mask_slice.sum() == 0 and np.random.rand() > 0.1:
            continue
        slices.append((img_slice, mask_slice))
    return slices

# --- Extract all ---
slice_count = 0
for img_file, lbl_file in tqdm(zip(image_files, label_files), total=len(image_files)):
    img_path = os.path.join(images_dir, img_file)
    lbl_path = os.path.join(labels_dir, lbl_file)
    slices = process_volume(img_path, lbl_path)
    for img_slice, mask_slice in slices:
        np.save(os.path.join(output_dir, "images", f"img_{slice_count:06d}.npy"), img_slice.astype(np.float32))
        np.save(os.path.join(output_dir, "masks", f"mask_{slice_count:06d}.npy"), mask_slice.astype(np.float32))
        slice_count += 1

print(f"✅ Extracted {slice_count} 2D slices into {output_dir}/")
