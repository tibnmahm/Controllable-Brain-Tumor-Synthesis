# evaluate_synthetic.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance, compute_surface_dice
from monai.networks.nets import BasicUNet
from glob import glob
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CONFIG: Change this to evaluate different methods ---
SYNTHETIC_DIR = "synthetic_final_gan"  # e.g., "synthetic_final", "synthetic_final_vae"
OUTPUT_METRICS_FILE = f"metrics_{os.path.basename(SYNTHETIC_DIR)}.txt"

# --- Dataset: Load .npy files as 4D tensors ---
class SyntheticDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_files = sorted(glob(os.path.join(img_dir, "img_*.npy")))
        self.mask_files = sorted(glob(os.path.join(mask_dir, "mask_*.npy")))
        assert len(self.img_files) == len(self.mask_files), "Mismatch in image/mask counts!"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]
        
        # Load raw arrays
        img_raw = np.load(img_path).astype(np.float32)
        mask_raw = np.load(mask_path).astype(np.float32)
        
        print(f"Loaded img shape: {img_raw.shape}, mask shape: {mask_raw.shape} from {img_path}")
        
        # Force to 2D by taking last two dimensions (H, W)
        if img_raw.ndim > 2:
            img = img_raw.reshape(-1, img_raw.shape[-2], img_raw.shape[-1])[-1]  # take last channel/slice
        else:
            img = img_raw
            
        if mask_raw.ndim > 2:
            mask = mask_raw.reshape(-1, mask_raw.shape[-2], mask_raw.shape[-1])[-1]
        else:
            mask = mask_raw

        # Now img and mask are (256, 256)
        assert img.shape == (256, 256), f"Final img shape: {img.shape}"
        assert mask.shape == (256, 256), f"Final mask shape: {mask.shape}"
        
        # Return as (1, 1, 256, 256) and (256, 256)
        return torch.tensor(img), torch.tensor(mask)

# --- Load Pre-trained Segmenter (MUST match train_segmenter.py) ---
segmenter = BasicUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    features=(32, 64, 128, 256, 512, 512),  # 6 values — critical!
    dropout=0.0,
).to(device)
segmenter.load_state_dict(torch.load("segmenter.pth", map_location=device, weights_only=False))
segmenter.eval()

# --- Evaluation ---
def evaluate():
    dataset = SyntheticDataset(
        img_dir=os.path.join(SYNTHETIC_DIR, "images"),
        mask_dir=os.path.join(SYNTHETIC_DIR, "masks")
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    hd95_list, assd_list, nsd_list = [], [], []

    for img, cond_mask in tqdm(dataloader, desc="Evaluating"):
        img = img.unsqueeze(1).to(device)          
        cond_mask = cond_mask.to(device) 

        with torch.no_grad():
            pred_logit = segmenter(img)
            pred_logit = torch.nn.functional.interpolate(pred_logit, size=(256, 256), mode="bilinear", align_corners=False)
            pred_mask = (torch.sigmoid(pred_logit) > 0.5).float()  # (1,1,256,256)

        # Reshape for MONAI metrics: (B, C, H, W)
        # pred_mask = pred_mask.squeeze(0)       # (1,256,256)
        cond_mask = cond_mask.unsqueeze(1)     # (1,256,256)

        # Compute metrics
        try:
            hd95 = compute_hausdorff_distance(pred_mask, cond_mask, percentile=95)
            assd = compute_average_surface_distance(pred_mask, cond_mask)
            nsd = compute_surface_dice(pred_mask, cond_mask, [1.0])  # 1mm tolerance
        except Exception as e:
            print(f"Skipping sample due to metric error: {e}")
            continue

        if not torch.isnan(hd95):
            hd95_list.append(hd95.item())
        if not torch.isnan(assd):
            assd_list.append(assd.item())
        if not torch.isnan(nsd):
            nsd_list.append(nsd.item())

    # Compute and print results
    results = [
        f"Method: {SYNTHETIC_DIR}",
        f"Num samples: {len(hd95_list)}",
        f"Mean HD95: {np.mean(hd95_list):.2f} mm",
        f"Mean ASSD: {np.mean(assd_list):.2f} mm",
        f"Mean NSD:  {np.mean(nsd_list):.2%}",
        "",
        f"HD95 (all): {hd95_list}",
        f"ASSD (all): {assd_list}",
        f"NSD (all):  {nsd_list}",
    ]

    print("\n".join(results))
    with open(OUTPUT_METRICS_FILE, "w") as f:
        f.write("\n".join(results))

    # Save arrays for plotting
    np.save(f"hd95_{os.path.basename(SYNTHETIC_DIR)}.npy", hd95_list)
    np.save(f"assd_{os.path.basename(SYNTHETIC_DIR)}.npy", assd_list)
    np.save(f"nsd_{os.path.basename(SYNTHETIC_DIR)}.npy", nsd_list)

if __name__ == "__main__":
    evaluate()
    print(f"✅ Evaluation complete. Metrics saved to {OUTPUT_METRICS_FILE}")