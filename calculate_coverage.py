import os
import json
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ============================================================================
# 1. SETUP & CONFIGURATION
# ============================================================================
base_dir = r"/home/neel/d_drive/ai_data/data/data_to_ivy"
class_map_path = os.path.join(base_dir, "class_map.json")

# IMPORTANT: Choose which folder of masks you want to analyze!
# If you ran the rescue script, you might want to point this to "rescued_pseudo_masks"
# Otherwise, point it to your original "pseudo_masks" or AI predictions folder.
masks_dir = os.path.join(base_dir, "pseudo_masks") 

output_csv = "percent_cover_report.csv"

# ============================================================================
# 2. LOAD CLASSES
# ============================================================================
with open(class_map_path, 'r') as f:
    class_map = json.load(f)
    num_classes = max(class_map.values()) + 1
    
# Reverse mapping to get names: {0: 'background', 1: 'Rock_Deepwatercove', ...}
id_to_class = {v: k for k, v in class_map.items()}

# ============================================================================
# 3. CALCULATE PERCENT COVER LOOP
# ============================================================================
mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
if not mask_files:
    raise FileNotFoundError(f"No .png masks found in {masks_dir}")

print(f"\nAnalyzing Percent Cover for {len(mask_files)} images in '{masks_dir}'...")

# Lists to store our data
all_image_stats = []
global_pixel_counts = np.zeros(num_classes, dtype=np.int64)
global_total_valid_pixels = 0

for mask_path in tqdm(mask_files, desc="Calculating Coverage"):
    filename = os.path.basename(mask_path)
    
    # Load mask as numpy array
    mask_np = np.array(Image.open(mask_path))
    
    # Exclude "Ignore" pixels (like 255) from the total count
    valid_pixels = mask_np[mask_np < num_classes]
    total_valid = len(valid_pixels)
    
    if total_valid == 0:
        continue # Skip if the image has no valid pixels at all
        
    # Tally up the pixels for each class in this specific image
    counts = np.bincount(valid_pixels, minlength=num_classes)
    
    # Calculate percentage for this image
    percentages = (counts / total_valid) * 100
    
    # Add to our global tracker for the overall summary
    global_pixel_counts += counts
    global_total_valid_pixels += total_valid
    
    # Create a dictionary for this image's row in the CSV
    image_data = {"Image_Name": filename}
    for c in range(num_classes):
        class_name = id_to_class.get(c, f"Class_{c}")
        # Save both the raw pixel count and the % cover
        image_data[f"{class_name} (%)"] = round(percentages[c], 2)
        
    all_image_stats.append(image_data)

# ============================================================================
# 4. SAVE TO CSV & PRINT SUMMARY
# ============================================================================
# Create Pandas DataFrame and save to CSV
df = pd.DataFrame(all_image_stats)
df.to_csv(output_csv, index=False)

print(f"\n✅ Saved detailed breakdown to: {output_csv}")

# Print Global Summary
print("\n" + "="*60)
print("GLOBAL PERCENT COVER (ACROSS ENTIRE DATASET)")
print("="*60)

global_percentages = (global_pixel_counts / global_total_valid_pixels) * 100

for c in range(num_classes):
    class_name = id_to_class.get(c, f"Class_{c}")
    pct = global_percentages[c]
    print(f"  {class_name:<45}: {pct:>6.2f}%")
print("="*60 + "\n")