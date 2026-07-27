import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torchvision.transforms.functional as TF
import gc

from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. SETUP & CONFIGURATION
# ============================================================================
base_dir = r"/home/neel/d_drive/ai_data/data/data_to_ivy"
checkpoint_path = r"/home/neel/stage2_ckpts/benthic-ssl-epoch=44-ssl_loss=13.00.ckpt"
probe_weights = "best_ft_probe.pth" 
model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Full HD
eval_h, eval_w = 1088, 1920
patch_size = 16

# Folder for per-class heatmaps
output_dir = "per_class_heatmaps"
os.makedirs(output_dir, exist_ok=True)

color_palette = np.array([
    [0.0, 0.0, 0.0],       # 0: background (Black)
    [0.8, 0.2, 0.2],       # 1: Rock (Red)
    [0.2, 0.8, 0.2],       # 2: Carpophyllum (Green)
    [0.8, 0.8, 0.2],       # 3: Ecklonia (Yellow)
    [0.2, 0.2, 0.8],       # 4: Amphiroa (Blue)
    [0.8, 0.2, 0.8]        # 5: Anthothoe (Purple)
])

# ============================================================================
# 2. LOAD MODELS
# ============================================================================
class DenseLinearProbe(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
    def forward(self, x, target_size):
        logits = self.classifier(x)
        return F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)

print("Loading Backbone and Probe...")
backbone = AutoModel.from_pretrained(model_id, attn_implementation='sdpa', torch_dtype=torch.float32)
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
new_state_dict = {k.replace('module.', '').replace('_orig_mod.', '').replace('student.backbone.', '').replace('model.', ''): v for k, v in state_dict.items()}
backbone.load_state_dict(new_state_dict, strict=False)
backbone.eval().to(device)

with open(os.path.join(base_dir, "class_map.json"), 'r') as f:
    class_map = json.load(f)
    num_classes = max(class_map.values()) + 1
    id_to_class = {v: k for k, v in class_map.items()}

probe = DenseLinearProbe(backbone.config.hidden_size, num_classes).to(device)
probe.load_state_dict(torch.load(probe_weights, map_location=device))
probe.eval()

# ============================================================================
# 3. INFERENCE & PLOTTING LOOP
# ============================================================================
images_dir = os.path.join(base_dir, "raw_images")
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])

num_images_to_process = min(20, len(image_files))

for i in range(num_images_to_process):
    sample_img = image_files[i] 
    base_name = os.path.splitext(sample_img)[0]
    img_path = os.path.join(images_dir, sample_img)

    # Load and Resize
    orig_img = Image.open(img_path).convert("RGB")
    img_pil_eval = TF.resize(orig_img, [eval_h, eval_w], interpolation=TF.InterpolationMode.BILINEAR)
    img_np = np.array(img_pil_eval)
    
    img_input = TF.normalize(TF.to_tensor(img_pil_eval), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).unsqueeze(0).to(device)

    # --- Run Inference ---
    with torch.no_grad():
        out = backbone(pixel_values=img_input, output_hidden_states=True)
        patches = out.last_hidden_state[:, 1+getattr(backbone.config, 'num_register_tokens', 4):, :]
        features = patches.reshape(1, eval_h // patch_size, eval_w // patch_size, backbone.config.hidden_size).permute(0, 3, 1, 2)
        
        logits = probe(features, (eval_h, eval_w)) 
        
        # 1. Convert logits to percentages (0.0 to 1.0)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy() # Shape: [6, 1088, 1920]
        
        # 2. Hard prediction for the overlay
        prediction = np.argmax(probabilities, axis=0)

    # --- Prepare Overlay ---
    pred_colored = color_palette[prediction]
    alpha_channel = np.where(prediction == 0, 0.0, 0.6)
    rgba_prediction = np.dstack((pred_colored, alpha_channel))

    # --- Plotting (2 Rows x 4 Columns) ---
    fig, axes = plt.subplots(2, 4, figsize=(32, 12))
    axes = axes.flatten() # Flattens the 2x4 grid into a 1D list of 8 panels

    # Panel 0: Raw Image
    axes[0].imshow(img_np)
    axes[0].set_title(f"Raw Input\n{sample_img}", fontsize=14)
    axes[0].axis('off')

    # Panel 1: Prediction Overlay
    axes[1].imshow(img_np)
    axes[1].imshow(rgba_prediction)
    axes[1].set_title("Winning AI Prediction", fontsize=14)
    axes[1].axis('off')
    patches_legend = [mpatches.Patch(color=color_palette[c], label=id_to_class.get(c, f"Class {c}")) for c in range(num_classes)]
    axes[1].legend(handles=patches_legend, bbox_to_anchor=(0.0, -0.05), loc='upper left', ncol=2)

    # Panels 2-7: Individual Class Confidence Heatmaps
    for c in range(num_classes):
        ax = axes[c + 2] # Shift by 2 because 0 and 1 are the raw/overlay images
        
        class_name = id_to_class.get(c, f"Class {c}")
        # Plot the specific class probability (0.0 to 1.0)
        heatmap = ax.imshow(probabilities[c], cmap='inferno', vmin=0.0, vmax=1.0)
        
        # Add a title matching the class color
        ax.set_title(f"{class_name} Confidence", fontsize=14, color=color_palette[c] if c != 0 else 'black')
        ax.axis('off')
        
        # Add colorbars to the right edge of each heatmap
        cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    
    # --- Save & Aggressively Clean RAM ---
    save_path = os.path.join(output_dir, f"perclass_{base_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    plt.clf()
    plt.cla()
    plt.close('all')
    del img_np, rgba_prediction, probabilities, prediction, logits, features, fig, axes
    gc.collect() 

    print(f"[{i+1}/{num_images_to_process}] Saved: {save_path}")

print("\n✅ Per-Class Confidence extraction finished!")