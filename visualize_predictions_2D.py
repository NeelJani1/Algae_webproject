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

# Paths to BOTH of the saved probes from your training run!
probe_ft_weights = "best_ft_probe_2D.pth" 
probe_base_weights = "best_base_probe_2D.pth" 

model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Full HD Resolution (Must be divisible by 16!)
# 1088 / 16 = 68 | 1920 / 16 = 120
eval_h = 1088
eval_w = 1920
patch_size = 16

# Folder for the comparison images
output_dir = "model_comparisons_HD"
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
# --- FIX: Updated to match your training script architecture ---
class SpatialDecoderProbe(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    def forward(self, x, target_size):
        logits = self.decoder(x)
        return F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)

print("Loading Fine-Tuned Backbone...")
backbone_ft = AutoModel.from_pretrained(model_id, attn_implementation='sdpa', torch_dtype=torch.float32)
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

new_state_dict = {}
for k, v in state_dict.items():
    k = k.replace('module.', '').replace('_orig_mod.', '')
    if k.startswith('model.'): k = k[6:]
    if k.startswith('student.backbone.'): k = k.replace('student.backbone.', '')
    if k.startswith('backbone.'): k = k.replace('backbone.', '')
    
    # Remap block naming to Hugging Face encoder naming
    if k.startswith('blocks.'): k = k.replace('blocks.', 'encoder.layer.')
    if k.startswith('norm.'): k = k.replace('norm.', 'layernorm.')
        
    new_state_dict[k] = v

incompatible = backbone_ft.load_state_dict(new_state_dict, strict=False)
print(f"✅ Loaded Fine-Tuned Backbone. (Missing expected keys: {len(incompatible.missing_keys)})")
backbone_ft.eval().to(device)

print("Loading Baseline Backbone...")
backbone_base = AutoModel.from_pretrained(model_id, attn_implementation='sdpa', torch_dtype=torch.float32)
backbone_base.eval().to(device)

print("Loading Both Trained Probes...")
with open(os.path.join(base_dir, "class_map.json"), 'r') as f:
    class_map = json.load(f)
    num_classes = max(class_map.values()) + 1
    id_to_class = {v: k for k, v in class_map.items()}

# --- FIX: Instantiate SpatialDecoderProbe instead of DenseLinearProbe ---
probe_ft = SpatialDecoderProbe(backbone_ft.config.hidden_size, num_classes).to(device)
probe_ft.load_state_dict(torch.load(probe_ft_weights, map_location=device))
probe_ft.eval()

probe_base = SpatialDecoderProbe(backbone_base.config.hidden_size, num_classes).to(device)
probe_base.load_state_dict(torch.load(probe_base_weights, map_location=device))
probe_base.eval()

# ============================================================================
# 3. BATCH INFERENCE & VISUALIZATION LOOP
# ============================================================================
images_dir = os.path.join(base_dir, "raw_images")
masks_dir = os.path.join(base_dir, "pseudo_masks")

image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
num_images_to_process = min(400, len(image_files))

print(f"\nProcessing {num_images_to_process} images. Saving to '{output_dir}/'...")

for i in range(num_images_to_process):
    sample_img = image_files[i] 
    img_path = os.path.join(images_dir, sample_img)
    base_name = os.path.splitext(sample_img)[0]
    mask_path = os.path.join(masks_dir, base_name + '.png')
    
    if not os.path.exists(mask_path):
        continue

    # Load images
    orig_img = Image.open(img_path).convert("RGB")
    orig_mask = Image.open(mask_path).convert("L") # Ensure mask is single channel!

    # Transform (Resize to 1920x1088 rectangle)
    img_pil_eval = TF.resize(orig_img, [eval_h, eval_w], interpolation=TF.InterpolationMode.BILINEAR)
    img_input = TF.normalize(TF.to_tensor(img_pil_eval), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).unsqueeze(0).to(device)

    # Process ground truth mask
    mask_tensor = TF.resize(orig_mask, [eval_h, eval_w], interpolation=TF.InterpolationMode.NEAREST)
    gt_mask = np.array(mask_tensor)

    # --- Run Inference ---
    with torch.no_grad():
        num_reg = getattr(backbone_ft.config, 'num_register_tokens', 0)
        grid_h, grid_w = eval_h // patch_size, eval_w // patch_size
        
        # 1. Fine-Tuned Inference
        out_ft = backbone_ft(pixel_values=img_input, output_hidden_states=True)
        patches_ft = out_ft.last_hidden_state[:, 1+num_reg:, :]
        features_ft = patches_ft.reshape(1, grid_h, grid_w, backbone_ft.config.hidden_size).permute(0, 3, 1, 2)
        logits_ft = probe_ft(features_ft, (eval_h, eval_w))
        prediction_ft = torch.argmax(logits_ft, dim=1).squeeze(0).cpu().numpy()

        # 2. Baseline Inference
        out_base = backbone_base(pixel_values=img_input, output_hidden_states=True)
        patches_base = out_base.last_hidden_state[:, 1+num_reg:, :]
        features_base = patches_base.reshape(1, grid_h, grid_w, backbone_base.config.hidden_size).permute(0, 3, 1, 2)
        logits_base = probe_base(features_base, (eval_h, eval_w))
        prediction_base = torch.argmax(logits_base, dim=1).squeeze(0).cpu().numpy()

    # --- Prepare Colors ---
    safe_gt = np.where(gt_mask < len(color_palette), gt_mask, 0)
    gt_colored = color_palette[safe_gt]
    pred_colored_ft = color_palette[prediction_ft]
    pred_colored_base = color_palette[prediction_base]

    # --- Plotting (2x2 Grid) ---
    fig, axes = plt.subplots(2, 2, figsize=(24, 13))
    axes = axes.flatten()

    axes[0].imshow(img_pil_eval)
    axes[0].set_title(f"Input Image (Full HD)\n{sample_img}", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(gt_colored)
    axes[1].set_title("Ground Truth Pseudo-Mask", fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(pred_colored_ft)
    axes[2].set_title("Fine-Tuned Model Prediction", fontsize=14)
    axes[2].axis('off')

    axes[3].imshow(pred_colored_base)
    axes[3].set_title("Baseline Model Prediction", fontsize=14)
    axes[3].axis('off')

    # Global Legend at the bottom
    patches_legend = [mpatches.Patch(color=color_palette[c], label=id_to_class.get(c, f"Class {c}")) for c in range(num_classes)]
    fig.legend(handles=patches_legend, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.02), fontsize=14)

    plt.tight_layout(rect=[0, 0.05, 1, 1]) 
    
    # --- Save to Folder & Clean RAM ---
    save_path = os.path.join(output_dir, f"compare_{base_name}.png")
    plt.savefig(save_path, dpi=200)
    
    plt.clf()
    plt.cla()
    plt.close('all')
    
    # Aggressively delete everything to prevent memory leaks in WSL/Linux
    del img_input, out_ft, patches_ft, features_ft, logits_ft, prediction_ft
    del out_base, patches_base, features_base, logits_base, prediction_base
    del img_pil_eval, mask_tensor, gt_mask, safe_gt, gt_colored, pred_colored_ft, pred_colored_base, fig, axes
    
    gc.collect() 
    torch.cuda.empty_cache()

    print(f"[{i+1}/{num_images_to_process}] Saved: {save_path}")

print("\n✅ All Model Comparisons generated successfully!")