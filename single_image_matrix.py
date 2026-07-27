import argparse
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
# 1. MODELS & SETUP
# ============================================================================
class DenseLinearProbe(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
    def forward(self, x, target_size):
        logits = self.classifier(x)
        return F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)

color_palette = np.array([
    [0.0, 0.0, 0.0],       # 0: Background / Unlabeled (Black)
    [0.8, 0.2, 0.2],       # 1: Rock (Red)
    [0.2, 0.8, 0.2],       # 2: Carpophyllum (Green)
    [0.8, 0.8, 0.2],       # 3: Ecklonia (Yellow)
    [0.2, 0.2, 0.8],       # 4: Amphiroa (Blue)
    [0.8, 0.2, 0.8]        # 5: Anthothoe (Purple)
])

def load_fine_tuned_backbone(model_id, checkpoint_path, device):
    print(f"Loading Fine-Tuned Backbone Checkpoint: {checkpoint_path}")
    model = AutoModel.from_pretrained(model_id, attn_implementation='sdpa', torch_dtype=torch.float32)
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

    model.load_state_dict(new_state_dict, strict=False)
    model.eval().to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description="Single Image Inference Tool")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output_dir", type=str, default="single_predictions", help="Where to save the result")
    parser.add_argument("--use_ft", action="store_true", help="Set flag to evaluate Fine-Tuned model instead of Baseline")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Error: Image '{args.image}' not found!")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    base_dir = r"/home/neel/d_drive/ai_data/data/data_to_ivy"
    checkpoint_path = r"/home/neel/stage2_ckpts/benthic-ssl-epoch=44-ssl_loss=13.00.ckpt"
    model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
    patch_size = 16

    # Choose Probe Weights & Backbone based on --use_ft flag
    if args.use_ft:
        probe_weights = "best_ft_probe.pth"
        print("🔍 Mode: FINE-TUNED SSL BACKBONE + PROBE")
    else:
        probe_weights = "best_base_probe.pth"
        print("🔍 Mode: BASELINE DINOv3 BACKBONE + PROBE")

    if not os.path.exists(probe_weights):
        print(f"❌ Error: Probe weights '{probe_weights}' not found in current directory!")
        return

    # ============================================================================
    # 2. LOAD CLASS MAP & MODELS
    # ============================================================================
    with open(os.path.join(base_dir, "class_map.json"), 'r') as f:
        class_map = json.load(f)
        num_classes = max(class_map.values()) + 1
        id_to_class = {v: k for k, v in class_map.items()}

    # Load Backbone
    if args.use_ft:
        backbone = load_fine_tuned_backbone(model_id, checkpoint_path, device)
    else:
        print("Loading Baseline Backbone...")
        backbone = AutoModel.from_pretrained(model_id, attn_implementation='sdpa', torch_dtype=torch.float32)
        backbone.eval().to(device)

    # Load Linear Probe
    probe = DenseLinearProbe(backbone.config.hidden_size, num_classes).to(device)
    probe.load_state_dict(torch.load(probe_weights, map_location=device))
    probe.eval()

    # ============================================================================
    # 3. LOAD IMAGE & DYNAMIC RESOLUTION
    # ============================================================================
    orig_img = Image.open(args.image).convert("RGB")
    orig_w, orig_h = orig_img.size
    
    # Calculate nearest dimensions perfectly divisible by 16 (for DINO)
    eval_w = max(16, int(np.round(orig_w / 16.0)) * 16)
    eval_h = max(16, int(np.round(orig_h / 16.0)) * 16)

    img_pil = TF.resize(orig_img, [eval_h, eval_w], interpolation=TF.InterpolationMode.BILINEAR)
    img_input = TF.normalize(TF.to_tensor(img_pil), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).unsqueeze(0).to(device)

    # ============================================================================
    # 4. RUN INFERENCE (Fixed Register Token Bug)
    # ============================================================================
    print(f"Running inference on {os.path.basename(args.image)}...")
    with torch.no_grad():
        out = backbone(pixel_values=img_input, output_hidden_states=True)
        
        # FIX: Set default fallback to 0 instead of 4
        num_reg = getattr(backbone.config, 'num_register_tokens', 0)
        patches = out.last_hidden_state[:, 1+num_reg:, :]
        
        features = patches.reshape(1, eval_h // patch_size, eval_w // patch_size, backbone.config.hidden_size).permute(0, 3, 1, 2)
        logits = probe(features, (eval_h, eval_w))
        prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    # ============================================================================
    # 5. CALCULATE SPREAD % (PERCENT COVER)
    # ============================================================================
    total_pixels = prediction.size
    counts = np.bincount(prediction.flatten(), minlength=num_classes)
    spread_pct = (counts / total_pixels) * 100

    model_type_str = "Fine-Tuned" if args.use_ft else "Baseline"
    print("\n" + "="*50)
    print(f"📊 SPREAD % [{model_type_str}] FOR: {os.path.basename(args.image)}")
    print("="*50)
    for c in range(num_classes):
        class_name = id_to_class.get(c, f"Class {c}")
        if spread_pct[c] > 0:
            print(f"  {class_name:<30}: {spread_pct[c]:>6.2f}%")
    print("="*50 + "\n")

    # ============================================================================
    # 6. VISUALIZE (RAW -> MASK -> OVERLAY)
    # ============================================================================
    pred_colored = color_palette[prediction]
    alpha_channel = np.where(prediction == 0, 0.0, 0.6)
    rgba_prediction = np.dstack((pred_colored, alpha_channel))

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    axes[0].imshow(img_pil)
    axes[0].set_title("Input Image", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(pred_colored)
    axes[1].set_title(f"Predicted Mask ({model_type_str})", fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(img_pil)
    axes[2].imshow(rgba_prediction)
    axes[2].set_title(f"Overlay ({model_type_str})", fontsize=14)
    axes[2].axis('off')

    # Legend
    present_classes = np.unique(prediction)
    patches_legend = [
        mpatches.Patch(
            color=color_palette[c], 
            label=f"{id_to_class.get(c, f'Class {c}')} ({spread_pct[c]:.1f}%)"
        ) for c in present_classes
    ]
    plt.legend(handles=patches_legend, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)

    plt.tight_layout()
    
    # Save
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    prefix = "ft" if args.use_ft else "base"
    save_path = os.path.join(args.output_dir, f"report_{prefix}_{base_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    plt.clf()
    plt.cla()
    plt.close('all')
    gc.collect()

    print(f"✅ Saved visual report to: {save_path}")

if __name__ == "__main__":
    main()