import os
import gc
import csv
import json
import logging
import argparse
import textwrap 
import shutil
from pathlib import Path
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torchvision.transforms.functional as TF
from transformers import AutoModel

from config import SeaDinoConfig
from utils import get_file_path
from models import PROBE_REGISTRY

logger = logging.getLogger("SeaDino")

class SeaDinoPipeline:
    def __init__(self, args: argparse.Namespace, config: SeaDinoConfig):
        self.args = args
        self.config = config
        self.class_map, self.id_to_class, self.num_classes = self._load_class_map()
        self.active_models = self._initialize_models()
        
        self.web_manifest_dict = {}

        self.total_metrics = {
            size: {
                bb_type: {"counts": np.zeros(self.num_classes, dtype=np.int64), "pixels": 0} 
                for bb_type in self.active_models.keys()
            }
            for size in self.args.sizes
        }

    def _load_class_map(self) -> Tuple[Dict[str, int], Dict[int, str], int]:
        map_path = get_file_path(os.path.join(self.args.base_dir, "class_map.json"), self.args.hf_repo)
        with open(map_path, 'r') as f:
            class_map = json.load(f)
        num_classes = max(class_map.values()) + 1
        id_to_class = {v: k for k, v in class_map.items()}
        return class_map, id_to_class, num_classes

    def _load_backbone(self, is_fg: bool, ckpt_name: str = None) -> nn.Module:
        display_name = "SeaDino-Seg-1-Fg" if is_fg else "SeaDino-Seg-1-Org"
        logger.info(f"Instantiating {display_name} Backbone...")
        
        model = AutoModel.from_pretrained(self.config.hf_model_id, attn_implementation='sdpa')
        
        if is_fg and ckpt_name:
            ckpt_path = get_file_path(ckpt_name, self.args.hf_repo)
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=self.config.device)
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
                
                new_state_dict = {}
                for k, v in state_dict.items():
                    k = k.replace('module.', '').replace('_orig_mod.', '')
                    if k.startswith('model.'): k = k[6:]
                    if k.startswith('student.backbone_ft.'): k = k.replace('student.backbone_ft.', '')
                    elif k.startswith('student.backbone.'): k = k.replace('student.backbone.', '')
                    elif k.startswith('backbone_ft.'): k = k.replace('backbone_ft.', '')
                    elif k.startswith('backbone.'): k = k.replace('backbone.', '')
                    if k.startswith('blocks.'): k = k.replace('blocks.', 'encoder.layer.')
                    if k.startswith('norm.'): k = k.replace('norm.', 'layernorm.')
                        
                    new_state_dict[k] = v
                
                model.load_state_dict(new_state_dict, strict=False)
            else:
                logger.warning(f"Checkpoint '{ckpt_name}' not found. Using baseline DINOv3.")
                
        return model.eval().to(self.config.device)

    def _initialize_models(self) -> Dict[str, Any]:
        models = {}
        if getattr(self.args, 'run_base', False): 
            models['org'] = {'display_name': 'SeaDino-Seg-1-Org', 'backbone': self._load_backbone(False, None), 'probes': []}
        if getattr(self.args, 'run_ft', False): 
            models['fg'] = {'display_name': 'SeaDino-Seg-1-Fg', 'backbone': self._load_backbone(True, self.args.ft_ckpt), 'probes': []}

        for bb_type in list(models.keys()):
            arg_prefix = 'base' if bb_type == 'org' else 'ft'
            for size in self.args.sizes:
                raw_weight_name = getattr(self.args, f"{arg_prefix}_probe_{size}")
                weight_path = get_file_path(raw_weight_name, self.args.hf_repo)
                
                if os.path.exists(weight_path):
                    try:
                        probe_model = PROBE_REGISTRY[size](models[bb_type]['backbone'].config.hidden_size, self.num_classes)
                        probe_model.load_state_dict(torch.load(weight_path, map_location=self.config.device))
                        models[bb_type]['probes'].append({
                            "size": size, "name": size.capitalize(), "model": probe_model.eval().to(self.config.device)
                        })
                        logger.info(f"Ready: {models[bb_type]['display_name']} ({size.upper()} Probe)")
                    except RuntimeError:
                        logger.warning(f"Skipped: Architecture mismatch in '{raw_weight_name}'.")
                else:
                    logger.warning(f"Skipped: File '{raw_weight_name}' not found locally or on HF.")
                    
            if not models[bb_type]['probes']:
                models.pop(bb_type)
        return models

    def run(self):
        if not self.active_models:
            logger.error("No valid probes found to run. Exiting.")
            return
            
        valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

        if getattr(self.args, 'image', None):
            img_path = self.args.image
            if not os.path.exists(img_path):
                logger.error(f"Image not found at {img_path}")
                return
            
            sample_img = os.path.basename(img_path)
            logger.info(f"Processing single image: {sample_img}")
            
            if not sample_img.endswith(valid_extensions):
                error_msg = f"Unsupported file format. Please upload JPG or PNG."
                logger.error(f"{sample_img} - {error_msg}")
                self._log_error_to_manifest(sample_img, error_msg)
            else:
                try:
                    self._process_single_image(sample_img, img_path, 1, 1)
                except Exception as e:
                    logger.error(f"Failed to process {sample_img}: {e}")
                    self._log_error_to_manifest(sample_img, str(e))
                
        else:
            potential_raw_dir = os.path.join(self.args.base_dir, "raw_images")
            if os.path.isdir(potential_raw_dir):
                img_dir = potential_raw_dir
            else:
                img_dir = self.args.base_dir

            if not os.path.exists(img_dir):
                logger.error(f"Directory {img_dir} does not exist.")
                return
                
            if not os.path.isdir(img_dir):
                logger.warning(f"Passed a file to --base_dir instead of a folder. Routing to single file logic.")
                all_files = [os.path.basename(img_dir)]
                img_dir = os.path.dirname(img_dir) 
            else:
                all_files = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
            
            if len(all_files) == 0:
                logger.error(f"No files found in {img_dir}")
                return

            num_images = min(self.args.num_imgs, len(all_files))

            for i in range(num_images):
                sample_img = all_files[i]
                img_path = os.path.join(img_dir, sample_img)
                
                if not sample_img.endswith(valid_extensions):
                    error_msg = f"Unsupported file format. Please upload JPG or PNG."
                    logger.warning(f"Skipping {sample_img} - {error_msg}")
                    self._log_error_to_manifest(sample_img, error_msg)
                    continue 
                    
                try:
                    self._process_single_image(sample_img, img_path, i + 1, num_images)
                except Exception as e:
                    logger.error(f"Failed to process {sample_img}: {e}")
                    self._log_error_to_manifest(sample_img, str(e))
                
            self._generate_final_summary(num_images)

        # NEW: Master Exports (JSON + CSV)
        if getattr(self.args, 'mode', 'all') in ['web_ui', 'all']:
            web_out_dir = Path(getattr(self.args, 'web_out_dir', 'web_ui_outputs'))
            web_out_dir.mkdir(parents=True, exist_ok=True)
            
            ui_legend = {}
            for c in range(self.num_classes):
                class_name = self.id_to_class.get(c, f"Class {c}")
                r, g, b = (self.config.color_palette[c] * 255).astype(int)
                ui_legend[class_name] = f"#{r:02x}{g:02x}{b:02x}"
            
            final_web_payload = {
                "ui_legend": ui_legend,
                "survey_results": list(self.web_manifest_dict.values())
            }
            
            manifest_path = web_out_dir / "outputs.json"
            with open(manifest_path, 'w') as f:
                json.dump(final_web_payload, f, indent=4)
            logger.info(f"✅ Web UI Master Manifest saved to: {manifest_path}")

            # NEW: Generate CSV coverage table
            csv_path = web_out_dir / "coverage.csv"
            self._generate_csv(csv_path)
            logger.info(f"✅ CSV Coverage Table saved to: {csv_path}")

        logger.info("SeaDino evaluation pipeline finished successfully!")

    # NEW: CSV Generator Function
    def _generate_csv(self, csv_path: Path):
        headers = ['Image', 'Status', 'Model'] + [self.id_to_class.get(c, f"Class {c}") for c in range(self.num_classes)]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for sample_img, data in self.web_manifest_dict.items():
                status = data.get("status", "unknown")
                if status == "error":
                    writer.writerow([sample_img, status, "N/A"] + [""] * self.num_classes)
                    continue
                    
                for pred_key, pred_data in data.get("predictions", {}).items():
                    coverage = pred_data.get("coverage", {})
                    row = [sample_img, status, pred_key]
                    for c in range(self.num_classes):
                        cls_name = self.id_to_class.get(c, f"Class {c}")
                        row.append(coverage.get(cls_name, 0.0))
                    writer.writerow(row)

    def _log_error_to_manifest(self, sample_img: str, error_msg: str):
        if getattr(self.args, 'mode', 'all') not in ['web_ui', 'all']: return
        web_out_dir = Path(getattr(self.args, 'web_out_dir', 'web_ui_outputs'))
        folder_name = web_out_dir.name
        
        self.web_manifest_dict[sample_img] = {
            "input": f"{folder_name}/images/{sample_img}",
            "status": "error",
            "error_message": error_msg
        }

    def _process_single_image(self, sample_img: str, img_path: str, current_idx: int, total_imgs: int):
        base_name = os.path.splitext(sample_img)[0]
        mask_path = os.path.join(self.args.masks_dir, base_name + '.png') if getattr(self.args, 'masks_dir', None) else ""

        orig_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = orig_pil.size
        img_np = np.array(orig_pil) 

        img_pil = TF.resize(orig_pil, [self.config.eval_h, self.config.eval_w], interpolation=TF.InterpolationMode.BILINEAR)
        img_input = TF.normalize(TF.to_tensor(img_pil), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).unsqueeze(0).to(self.config.device)

        results = {size: {} for size in self.args.sizes}

        for bb_type, bb_data in self.active_models.items():
            backbone = bb_data['backbone']
            with torch.no_grad():
                num_reg = getattr(backbone.config, 'num_register_tokens', 0)
                out = backbone(pixel_values=img_input, output_hidden_states=True)
                patches = out.last_hidden_state[:, 1+num_reg:, :]
                grid_h, grid_w = self.config.grid_size
                features = patches.reshape(1, grid_h, grid_w, backbone.config.hidden_size).permute(0, 3, 1, 2)
                
                for probe_info in bb_data['probes']:
                    size = probe_info["size"]
                    
                    logits = probe_info["model"](features, (orig_h, orig_w)) 
                    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy() 
                    prediction = np.argmax(probs, axis=0)

                    counts = np.bincount(prediction.flatten(), minlength=self.num_classes)
                    spread_pct = (counts / prediction.size) * 100
                    
                    self.total_metrics[size][bb_type]["counts"] += counts
                    self.total_metrics[size][bb_type]["pixels"] += prediction.size
                    
                    display_name = bb_data['display_name']
                    size_name = probe_info["name"]
                    print(f"\n📊 SPREAD % [{display_name} | {size_name}] FOR: {sample_img} (Size: {orig_w}x{orig_h})")
                    print("-" * 50)
                    for c in range(self.num_classes):
                        if spread_pct[c] > 0:
                            print(f"  {self.id_to_class.get(c, f'Class {c}'):<30}: {spread_pct[c]:>6.2f}%")
                    print("-" * 50)

                    results[size][bb_type] = {
                        "probs": probs, "pred": prediction, "spread_pct": spread_pct, "name": size_name
                    }
            del out, patches, features
            self._clean_memory()

        gt_colored = None
        gt_mask = None 
        if os.path.exists(mask_path):
            mask_tensor = TF.resize(Image.open(mask_path).convert("L"), [orig_h, orig_w], interpolation=TF.InterpolationMode.NEAREST)
            gt_mask = np.array(mask_tensor)
            gt_colored = self.config.color_palette[np.where(gt_mask < len(self.config.color_palette), gt_mask, 0)]

        if self.args.mode in ['heatmaps', 'all']:
            self._plot_heatmaps(sample_img, base_name, img_np, results)
            
        if self.args.mode in ['compare', 'all']:
            self._plot_comparison(sample_img, base_name, img_np, gt_colored, gt_mask, results) 
            
        if self.args.mode in ['compare_single', 'all']:
            if gt_colored is not None:
                self._plot_compare_single(sample_img, base_name, img_np, gt_colored, gt_mask, results) 
            else:
                logger.warning(f"Skipping compare_single: No GT mask found for {sample_img}")
                
        if self.args.mode in ['generate', 'all']:
            self._plot_generate(sample_img, base_name, img_np, results)
            
        if self.args.mode in ['web_ui', 'all']:
            self._export_web_ui(sample_img, base_name, img_np, results)
            # NEW: Route Matplotlib report directly into Web UI folders if flag is active
            if getattr(self.args, 'web_include_report', False):
                reports_dir = Path(getattr(self.args, 'web_out_dir', 'web_ui_outputs')) / "reports"
                self._plot_comparison(sample_img, base_name, img_np, gt_colored, gt_mask, results, override_out_dir=reports_dir)

    def _export_web_ui(self, sample_img: str, base_name: str, img_np: np.ndarray, results: dict):
        web_out_dir = Path(getattr(self.args, 'web_out_dir', 'web_ui_outputs'))
        images_dir = web_out_dir / "images"
        masks_dir = web_out_dir / "masks"
        conf_dir = web_out_dir / "confidence" # NEW: Folder for Hover Confidence Maps
        
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        conf_dir.mkdir(parents=True, exist_ok=True)
        folder_name = web_out_dir.name
        
        web_input_path = images_dir / sample_img
        original_img_path = os.path.join(self.args.base_dir, sample_img)
        potential_raw_dir = os.path.join(self.args.base_dir, "raw_images")
        if os.path.isdir(potential_raw_dir):
            original_img_path = os.path.join(potential_raw_dir, sample_img)

        if not web_input_path.exists() and os.path.exists(original_img_path):
            shutil.copy(original_img_path, web_input_path)

        orig_h, orig_w = img_np.shape[:2]
        aspect_ratio = round(orig_w / orig_h, 4)

        if sample_img not in self.web_manifest_dict:
            self.web_manifest_dict[sample_img] = {
                "input": f"{folder_name}/images/{sample_img}",
                "status": "success",
                "original_dimensions": {"width": orig_w, "height": orig_h},
                "rescaled_dimensions": {"width": self.config.eval_w, "height": self.config.eval_h},
                "aspect_ratio": aspect_ratio,
                "predictions": {}
            }

        for size, bb_results in results.items():
            for bb_type, res in bb_results.items():
                display_name = self.active_models[bb_type]['display_name']
                pred = res["pred"]
                probs = res["probs"] # Grab the raw confidence floats!
                spread_pct = res["spread_pct"]
                
                pred_rgb = (self.config.color_palette[pred] * 255).astype(np.uint8)
                alpha = np.where(pred == 0, 0, 153).astype(np.uint8)
                combined_mask_pil = Image.fromarray(np.dstack((pred_rgb, alpha)), "RGBA")
                combined_mask_name = f"overlay_{base_name}_{bb_type}-{size}_ALL.png"
                combined_mask_pil.save(masks_dir / combined_mask_name, format="PNG")
                
                spread_data = {}
                individual_masks = {}
                confidence_maps = {} # NEW: Track confidence maps
                
                for c in range(self.num_classes):
                    class_name = self.id_to_class.get(c, f"Class {c}")
                    spread_data[class_name] = round(float(spread_pct[c]), 2)
                    
                    if spread_pct[c] > 0 and "background" not in class_name.lower():
                        class_pixels = (pred == c)
                        rgba_img = np.zeros((pred.shape[0], pred.shape[1], 4), dtype=np.uint8)
                        color = (self.config.color_palette[c] * 255).astype(np.uint8)
                        rgba_img[class_pixels, :3] = color
                        rgba_img[class_pixels, 3] = 153 
                        
                        class_mask_pil = Image.fromarray(rgba_img, "RGBA")
                        safe_class_name = class_name.replace(" ", "_")
                        class_mask_name = f"layer_{base_name}_{bb_type}-{size}_{safe_class_name}.png"
                        class_mask_pil.save(masks_dir / class_mask_name, format="PNG")
                        individual_masks[class_name] = f"{folder_name}/masks/{class_mask_name}"
                        
                        # NEW: Generate Hover Confidence Map (Grayscale: 0-255)
                        conf_img = (probs[c] * 255).astype(np.uint8)
                        conf_pil = Image.fromarray(conf_img, "L") 
                        conf_name = f"conf_{base_name}_{bb_type}-{size}_{safe_class_name}.png"
                        conf_pil.save(conf_dir / conf_name, format="PNG")
                        confidence_maps[class_name] = f"{folder_name}/confidence/{conf_name}"
                
                prediction_key = f"{display_name}-{size}"
                self.web_manifest_dict[sample_img]["predictions"][prediction_key] = {
                    "overlay_mask_combined": f"{folder_name}/masks/{combined_mask_name}",
                    "overlay_mask_layers": individual_masks,
                    "confidence_maps": confidence_maps, # Sent directly to the JSON!
                    "coverage": spread_data
                }

    def _generate_final_summary(self, total_images: int):
        print("\n" + "="*65)
        print(f"🌍 FINAL SPREAD % (TOTAL COVERAGE ACROSS {total_images} IMAGES)")
        print("="*65)
        
        for size, bb_results in self.total_metrics.items():
            for bb_type, metrics in bb_results.items():
                if metrics["pixels"] == 0: continue
                    
                display_name = self.active_models[bb_type]['display_name']
                print(f"\n▶ Model: {display_name} | Size: {size.capitalize()}")
                print("-" * 65)
                
                global_spread_pct = (metrics["counts"] / metrics["pixels"]) * 100
                for c in range(self.num_classes):
                    if global_spread_pct[c] > 0:
                        print(f"  {self.id_to_class.get(c, f'Class {c}'):<30}: {global_spread_pct[c]:>6.2f}%")
        print("="*65 + "\n")

    def _plot_heatmaps(self, sample_img: str, base_name: str, img_np: np.ndarray, results: dict):
        for size, bb_results in results.items():
            for bb_type, res in bb_results.items():
                display_name = self.active_models[bb_type]['display_name']
                out_dir = Path(f"output_heatmaps_{display_name}_{size.upper()}")
                out_dir.mkdir(exist_ok=True)

                pred, probs, spread_pct, p_name = res["pred"], res["probs"], res["spread_pct"], res["name"]
                rgba_pred = np.dstack((self.config.color_palette[pred], np.where(pred == 0, 0.0, 0.6)))

                fig, axes = plt.subplots(2, 4, figsize=(32, 12))
                axes = axes.flatten()
                axes[0].imshow(img_np); axes[0].set_title(f"Raw Input\n{sample_img}", fontsize=14); axes[0].axis('off')
                axes[1].imshow(img_np); axes[1].imshow(rgba_pred)
                axes[1].set_title(f"Winning Prediction [{display_name} | {p_name}]", fontsize=14); axes[1].axis('off')
                
                present_classes = np.unique(pred)
                legend_patches = [
                    mpatches.Patch(color=self.config.color_palette[c], label=f"{self.id_to_class.get(c, f'Class {c}')} ({spread_pct[c]:.2f}%)")
                    for c in present_classes
                ]
                axes[1].legend(handles=legend_patches, bbox_to_anchor=(0.0, -0.05), loc='upper left', ncol=2)

                for c in range(self.num_classes):
                    ax = axes[c + 2] 
                    heatmap = ax.imshow(probs[c], cmap='inferno', vmin=0.0, vmax=1.0)
                    ax.set_title(f"{self.id_to_class.get(c, f'Class {c}')} Confidence", fontsize=14, color=self.config.color_palette[c] if c!=0 else 'black')
                    ax.axis('off')
                    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=10)

                plt.tight_layout()
                plt.savefig(out_dir / f"{base_name}_{display_name}_{p_name}.png", dpi=self.config.dpi, bbox_inches='tight')
                fig.clf() 
                plt.close(fig)

    # NEW: Added override_out_dir parameter to support the Web export flag
    def _plot_comparison(self, sample_img: str, base_name: str, img_np: np.ndarray, gt_colored: np.ndarray, gt_mask: np.ndarray, results: dict, override_out_dir: Path = None):
        rgba_gt = None
        if gt_colored is not None and gt_mask is not None:
            gt_safe = np.where(gt_mask < len(self.config.color_palette), gt_mask, 0)
            gt_alpha = np.where(gt_safe == 0, 0.0, 0.6)
            rgba_gt = np.dstack((gt_colored, gt_alpha))

        for size, bb_results in results.items():
            if not bb_results: continue 
            
            # USE override_out_dir if provided!
            out_dir = override_out_dir if override_out_dir else Path(f"output_comparisons_{size.upper()}")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            fig, axes = plt.subplots(2, 2, figsize=(24, 13))
            axes = axes.flatten()
            axes[0].imshow(img_np); axes[0].set_title(f"Input Image\n{sample_img}", fontsize=14); axes[0].axis('off')
            
            if rgba_gt is not None:
                axes[1].imshow(img_np)
                axes[1].imshow(rgba_gt)
                axes[1].set_title("Ground Truth Overlay", fontsize=14)
            else:
                axes[1].text(0.5, 0.5, "GT Mask Not Found", ha='center', va='center', fontsize=20, color='gray'); axes[1].set_title("Ground Truth", fontsize=14)
            axes[1].axis('off')

            if 'fg' in bb_results:
                pred = bb_results['fg']['pred']
                rgba_fg = np.dstack((self.config.color_palette[pred], np.where(pred == 0, 0.0, 0.6)))
                axes[2].imshow(img_np)
                axes[2].imshow(rgba_fg)
                axes[2].set_title(f"SeaDino-Seg-1-Fg Overlay ({size.capitalize()})", fontsize=14, fontweight='bold')
            else:
                axes[2].text(0.5, 0.5, "No SeaDino-Seg-1-Fg Loaded", ha='center', va='center', fontsize=20, color='gray')
            axes[2].axis('off')

            if 'org' in bb_results:
                pred = bb_results['org']['pred']
                rgba_org = np.dstack((self.config.color_palette[pred], np.where(pred == 0, 0.0, 0.6)))
                axes[3].imshow(img_np)
                axes[3].imshow(rgba_org)
                axes[3].set_title(f"SeaDino-Seg-1-Org Overlay ({size.capitalize()})", fontsize=14)
            else:
                axes[3].text(0.5, 0.5, "No SeaDino-Seg-1-Org Loaded", ha='center', va='center', fontsize=20, color='gray')
            axes[3].axis('off')

            legend_patches = []
            for c in range(self.num_classes):
                raw_name = self.id_to_class.get(c, f"Class {c}").replace('_', ' ')
                clean_name = textwrap.fill(raw_name, width=25) 
                legend_patches.append(mpatches.Patch(color=self.config.color_palette[c], label=clean_name))

            fig.legend(handles=legend_patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=14)
            plt.tight_layout(rect=[0, 0.08, 1, 1]) 
            
            plt.savefig(out_dir / f"compare_{base_name}_{size}.png", dpi=self.config.dpi, bbox_inches='tight')
            fig.clf() 
            plt.close(fig)

    def _plot_compare_single(self, sample_img: str, base_name: str, img_np: np.ndarray, gt_colored: np.ndarray, gt_mask: np.ndarray, results: dict):
        gt_safe = np.where(gt_mask < len(self.config.color_palette), gt_mask, 0)
        gt_alpha = np.where(gt_safe == 0, 0.0, 0.6)
        rgba_gt = np.dstack((gt_colored, gt_alpha))

        for size, bb_results in results.items():
            for bb_type, res in bb_results.items():
                display_name = self.active_models[bb_type]['display_name']
                out_dir = Path(f"output_compare_single_{size.upper()}")
                out_dir.mkdir(exist_ok=True)

                pred, spread_pct = res["pred"], res["spread_pct"]
                pred_colored = self.config.color_palette[pred]
                rgba_pred = np.dstack((pred_colored, np.where(pred == 0, 0.0, 0.6)))

                fig, axes = plt.subplots(1, 3, figsize=(24, 8))
                
                axes[0].imshow(img_np); axes[0].set_title(f"Input Image\n{sample_img}", fontsize=14); axes[0].axis('off')
                axes[1].imshow(img_np); axes[1].imshow(rgba_gt); axes[1].set_title("Ground Truth Overlay", fontsize=14); axes[1].axis('off')
                axes[2].imshow(img_np); axes[2].imshow(rgba_pred); axes[2].set_title(f"{display_name} Overlay ({size.capitalize()})", fontsize=14, fontweight='bold'); axes[2].axis('off')

                present_classes = np.unique(pred)
                legend_patches = []
                for c in present_classes:
                    raw_name = self.id_to_class.get(c, f"Class {c}").replace('_', ' ')
                    clean_name = textwrap.fill(raw_name, width=25)
                    label_str = f"{clean_name}\n({spread_pct[c]:.2f}%)"
                    legend_patches.append(mpatches.Patch(color=self.config.color_palette[c], label=label_str))

                fig.legend(handles=legend_patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=14)
                plt.tight_layout(rect=[0, 0.12, 1, 1]) 
                
                plt.savefig(out_dir / f"compare_single_{base_name}_{display_name}_{size}.png", dpi=self.config.dpi, bbox_inches='tight')
                fig.clf()
                plt.close(fig)

    def _plot_generate(self, sample_img: str, base_name: str, img_np: np.ndarray, results: dict):
        for size, bb_results in results.items():
            for bb_type, res in bb_results.items():
                display_name = self.active_models[bb_type]['display_name']
                out_dir = Path(f"output_generate_{size.upper()}")
                out_dir.mkdir(exist_ok=True)

                pred, spread_pct = res["pred"], res["spread_pct"]
                pred_colored = self.config.color_palette[pred]
                rgba_pred = np.dstack((pred_colored, np.where(pred == 0, 0.0, 0.6)))

                fig, axes = plt.subplots(1, 3, figsize=(24, 8))
                
                axes[0].imshow(img_np); axes[0].set_title(f"Input Image\n{sample_img}", fontsize=14); axes[0].axis('off')
                axes[1].imshow(pred_colored); axes[1].set_title(f"Predicted Mask ({size.capitalize()})", fontsize=14); axes[1].axis('off')
                axes[2].imshow(img_np); axes[2].imshow(rgba_pred); axes[2].set_title(f"Overlay Mask ({display_name})", fontsize=14, fontweight='bold'); axes[2].axis('off')

            
                present_classes = np.unique(pred)
                legend_patches = []
                for c in present_classes:
                    raw_name = self.id_to_class.get(c, f"Class {c}").replace('_', ' ')
                    clean_name = textwrap.fill(raw_name, width=25)
                    label_str = f"{clean_name}\n({spread_pct[c]:.2f}%)"
                    legend_patches.append(mpatches.Patch(color=self.config.color_palette[c], label=label_str))

                fig.legend(handles=legend_patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=14)
                plt.tight_layout(rect=[0, 0.12, 1, 1]) 
                
                plt.savefig(out_dir / f"generate_{base_name}_{display_name}_{size}.png", dpi=self.config.dpi, bbox_inches='tight')
                fig.clf()
                plt.close(fig)
                
    @staticmethod
    def _clean_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()