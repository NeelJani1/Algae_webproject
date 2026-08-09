import os
import logging
import argparse

from config import SeaDinoConfig
from pipeline import SeaDinoPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Tiny-Big Probes for SeaDino-Seg-1 Models.")
    
    parser.add_argument('--mode', type=str, 
                        choices=['heatmaps', 'compare', 'compare_single', 'generate', 'all', 'web_ui'], 
                        default='all')
    parser.add_argument('--web_out_dir', type=str, default="web_ui_outputs", 
                        help="Directory to save raw overlay masks and JSON manifests for the Web UI.")
    
    # NEW: Flag to include Matplotlib reports in the web export
    parser.add_argument('--web_include_report', action='store_true', 
                        help="Save Matplotlib comparison grids inside the web_ui folder.")
    
    parser.add_argument('--hf_repo', type=str, default="Neel536/Algea_Segmentation_Model")
    parser.add_argument('--sizes', type=str, nargs='+', choices=['tiny', 'small', 'medium', 'big'], default=['small'])
    
    parser.add_argument('--run_base', action='store_true', help="Run Original Baseline.")
    parser.add_argument('--run_ft', action='store_true', help="Run Fine-Tuned.")
    
    parser.add_argument('--ft_ckpt', type=str, default="SeaDino-Seg-1-Fg-Backbone.ckpt")
    parser.add_argument('--base_probe_tiny', type=str, default="SeaDino-Seg-1-Org-Tiny.pth")
    parser.add_argument('--base_probe_small', type=str, default="SeaDino-Seg-1-Org-Small.pth")
    parser.add_argument('--base_probe_medium', type=str, default="SeaDino-Seg-1-Org-Medium.pth")
    parser.add_argument('--base_probe_big', type=str, default="SeaDino-Seg-1-Org-Big.pth")
    parser.add_argument('--ft_probe_tiny', type=str, default="SeaDino-Seg-1-Fg-Tiny.pth")
    parser.add_argument('--ft_probe_small', type=str, default="SeaDino-Seg-1-Fg-Small.pth")
    parser.add_argument('--ft_probe_medium', type=str, default="SeaDino-Seg-1-Fg-Medium.pth")
    parser.add_argument('--ft_probe_big', type=str, default="SeaDino-Seg-1-Fg-Big.pth")
    
    parser.add_argument('--base_dir', type=str, default="/home/neel/d_drive/ai_data/data/data_to_ivy")
    parser.add_argument('--masks_dir', type=str, default=None)
    parser.add_argument('--num_imgs', type=int, default=20)
    parser.add_argument('--image', type=str, default=None, help="Path to a single image file (overrides base_dir loop).")

    parser.add_argument('--eval_h', type=int, default=1088)
    parser.add_argument('--eval_w', type=int, default=1920)
    parser.add_argument('--dpi', type=int, default=200)
    
    args = parser.parse_args()
    
    if not args.run_base and not args.run_ft:
        logger = logging.getLogger("SeaDino")
        logger.warning("No backbone specified. Defaulting to --run_ft.")
        args.run_ft = True
    if args.masks_dir is None:
        args.masks_dir = os.path.join(args.base_dir, "pseudo_masks")
        
    return args

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    args = parse_args()
    config = SeaDinoConfig(eval_h=args.eval_h, eval_w=args.eval_w, dpi=args.dpi)
    
    pipeline = SeaDinoPipeline(args, config)
    pipeline.run()

if __name__ == "__main__":
    main()