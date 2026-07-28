# SeaDino-Seg-1: Benthic Algae Segmentation Pipeline

An evaluation pipeline for DINOv3-based benthic segmentation models. 
Supports multiple spatial decoder sizes (tiny, small, medium, big), dynamic resolution, automated Hugging Face weight retrieval, and percent-cover metrics across surveys.

---

## 📁 Project Structure

```text
SeaDino_Project/
├── config.py          # Static settings (DPI, color palette, resolutions)
├── models.py          # Neural Network definitions (Tiny to Big heads)
├── utils.py           # Helper functions (Hugging Face Hub downloader)
├── pipeline.py        # Core processing engine (Inference, visualization, statistics)
├── evaluate.py        # CLI Entrypoint (The only script you run)
├── environment.yml    # Anaconda environment specification (Python 3.13)
├── requirements.txt   # Standard pip requirements
└── README.md          # Documentation (This file)
```

---

## 🚀 Installation & Setup

Ensure you have Anaconda or Miniconda installed, then set up the environment:

```bash
# 1. Create and activate the conda environment
conda env create -f environment.yml
conda activate seadino_env
```

---

## 📊 How to Run Evaluations

Run the pipeline using `evaluate.py`. The script automatically retrieves the necessary backbone and probe weights from the Hugging Face registry [Neel536/Algea_Segmentation_Model](https://huggingface.co/Neel536/Algea_Segmentation_Model).

### 1. Side-by-Side Comparison (2x2 Grid)
Generates a grid displaying the Input Image, Ground Truth Pseudo-Mask (if available), Fine-Tuned (`SeaDino-Seg-1-Fg`) prediction, and Baseline (`SeaDino-Seg-1-Org`) prediction.
```bash
python evaluate.py --run_base --run_ft --sizes small --mode compare
```

### 2. Class Confidence Heatmaps (2x4 Grid)
Generates confidence heatmaps for all 6 benthic classes individually.
```bash
python evaluate.py --run_base --run_ft --sizes small --mode heatmaps
```

### 3. Generate Everything
Runs both comparison and heatmap visualizations simultaneously.
```bash
python evaluate.py --run_base --run_ft --sizes small --mode all
```

---

## ⚙️ Advanced Command-Line Flags

*   `--sizes`: Select probe sizes to evaluate (`tiny`, `small`, `medium`, `big`). You can run multiple sizes sequentially:
    ```bash
    python evaluate.py --run_ft --sizes tiny small
    ```
*   `--eval_w` and `--eval_h`: Change the image evaluation resolution (must be divisible by 16):
    ```bash
    python evaluate.py --run_ft --sizes small --eval_w 1280 --eval_h 720
    ```
*   `--dpi`: Set image export quality. Lower values speed up file writing (recommended range: 100 to 600):
    ```bash
    python evaluate.py --run_ft --sizes small --dpi 100
    ```
*   `--num_imgs`: Limit the number of images processed from your raw folder (default is 20).

---

## 📈 Percent Cover Reporting

The pipeline automatically calculates and logs the **Spread % (Percent Cover)** of each benthic class both per-image and globally across the entire batch (total survey coverage) at the end of execution.

---

## 📝 License & Credits

*   **Model Backbone:** [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
*   **Pipeline Development:** Neel / SeaDino-seg-1
*   **License:** MIT License
