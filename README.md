# SeaDino-Seg-1: Benthic Algae Segmentation Pipeline

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/NeelJani1/Algae_webproject)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model%20Weights-yellow?logo=huggingface)](https://huggingface.co/Neel536/Algea_Segmentation_Model)

An evaluation pipeline for DINOv3-based benthic segmentation models. 
Supports multiple spatial decoder sizes (tiny, small, medium, big), dynamic resolution, automated Hugging Face weight retrieval, and percent-cover metrics across surveys.

---

## 📁 Project Structure

```text
SeaDino_Project/
├── input/             # Place your raw survey/benthic images here to evaluate
├── .env.example       # Environment template for secure Hugging Face token
├── config.py          # Static settings (DPI, color palette, resolutions)
├── models.py          # Neural Network definitions (Tiny to Big heads)
├── utils.py           # Helper functions (Secure Hugging Face Hub downloader)
├── pipeline.py        # Core processing engine (Inference, visualization, statistics)
├── evaluate.py        # CLI Entrypoint (The only script you run)
├── environment.yml    # Anaconda environment specification (Python 3.13)
├── requirements.txt   # Standard pip requirements
└── README.md          # Documentation (This file)
```

---

## 🚀 Installation & Setup

Ensure you have Anaconda or Miniconda installed, then follow these steps:

### 1. Create and activate the conda environment
```bash
conda env create -f environment.yml
conda activate seadino_env
```
*(Note: If you are setting this up manually without Conda, you can use Python 3.13 and run `pip install -r requirements.txt` instead).*

> **GPU Users:** Ensure your environment has the correct PyTorch version matching your CUDA architecture installed so the model runs efficiently on a GPU.

### 2. Configure Secure Hugging Face Access (Required)
Model weights are hosted in a private Hugging Face repository. To allow the script to download them securely:
1. Copy or rename `.env.example` to `.env`.
2. Open the `.env` file and insert your Hugging Face Access Token:
   ```text
   HF_TOKEN=hf_YourCopiedTokenHere
   ```

---

## 📊 How to Run Evaluations

Place your target images into your input directory, then run the pipeline using `evaluate.py`. The script will automatically fetch necessary backbone and probe weights using your `.env` configuration.

### 1. Web UI Export (Optimized for Frontend Integration)

Generates a production-ready, highly organized export designed for web servers and interactive dashboards.
* **Organized Architecture:** Saves assets cleanly into `/images`, `/masks`, and `/confidence` subfolders.
* **Pixel-Perfect Alignment:** Masks are dynamically upsampled in the backend to match the exact original aspect ratio of the raw uploaded images (e.g., 1920x1080).
* **Automatic CSV Generation:** Generates a clean `coverage.csv` table at the end of the run containing raw, mathematically correct species-spread statistics for all processed images.
* **Performance-Optimized Extra Assets:** To prevent server disk/I/O bloat, heavy visualization assets (individual class layers and hover confidence maps) are turned off by default. Use `--web_export_extras` to generate them.
* **Species-Specific Filtering:** Use `--web_target_classes` to generate individual masks/confidence maps only for specific species selected by the user.
* **Dynamic Reports:** Use `--web_include_report` and `--web_report_type` to automatically generate and save Matplotlib reports directly inside `/reports/`.

**Example: Run the basic fast Web UI export:**
```bash
python evaluate.py --run_ft --sizes small --mode web_ui --web_out_dir web_ui_outputs
```

**Example: Advanced Run (Export extras only for Ecklonia, and generate reports):**
```bash
python evaluate.py --run_ft --sizes small --mode web_ui --web_export_extras --web_target_classes "Ecklonia_Deepwatercove" --web_include_report --web_report_type generate heatmaps
```

---

### 2. Side-by-Side Comparison (2x2 Grid)

Generates a comparison grid dynamically pairing and displaying **any two** predictions side-by-side (e.g., comparing model sizes like `Fg (Tiny)` vs `Fg (Small)` or architectures like `Baseline (Org)` vs `Fine-Tuned (Fg)`).

```bash
python evaluate.py --run_base --run_ft --sizes small --mode compare
```

### 3. Class Confidence Heatmaps (2x4 Grid)

Generates confidence heatmaps for all benthic classes individually.

```bash
python evaluate.py --run_base --run_ft --sizes small --mode heatmaps
```

### 4. Generate Everything (Research Mode)

Runs both comparison and heatmap visualizations simultaneously.

```bash
python evaluate.py --run_base --run_ft --sizes small --mode all
```

---

## ⚙️ Advanced Command-Line Flags

Fine-tune your evaluation runs using the parameters below:

### 🧩 Model & Size Configuration
* `--sizes`: Select global probe sizes to evaluate (`tiny`, `small`, `medium`, `big`). Run multiple sequentially:
  ```bash
  python evaluate.py --run_ft --sizes tiny small
  ```
* `--base_sizes` / `--ft_sizes`: Override probe sizes specifically for the Original Baseline or Fine-Tuned models independently.

### 🌐 Web UI Export Tweaks
* `--web_export_extras`: Toggle export of heavy visual assets (individual transparent class masks and pixel-wise grayscale confidence maps) to prevent server disk bloat.
* `--web_target_classes`: Restrict extra visual assets only to specified classes:
  ```bash
  python evaluate.py --run_ft --sizes small --mode web_ui --web_export_extras --web_target_classes "Rock_Deepwatercove" "Ecklonia_Deepwatercove"
  ```
* `--web_include_report`: Enable saving Matplotlib summary reports directly into the web output directory.
* `--web_report_type`: Choose specific report layouts (supports multiple simultaneously: `compare`, `compare_single`, `generate`, `heatmaps`, `all`).

### 📐 Resolution, Quality & Batch Control
* `--eval_w` & `--eval_h`: Change image evaluation resolution *(Note: Dimensions must be divisible by 16)*:
  ```bash
  python evaluate.py --run_ft --sizes small --eval_w 1280 --eval_h 720
  ```
* `--dpi`: Set image export quality. Lower values speed up file writing (recommended range: `100` to `600`).
* `--num_imgs`: Limit the total number of images processed from your raw folder (default is `20`).

---

## 📈 Percent Cover Reporting

The pipeline automatically calculates and logs the **Spread % (Percent Cover)** of each benthic class both per-image and globally across the entire batch (total survey coverage) upon completion.

---

## 📝 License & Credits

* **Model Backbone:** [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
* **Pipeline Development:** [Neel Jani](https://github.com/NeelJani1/Algae_webproject) / SeaDino-Seg-1
* **Hugging Face Weights:** [Neel536/Algea_Segmentation_Model](https://huggingface.co/Neel536/Algea_Segmentation_Model)
* **License:** MIT License
