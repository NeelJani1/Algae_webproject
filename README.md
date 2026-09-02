# SeaDino-Seg-1: Benthic Algae Segmentation Pipeline

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/NeelJani1/Algae_webproject)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model%20Weights-yellow?logo=huggingface)](https://huggingface.co/SeaDinoWeb/Algea_Segmentation_Model)

An advanced evaluation pipeline for DINOv3-based benthic segmentation models. Trained on pseudo-labeled marine imagery collected by the IVS Lab (University of Auckland), this pipeline supports multiple spatial decoder sizes, dynamic resolution, automated secure weight retrieval, and percent-cover metrics across ecological surveys.

---

## 📂 Project Structure

```text
SeaDino_Project/
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

## 🚀 Quick Start

Ensure you have Anaconda or Miniconda installed, then set up the environment:

**1. Create and activate the conda environment:**
```bash
conda env create -f environment.yml
conda activate seadino_env
pip install -r requirements.txt
```

**2. Configure Secure Hugging Face Access (Required):**
The fine-tuned model weights are hosted in a private Hugging Face repository. To allow the script to download them securely:
* Rename `.env.example` to `.env`
* Open the `.env` file and replace the placeholder with your Hugging Face Access Token:
  ```text
  HF_TOKEN=hf_YourCopiedTokenHere
  ```

---

## 💻 Evaluation Pipeline

Run the pipeline using `evaluate.py`. The script automatically retrieves the necessary backbone and probe weights securely using your `.env` configuration.

### 1. Web UI Export (Optimized for Frontend Integration)

Generates a production-ready, highly organized export designed for web servers and interactive dashboards.
* **Organized Architecture:** Saves assets cleanly into `/images`, `/masks`, and `/confidence` subfolders.
* **Pixel-Perfect Alignment:** AI masks are dynamically upsampled in the backend to match the exact original aspect ratio of the raw uploaded images (e.g., 1920x1080), eliminating padding artifacts.
* **Automatic CSV Generation:** Generates a clean `coverage.csv` table containing the raw, mathematically correct species-spread statistics for all processed images.
* **Interactive Web Layers:** Generates hidden class and confidence maps allowing frontend UIs to build real-time pixel-wise hover tooltips.
* **Performance-Optimized:** To prevent server disk bloat, heavy visualization assets (individual class layers and hover confidence maps) are turned off by default. Use `--web_export_extras` to generate them.
* **Species-Specific Filtering:** Use `--web_target_classes` to generate individual masks/confidence maps only for the exact species selected by the user.

**Example: Run the basic fast Web UI export:**
```bash
python evaluate.py --run_ft --sizes small --mode web_ui --web_out_dir web_ui_outputs
```

**Example: Advanced Run (Export extras only for Ecklonia, and generate both Generate and Heatmap reports):**
```bash
python evaluate.py --run_ft --sizes small --mode web_ui --web_export_extras --web_target_classes "Ecklonia_Deepwatercove" --web_include_report --web_report_type generate heatmaps
```

---

### 2. Side-by-Side Comparison (2x2 Grid)

Generates a comparison grid.
* **Fully Generalized:** The engine dynamically pairs and displays **any two** predictions side-by-side. 
* This allows you to compare different model sizes (e.g., `Fg (Tiny)` vs `Fg (Small)`) or different architectures (e.g., `Model Kiwi (Org)` vs `Model Moana (Fg)`) on a single canvas.

```bash
python evaluate.py --run_base --run_ft --sizes small --mode compare
```

### 3. Class Confidence Heatmaps (2x4 Grid)

Generates confidence heatmaps for all 6 benthic classes individually.

```bash
python evaluate.py --run_base --run_ft --sizes small --mode heatmaps
```

### 4. Dimensionality Reduction (UMAP)

Extracts DINOv3 latent features and generates a Supervised 2D UMAP scatterplot to mathematically verify class clustering and AI biological distinction.

```bash
python evaluate.py --run_ft --sizes small --mode umap
```

---

## 🛠️ Advanced Configuration Flags

* `--sizes`: Select global probe sizes to evaluate (`tiny`, `small`, `medium`, `big`). You can run multiple sizes sequentially.
* `--base_sizes`: Override probe sizes specifically for the Original Baseline model (Model Kiwi).
* `--ft_sizes`: Override probe sizes specifically for the Fine-Tuned model (Model Moana).
* `--web_export_extras`: Toggle the export of individual transparent class masks and pixel-wise grayscale confidence maps.
* `--web_target_classes`: Restrict extra visual assets only to a specified list of class names.
* `--web_include_report`: Enable saving of Matplotlib reports into the web folder.
* `--web_report_type`: Select which Matplotlib reports to include (supports multiple: `compare`, `compare_single`, `generate`, `heatmaps`, `all`).
* `--eval_w` and `--eval_h`: Change the image evaluation resolution (must be divisible by 16).
* `--dpi`: Set image export quality. Lower values speed up file writing (recommended range: 100 to 600).
* `--num_imgs`: Limit the number of images processed from your raw folder.

---

## 📊 Analytics & Reporting

The pipeline automatically calculates and logs the **Spread % (Percent Cover)** of each benthic class both per-image and globally across the entire batch (total survey coverage) at the end of execution. Bad uploads (corrupted images, PDFs) are safely intercepted and logged as errors in the JSON manifest without crashing the pipeline.

---

## 📄 License & Access

* **Pipeline Code:** The Python scripts, data engineering, and evaluation code developed by Neel Jani are open-source under the **MIT License**.
* **Model Weights:** The fine-tuned SeaDino-Seg-1 weights (derived from DINOv3) are maintained in a **Private** Hugging Face repository for internal server use and are governed by the **Meta DINO License**. 
* **Model Outputs:** The segmentation masks, CSV coverage tables, and JSON data generated by the pipeline are freely available for research and analysis.

**Attribution:**
* **Pipeline Development:** Neel Jani — [LinkedIn](https://www.linkedin.com/in/neel-jani-5a0173222/) | [GitHub](https://github.com/NeelJani1)
* **Data Collection & Annotation:** [Intelligent Vision Systems (IVS) Lab](https://www.ivslab.auckland.ac.nz/), University of Auckland (UoA), NZ
* **Hugging Face Hub:** [SeaDinoWeb/Algea_Segmentation_Model](https://huggingface.co/SeaDinoWeb/Algea_Segmentation_Model) (Gated/Private Access)
* **Backbone Reference:** [DINOv3 by Meta AI / FAIR](https://github.com/facebookresearch/dinov3)
