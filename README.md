# ECharts-SynChart: A Synthetic Dataset for Chart Classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19401089.svg)](https://doi.org/10.5281/zenodo.19401089)

This repository contains the code and dataset for the paper *"ECharts-SynChart: A Large-Scale Synthetic Dataset for Chart Type Classification"*.

## Dataset

- **Size**: 10,272 images, 24 chart types.
- **Source**: Generated from [ECharts](https://echarts.apache.org/) official examples with data augmentation (numerical perturbation, color variation, label replacement).
- **Download**: [https://doi.org/10.5281/zenodo.19401089](https://doi.org/10.5281/zenodo.19401089)  
  The dataset is accompanied by train/val/test CSV split files (included in this repository).
  The dataset is provided as a single compressed archive (`echarts-synchart-images.tar.gz`, ~486 MB). After downloading, extract the images; the archive contains an images/folder. Place this images/folder in the repository root (alongside README.md).

## Code Structure

- `scripts/` – Python scripts for data augmentation, training, and evaluation.
  - `train_classifier.py` – Train ResNet50 baseline model.
  - `plot_confusion_matrix.py` – Generate confusion matrix on test set.
  - `csv_to_jsonl.py` – Convert CSV labels to JSONL format (for Qwen-VL).
  - `unified_augment.py` – Data augmentation for ECharts JS files.
  - `generate_labels.py` – Extract chart type labels from JS files.
  - `merge_datasets.py` – Merge multiple PNG directories into one dataset.
  - `check_png.py` – Detect and optionally delete blank images.
  - `config.py` & `utils.py` – Configuration and helper functions.
- `node_scripts/` – Node.js scripts for rendering ECharts JS to PNG.
  - `render_echarts.js` – Render a JS file to PNG using Puppeteer.
- `requirements.txt` – Python dependencies.

## Quick Start

1. **Clone this repository**  
  ```bash
  git clone https://github.com/dfps-1234/echarts-synchart.git
  cd echarts-synchart
  ```
2. **Install dependencies**  
  ```bash
  pip install -r requirements.txt
  ```
3. **Download the dataset**  
   Download the archive from Zenodo and extract it:
   ```bash
   tar -xzf echarts-synchart-images.tar.gz      # 解压后得到 images/ 文件夹
   ```
   Place the images/folder in the root directory of this repository (alongside README.md). The default DATA_ROOT in train_classifier.py is set to'.', which expects the images/folder in the repository root.
   The CSV files (train.csv, val.csv, test.csv) are already included in the repository – you do not need to regenerate them.
4. **Train the baseline model**
   ```bash  
     # Ensure DATA_ROOT in train_classifier.py points to the directory containing images/ (e.g., DATA_ROOT = '.')
   python scripts/train_classifier.py --batch_size 64 --epochs 20
   ```  
5. **Generate confusion matrix**
   ```bash  
     # The confusion matrix script uses the same DATA_ROOT setting as train_classifier.py.
   python scripts/plot_confusion_matrix.py
   ```  

## Requirements
Python 3.8 or higher is required.
See requirements.txt for a full list. Main packages:

- torch >= 2.0.0
- torchvision >= 0.15.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.5.0
- pandas >= 1.5.0
- tqdm >= 4.65.0
- Pillow >= 9.0.0
- numpy >= 1.23.0

## Citation
If you use this code or dataset in your research, please cite:
```bibtex
@article{li2024echarts,
  title={ECharts-SynChart: A Large-Scale Synthetic Dataset for Chart Type Classification},
  author={Li, Yunzhe},
  journal={...},
  year={2024},
  doi={10.5281/zenodo.19401089}
}
```
## License
This project is licensed under the MIT License – see the LICENSE file for details.