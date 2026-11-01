# ♻️ Bottleneck Global Self-Attention for Efficient Waste Image Classification

A lightweight waste classifier built on **EfficientNet-B0 + Bottleneck Global Self-Attention (BoGSA)**, trained with **knowledge distillation** from an EfficientNetV2-S teacher. Only **5.15M parameters (~20.9 MB)**, small enough for smartphones and smart bins, with a working **web + mobile app**.

**Authors:** Nanda Vardhan Reddy, Darshan Gera, S Balasubramanian, Rushikeswara Reddy

**Affiliations:** Sri Sathya Sai Institute of Higher Learning (SSSIHL), Prasanthi Nilayam, A.P., India · SASTRA Deemed to be University

**Keywords:** waste classification, sustainability, knowledge distillation, attention mechanism, EfficientNet, edge deployment

> 📄 Paper link: _add after publication_ &nbsp;|&nbsp; 🌐 Live app: _add your Streamlit URL here_

---

## 📌 Highlights

- **BoGSA module**: shrinks the 7×7×1280 feature map to 320 channels *before* multi-head self-attention (4 heads), then restores 1280 channels with a residual connection. Global context at low cost.
- **Knowledge distillation**: EfficientNetV2-S teacher (training only) → EfficientNet-B0 + BoGSA student. The deployed model stays small.
- **Three benchmarks**: RealWaste (9 classes), Garbage Dataset (10 classes), Garbage Classification Dataset (12 classes).
- **Working application** for web and mobile browsers.

## 🧭 Proposed Pipeline

<p align="center">
  <img src="images/pipeline.png" alt="Proposed pipeline" width="100%">
</p>

1. **Input & pre-processing**: resize to 224 × 224, normalize, augment (horizontal flip, rotation, color jitter).
2. **Lightweight student**: EfficientNet-B0 backbone → BoGSA → classification head.
3. **Knowledge distillation (training only)**: teacher soft targets (T = 4.0), loss = CE (hard labels) + KL divergence (α = 0.7).
4. **Output & deployment**: trained student classifies waste in real time (plastic, paper, metal, glass, organic, etc.).

### BoGSA module

<p align="center">
  <img src="images/bogsa.png" alt="BoGSA module" width="85%">
</p>

(a) 1280 → 320 channel bottleneck, multi-head self-attention over 49 spatial tokens, back to 1280, residual add. (b) Inside the attention layer: Q, K, V projections for 4 heads (d = 80 each), scaled dot-product attention, concat, linear projection.

## 🗂️ Datasets

| Dataset | Classes | Images | Train / Val / Test (80/10/10) |
|---|---|---|---|
| RealWaste | 9 | 4,752 | 3,801 / 475 / 476 |
| Garbage Dataset (GD) | 10 | 12,259 | 9,807 / 1,225 / 1,227 |
| Garbage Classification Dataset | 12 | 15,515 | 12,412 / 1,551 / 1,552 |

Sources: [RealWaste](https://doi.org/10.3390/info14120633) · [Garbage Classification V2 (GD)](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) · [Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

<p align="center">
  <img src="images/sample_images.png" alt="Sample images" width="55%"><br>
  <em>Sample images. Top: RealWaste · Middle: Garbage Dataset · Bottom: Garbage Classification Dataset.</em>
</p>

## ⚙️ Experimental Setup

| Hyperparameter | Value |
|---|---|
| Student backbone | EfficientNet-B0 |
| Teacher | EfficientNetV2-S |
| Image size | 224 × 224 |
| Batch size | 32 |
| Epochs | 25 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Distillation T / α | 4.0 / 0.7 |
| BoGSA channels / heads | 320 / 4 |

Hardware: Kaggle (Linux), NVIDIA Tesla P100 16 GB, 29 GB RAM.

## 📊 Results

Mean ± std over three random seeds.

| Dataset | Classes | Teacher Val Acc (%) | Student Test Acc (%) | Macro F1 | Params (M) | FLOPs (M) | Size (MB) |
|---|---|---|---|---|---|---|---|
| RealWaste | 9 | 94.11 ± 0.19 | **94.12 ± 0.32** | 0.9426 ± 0.003 | 5.150 | 469.384 | 20.91 |
| Garbage Dataset | 10 | 96.90 ± 0.22 | **95.84 ± 0.19** | 0.9503 ± 0.003 | 5.151 | 469.385 | 20.92 |
| Garbage Classification | 12 | 98.71 ± 0.13 | **98.07 ± 0.18** | 0.9730 ± 0.002 | 5.153 | 469.388 | 20.93 |

### Comparison on the Garbage Dataset (10 classes)

| Model | Params (M) | Accuracy (%) |
|---|---|---|
| MobileNet | 3.5 | 68.89 |
| ResNet50 | 25.6 | 94.63 |
| ResNet101 | 44.5 | 94.46 |
| EfficientNetV2S | 20.5 | 95.13 |
| EfficientNetV2S (optimized) | 20.5 | 96.41 |
| EfficientNetV2M | 54.1 | 96.37 |
| **BoGSA + KD (Ours)** | **5.151** | **95.84** |

### Ablation (RealWaste)

| Variant | Test Acc (%) | Macro F1 | Params (M) | Size (MB) |
|---|---|---|---|---|
| EfficientNet-B0 only | 92.23 | 0.9251 | 4.02 | 16.37 |
| EfficientNet-B0 + KD (no BoGSA) | 92.44 | 0.92 | 4.02 | 16.37 |
| EfficientNet-B0 + BoGSA (no KD) | 92.86 | 0.9267 | 5.150 | 20.91 |
| **EfficientNet-B0 + BoGSA + KD (Ours)** | **94.12** | **0.9426** | **5.150** | **20.91** |
| EfficientNet-B0 + Full MHA + KD | 93.28 | 0.9325 | 10.58 | 42.63 |

### Training curves (teacher vs. student)

<table>
  <tr>
    <td align="center"><img src="images/acc_realwaste.png" width="100%"><br><sub>RealWaste</sub></td>
    <td align="center"><img src="images/acc_garbage_dataset.png" width="100%"><br><sub>Garbage Dataset</sub></td>
    <td align="center"><img src="images/acc_garbage_classification.png" width="100%"><br><sub>Garbage Classification</sub></td>
  </tr>
</table>

### Confusion matrices

<table>
  <tr>
    <td align="center"><img src="images/cm_realwaste.png" width="100%"><br><sub>RealWaste</sub></td>
    <td align="center"><img src="images/cm_garbage_dataset.png" width="100%"><br><sub>Garbage Dataset</sub></td>
    <td align="center"><img src="images/cm_garbage_classification.png" width="100%"><br><sub>Garbage Classification</sub></td>
  </tr>
</table>

### Grad-CAM visualizations

*Left: original image · Middle: Grad-CAM heatmap · Right: overlay with predicted class.*

<p align="center">
  <img src="images/gradcam.png" alt="Grad-CAM" width="70%">
</p>

### t-SNE feature embeddings

<p align="center">
  <img src="images/tsne.png" alt="t-SNE" width="100%"><br>
  <em>RealWaste · Garbage Dataset · Garbage Classification Dataset</em>
</p>

## 📱💻 Web & Mobile App

Upload an image or use the camera, choose the dataset/model (e.g. **GD, 10 classes**), tap **Classify**, and get the predicted class with per-class confidence. The layout is responsive, so the same app works in a desktop browser and on a phone.

<p align="center">
  <img src="images/app_screenshot.png" alt="Waste Classifier app" width="320"><br>
  <em>Waste Classifier app (mobile view): a cantaloupe photo is predicted as <b>biological</b> (100.0%).</em>
</p>

### Run locally

```bash
git clone https://github.com/Nanda89199/Waste_Management.git
cd Waste_Management
pip install -r requirements.txt
streamlit run app.py
```

> Replace `app.py` with the actual app filename in your repo.

## 🏗️ Repository Structure

```
Waste_Management/
├── images/          # figures used in this README
├── models/          # BoGSA + EfficientNet-B0 student, EfficientNetV2-S teacher
├── app.py           # Streamlit web/mobile app
├── requirements.txt
└── README.md
```

_Adjust to match your actual folders._

## 📝 Citation

```bibtex
@article{reddy2026bogsa,
  title   = {Bottleneck Global Self-Attention for Efficient Waste Image Classification},
  author  = {Reddy, Nanda Vardhan and Gera, Darshan and Balasubramanian, S and Reddy, Rushikeswara},
  year    = {2026}
}
```

## 🙏 Acknowledgments

We dedicate this work to Bhagawan Sri Sathya Sai Baba, Divine Founder Chancellor of Sri Sathya Sai Institute of Higher Learning, Prasanthi Nilayam, A.P., India.
