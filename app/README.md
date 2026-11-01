# Bottleneck Global Self-Attention for Efficient Waste Image Classification

**Authors:** Nanda Vardhan Reddy, Darshan Gera, S Balasubramanian, Rushikeswara Reddy
*Sri Sathya Sai Institute of Higher Learning · SASTRA Deemed to be University*

**Keywords:** waste classification · sustainability · knowledge distillation · attention mechanism · EfficientNet · edge deployment

A lightweight waste classifier (**5.15 M parameters, ~20.9 MB**) built on an **EfficientNet-B0** backbone with a **Bottleneck Global Self-Attention (BoGSA)** module, trained by **knowledge distillation** from an **EfficientNetV2-S** teacher. It ships with a **web and mobile app** that classifies waste from an uploaded or camera-captured image.

| Dataset | Classes | Test Acc. (%) | Macro F1 |
|---|---|---|---|
| RealWaste | 9 | **94.12** | 0.9426 |
| Garbage Dataset (GD) | 10 | **95.84** | 0.9503 |
| Garbage Classification Dataset | 12 | **98.07** | 0.9730 |

---

## Live Demo (Web & Mobile)

The app runs in any modern browser, on desktop and on phones. Pick a dataset/model, upload an image (or use the camera on mobile), tap **Classify**, and get the predicted class with per-class confidence bars.

🔗 **Live app:** `<ADD YOUR APP URL HERE>`

<p align="center">
  <img src="images/app_mobile.png" alt="Waste Classifier web and mobile app" width="300">
</p>

*Waste Classifier app: dataset/model selector, Upload / Camera tabs, Classify button and confidence bars (shown here on a mobile screen).*

---

## Proposed Framework

The framework has four stages: (1) input and pre-processing (224 × 224, normalize, augment), (2) the lightweight student (EfficientNet-B0 + BoGSA + classification head), (3) knowledge distillation from the EfficientNetV2-S teacher (training only), and (4) output and real-time deployment.

<p align="center">
  <img src="images/pipeline.png" alt="General pipeline of the proposed lightweight waste-classification framework">
</p>

### Bottleneck Global Self-Attention (BoGSA)

The 7 × 7 × 1280 backbone features are projected to 320 channels, processed as 49 tokens by 4-head self-attention (d<sub>k</sub> = 80), projected back to 1280 channels and added to the input through a residual connection. Reducing channels *before* attention keeps global context cheap: BoGSA adds ~1.1 M parameters, whereas full MHA adds ~6.6 M and is less accurate (see ablation).

<p align="center">
  <img src="images/bogsa.png" alt="BoGSA module structure">
</p>

### Knowledge Distillation

The teacher is used **only during training**, so the deployed model stays small. Loss = cross-entropy on hard labels + KL divergence between teacher and student soft targets (T = 4.0, α = 0.7).

---

## Datasets

| Dataset | Classes | Images | Train (80%) | Val (10%) | Test (10%) |
|---|---|---|---|---|---|
| [RealWaste](https://doi.org/10.3390/info14120633) | 9 | 4,752 | 3,801 | 475 | 476 |
| [Garbage Dataset (GD)](https://doi.org/10.48550/arXiv.2602.10500) | 10 | 12,259 | 9,807 | 1,225 | 1,227 |
| [Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) | 12 | 15,515 | 12,412 | 1,551 | 1,552 |

<p align="center">
  <img src="images/sample_images.png" alt="Sample images" width="520">
</p>

*Top row: RealWaste · middle row: Garbage Dataset · bottom row: Garbage Classification Dataset.*

---

## Experimental Setup

| Hyperparameter | Value |
|---|---|
| Student backbone | EfficientNet-B0 |
| Teacher | EfficientNetV2-S |
| Image size | 224 × 224 |
| Batch size | 32 |
| Epochs | 25 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Distillation temperature (T) | 4.0 |
| Distillation alpha (α) | 0.7 |
| BoGSA channels / heads | 320 / 4 |
| Augmentation | horizontal flip, rotation, color jitter |

Trained on Kaggle (Linux, NVIDIA Tesla P100 16 GB, 29 GB RAM, CUDA).

---

## Results

Mean ± std over three random seeds.

| Dataset | Teacher Val. Acc. (%) | Student Val. Acc. (%) | Student Test Acc. (%) | Macro F1 | Params (M) | FLOPs (M) | Size (MB) |
|---|---|---|---|---|---|---|---|
| RealWaste | 94.11 ± 0.19 | 93.68 ± 0.24 | 94.12 ± 0.32 | 0.9426 ± 0.003 | 5.150 | 469.4 | 20.91 |
| Garbage Dataset | 96.90 ± 0.22 | 96.41 ± 0.19 | 95.84 ± 0.19 | 0.9503 ± 0.003 | 5.151 | 469.4 | 20.92 |
| Garbage Classification | 98.71 ± 0.13 | 98.07 ± 0.13 | 98.07 ± 0.18 | 0.9730 ± 0.002 | 5.153 | 469.4 | 20.93 |

### Training curves (teacher vs. student, 25 epochs)

<p align="center">
  <img src="images/curves_realwaste.png" width="32%" alt="RealWaste curves">
  <img src="images/curves_garbage_dataset.png" width="32%" alt="Garbage Dataset curves">
  <img src="images/curves_garbage_classification.png" width="32%" alt="Garbage Classification curves">
</p>

*Left to right: RealWaste (9 classes), Garbage Dataset (10 classes), Garbage Classification Dataset (12 classes).*

### Ablation (RealWaste)

| Variant | Test Acc. (%) | Macro F1 | Params (M) | Size (MB) |
|---|---|---|---|---|
| EfficientNet-B0 only | 92.23 | 0.9251 | 4.02 | 16.37 |
| EfficientNet-B0 + KD (no BoGSA) | 92.46 | 0.9200 | 4.02 | 16.37 |
| EfficientNet-B0 + BoGSA (no KD) | 92.86 | 0.9267 | 5.150 | 20.91 |
| **EfficientNet-B0 + BoGSA + KD (Ours)** | **94.12** | **0.9426** | **5.150** | **20.91** |
| EfficientNet-B0 + Full MHA + KD | 93.28 | 0.9325 | 10.58 | 42.63 |

BoGSA and KD are complementary: together they give +1.89 points over the backbone, more than the sum of their individual gains (+0.63 and +0.23). Full MHA doubles the parameters and is still less accurate.

### Comparison with the state of the art

| Dataset | Method | Params (M) | Test Acc. (%) |
|---|---|---|---|
| RealWaste | Inception V3 (Single et al.) | – | 89.19 |
| RealWaste | **Ours** | **5.15** | **94.12** |
| Garbage Dataset | EfficientNetV2S (Kunwar) | 20.5 | 95.13 |
| Garbage Dataset | **Ours** | **5.15** | **95.84** |
| Garbage Classification | ResNet-based CNN (Ahmad et al.) | – | 98.16 |
| Garbage Classification | **Ours** | **5.15** | 98.07 |

On the Garbage Dataset, the model beats EfficientNetV2S (20.5 M) and is within 0.6 points of the optimized EfficientNetV2S (96.41%, ~4× larger) and EfficientNetV2M (96.37%, ~10× larger). Literature numbers come from the cited papers and may use different splits.

### Confusion matrices

<p align="center">
  <img src="images/cm_realwaste.png" width="32%" alt="RealWaste confusion matrix">
  <img src="images/cm_garbage_dataset.png" width="32%" alt="Garbage Dataset confusion matrix">
  <img src="images/cm_garbage_classification.png" width="32%" alt="Garbage Classification confusion matrix">
</p>

On RealWaste the largest confusions are Plastic ↔ Metal/Glass and Textile Trash ↔ Miscellaneous Trash. Batteries are handled reliably (75/75 on GD, 110/113 on the 12-class set), which matters because they need a separate collection route.

### Grad-CAM

*Activation maps (original · Grad-CAM heatmap · overlay with predicted class) show the model focusing on the battery body, the mask surface and the object shape.*

<p align="center">
  <img src="images/gradcam.png" alt="Grad-CAM visualizations" width="720">
</p>

---

## Getting Started

```bash
git clone https://github.com/Nanda89199/Waste_Management.git
cd Waste_Management
pip install -r app/requirements.txt
```

Run the web/mobile app locally:

```bash
streamlit run app/app.py
```

Then open the shown URL in a desktop browser, or on a phone connected to the same network.

> Adjust the file names above (`app/requirements.txt`, `app/app.py`) to match your repository.

---

## Citation

```bibtex
@misc{reddy2026bogsa,
  title  = {Bottleneck Global Self-Attention for Efficient Waste Image Classification},
  author = {Reddy, Nanda Vardhan and Gera, Darshan and Balasubramanian, S and Reddy, Rushikeswara},
  year   = {2026},
  note   = {Source code: https://github.com/Nanda89199/Waste_Management}
}
```

## Acknowledgments

Sri Sathya Sai Institute of Higher Learning, and the authors of the RealWaste, Garbage Dataset and Garbage Classification datasets.
