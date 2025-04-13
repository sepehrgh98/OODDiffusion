<h1 align="center">
  <i><font color="#4A7EBB">OODDiffusion</font></i>
</h1>

<p align="">
  <b>A Deep Diffusion-based Blind Image Super-Resolution Framework with Out-of-distribution Detection and Controllable Sampling (Submitted to IEEE Transaction on Computational Imaging 2025)</b>

  <b>Sepehr Ghamari*, Alireza Esmaeilzehi†, M. Omair Ahmad*, M.N.S. Swamy*<b>

  <b>* Electrical and Computer Engineering Department, Concordia University<b>

  <b>† The Edward S. Rogers Sr. Department of Electrical and Computer Engineering, University of Toronto<b>
</p>

<p align="left">
  <a href="https://arxiv.org/abs/your_arxiv_id">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" />
  </a>
  <a href="https://colab.research.google.com/github/your_repo_link">
    <img src="https://img.shields.io/badge/Open%20in-Colab-blue?logo=google-colab" />
  </a>
    <a href="https://huggingface.co/sepehrgh98/OODDiffusion">
    <img src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface" />
    </a>
</p>

<p align="center">
  <img src="assets/fig2.png" width="90%"/>
</p>
<p align="center"><i>
The proposed blind image super resolution scheme. It consists of seven modules: image artifact reduction module, out-of-distribution detection module, encoder, denoiser, decoder, condition network, and controllable sampling process module. (a) Overall architecture. (b) The architecture of the out-of-distribution detection module. (c) Controllable sampling process module.
</i></p>

<p align=""><b>⭐️ If <code>OODDiffusion</code> is helpful for you, please help star this repo. Thanks! 😊</b></p>


## 📚 Table of Contents

- [🆕 Update](#update)
- [📸 Visual Results](#-visual-results-on-real-world-images)
<!-- - [🧠 TODO](#-todo) -->
- [⚙️ Installation](#-installation)
<!-- - [🚀 Quick Start](#-quick-start) -->
- [🎯 Pretrained Models](#-pretrained-models)
- [🏋️ Train](#-train)
- [🧪 Inference](#-inference)


## 🆕 Update

- **2025.04.11** ✅ Initial public release of **OODDiffusion** with full codebase, pretrained checkpoints, and test pipeline. 
- **2025.03.28** ✅ Finalized manuscript submitted to *IEEE Transactions on Computational Imaging*. View [paper](https://www.google.com/search?q=OODDiffusion+IEEE+TCI).
- **2025.03.05** ✅ First draft of manuscript is finalized.
- **2025.02.02** ✅ Comparitive study with the state-of-the-art methods has been added!
- **2025.01.10** ✅ Added support for adaptive noise scheduling: `linear`, `sigmoid`, and `Gaussian` with controllable steepness.
- **2024.12.21** ✅ Inference stage of the network has been added.
- **2024.12.07** ✅ Weighted beta scheduler approach has been added to controllable sampling process module.
- **2024.11.27** ✅ The Controllable sampling process module is created.
- **2024.11.10** ✅ Trained and integrated **OOD Detection Module** using Decomposed Confidence + ResNet50 backbone.
- **2024.11.04** ✅ Added OOD detection module, histogram visualizations, cosine similarity analysis, and OOD confidence metrics.


## 👀 Visual Results On Real-world Images
<p align="center">
  <img src="assets/OOD.png"/>
</p>

## 📦 Installation

```bash
# clone this repo
git clone https://github.com/sepehrgh98/OODDiffusion.git
cd OODDiffusion

# create environment
conda create -n ooddiffusion python=3.9
conda activate ooddiffusion
pip install -r requirements.txt
```

## 🧠 Pretrained Models

Here we list pretrained weight of stage 2 model (IRControlNet) and our trained SwinIR, which was used for degradation removal during the training of stage 2 model.

| **Model Name**                  | **Description**                                               | **HuggingFace**        | 
|----------------------------------|---------------------------------------------------------------|-------------------------|
| `IAR.pt`                        | Image Artifact Reduction Module | [download](https://huggingface.co/sepehrgh98/OODDiffusion/blob/main/IAR.pt)           | 
| `OODDetector.pth`                         | Out-of-distribution Detection Module                   | [download](https://huggingface.co/sepehrgh98/OODDiffusion/blob/main/OODDetector.pt)           | 
| `OODDiffusion_controlnet.pt`       | IRControlNet trained on DrealSR & DIV2K  | [download](https://huggingface.co/sepehrgh98/OODDiffusion/blob/main/OODDiffusion_controlnet.pt)           | 
| `v2-1_512-ema-pruned.ckpt`                    | Diffusion Model | [download](https://huggingface.co/sepehrgh98/OODDiffusion/blob/main/v2-1_512-ema-pruned.ckpt)           | 


## 🏋️ Train

### Stage 1

First, we fine-tune **DiffBIR** on our target dataset. This forms the backbone of our method and provides the initial super-resolution capability.

To fine-tune **DiffBIR**, please follow the two-stage training procedure described in the [official DiffBIR repository](https://github.com/XPixelGroup/DiffBIR):

1. **Stage I**: Train the SwinIR-based artifact reduction module.
2. **Stage II**: Train the ControlNet module for conditional generation.

At the end of this step, you should obtain:
- A SwinIR checkpoint from Stage I (artifact removal module)
- A ControlNet checkpoint from Stage II (conditioning network)

### Stage 2

In this stage, we train the **Out-of-Distribution (OOD) Detection Module**, which enables the model to distinguish between known (in-distribution) and unknown (out-of-distribution) degradations.

---

1. **Prepare the dataset directory** with the following structure:

```
dataset_dir/
  ├── dataset1/
  │   ├── lq/        # Low-quality (degraded) images
  │   └── hq/        # High-quality ground truth
  ├── dataset2/
  │   ├── lq/
  │   └── hq/
```

> Ensure each `lq/` and `hq/` pair is properly aligned across datasets. These will be used to contrast in-distribution vs out-of-distribution samples during training.

---

2. **Start training** the OOD Detection Module:

```bash
python -u train.py --config ./config/train.yaml
```

> 🧠 The training script will learn to classify inputs based on degradation type, enabling downstream modules to adapt their behavior accordingly (e.g., noise scheduling).

## 🧪 Inference

To run inference using **OODDiffusion**, follow the steps below:

1. **Download the required checkpoints**:
   - SwinIR (artifact removal)
   - CLDM (conditioning module)
   - OOD Detector
   - ControlNet

   Make sure the paths to these checkpoints are correctly set in [`config/inference.yaml`](./config/inference.yaml).

2. **Run inference** with the following command:

```bash
python -u inference.py \
  --better_start \
  --guidance \
  --strength 1.0 \
  --g_loss w_mse \
  --g_scale 0.5 \
  --g_space rgb \
  --upscale 4 \
  --cfg_scale 4.0 \
  --config ./config/inference.yaml
```

> 💡 This command applies OOD-aware guidance during inference using cosine similarity and a weighted MSE loss in RGB space.


## 📖 Citation

Please cite us if our work is useful for your research.

```bibtex
@article{ghamari2025ooddiffusion,
  title     = {OODDiffusion: A Deep Diffusion-based Blind Image Super Resolution Scheme using Out-of-distribution Detection and Controllable Sampling Process},
  author    = {Sepehr Ghamari and Alireza Esmaeilzehi and M. Omair Ahmad and M.N.S. Swamy},
  journal   = {IEEE Transactions on Computational Imaging},
  year      = {2025},
  note      = {Submitted},
}
```


## 🙏 Acknowledgement

This project builds upon the foundation of [DiffBIR](https://github.com/XPixelGroup/DiffBIR). We sincerely thank the original authors for their outstanding work and contributions to the community.

This work was supported in part by the **Natural Sciences and Engineering Research Council of Canada (NSERC)**  
and in part by the **Regroupement Stratégique en Microélectronique du Québec (ReSMiQ)**.


## 📬 Contact

If you have any questions or feedback, feel free to contact us at:

📧 se_gham@encs.concordia.ca 

📧 sepehrghamri@gmail.com