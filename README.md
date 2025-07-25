# SpectraGAN: Blind Quality Assessment for Super-Resolved Satellite Images

## Problem Statement

High-resolution satellite imagery can be used in urban planning, agriculture, and disaster management. However, acquiring high-resolution data from satellite platforms is an expensive operation due to resource constraints. Hence, satellites acquire multiple low-resolution images often shifted by half a pixel in both along and across track directions. These low-resolution images are utilized to generate high-resolution images. We want to provide super resolution models apt for low res- satellite imagery, that can capture high frequency changes of aerial images.

---

## Project Description

### What the App Does

SpectraGAN is a web-based application that allows users to upload or select low-res satellite images and receive a 4x scale high-res one. The app leverages a fine-tuned deep learning model to evaluate the perceptual quality and fidelity of super-resolved images, providing actionable feedback even in the absence of ground-truth high-resolution references.

![results.png](https://github.com/ParamAhuja/SpectraGAN/blob/main/backend/results/result.png)

### How the ML Model Was Developed and Trained

- **Model Architecture:**  
  The core of our solution is based on the ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) architecture, which is renowned for its ability to generate perceptually realistic high-resolution images from low-resolution inputs.
- **Dataset:**  
  We fine-tuned the ESRGAN model using the [SpectraGAN_DATA dataset](http://huggingface.co/datasets/ParamDev/SAT_ELITE_DATA), which contains thousands of satellite images in both low and high resolutions. This dataset is specifically curated for satellite image super-resolution tasks.
- **Training Process:**  
  The model was initially pre-trained on generic image datasets and then fine-tuned on SpectraGAN_DATA to adapt to the unique characteristics of satellite imagery. We employed data augmentation, perceptual loss, and adversarial training to ensure the model not only enhances resolution but also preserves critical features relevant to satellite analysis.
- **Quality Assessment:**  
  For blind quality assessment, we implemented a no-reference image quality evaluation module that leverages deep features to predict perceptual quality scores, enabling robust assessment without ground-truth images.

## Dataset

- **Name:** SpectraGAN_DATA
- **Source:** [Hugging Face Dataset Link](http://huggingface.co/datasets/ParamDev/SAT_ELITE_DATA)
- **Description:** Contains train and validation sets of satellite images for super-resolution tasks, including both low-res (sentinel2) and high-res (naip) pairs.

| Domain   | Source     | Spatial Resolution | Description                                  |
| -------- | ---------- | ------------------ | -------------------------------------------- |
| Low-Res  | Sentinel-2 | 10 meters/pixel    | Multispectral satellite imagery (RGB subset) |
| High-Res | NAIP       | 1 meter/pixel      | Aerial imagery from USDA's NAIP program      |

The dataset is organized to support **paired image super-resolution tasks**, where each Sentinel-2 patch (input) corresponds spatially and temporally to a high-resolution NAIP patch (target output).

---

## Live Demo

- [Hugging Face Spaces Deployment](https://huggingface.co/spaces/Rockerleo/esrgan)

<table> <tr> <td><img src="https://github.com/ParamAhuja/SAT_ELITE/blob/main/backend/LR/baboon.png" alt="Low-res" height=300px/></td> <td><img src="https://github.com/ParamAhuja/SAT_ELITE/blob/main/backend/results/baboon.png" alt="result" height=300px/></td> </tr> </table>

---

### How It Integrates Into the App

- The backend hosts the fine-tuned ESRGAN model and the blind quality assessment module.
- The frontend provides a user-friendly interface for uploading images, running super-resolution, and viewing quality scores.
- When a user submits an image, the backend processes it through the ESRGAN model to generate a super-resolved version, then evaluates its quality using the blind assessment module.
- Results, including the enhanced image and its quality score, are displayed in the web app.

### The Problem It Solves and Our Approach

**Problem:**  
Satellite imagery is vital for numerous applications, but acquiring high-resolution images is costly and often infeasible. Super-resolution algorithms can generate high-res images from low-res inputs, but assessing their quality is difficult without ground-truth references.

**Our Approach:**  
We address this by:

- Fine-tuning a state-of-the-art super-resolution model (ESRGAN) on a domain-specific satellite dataset.
- Integrating a blind quality assessment module to evaluate the perceptual and structural fidelity of generated images.
- Providing an accessible web interface for researchers and practitioners to enhance and assess satellite images without the need for ground-truth data.

---

## Repository

- [GitHub Repository](https://github.com/ParamAhuja/SpectraGAN)

---

## Setup and Deployment Instructions

### One-Click Deployment

A live demo is available at:  
[https://huggingface.co/spaces/Rockerleo/esrgan](https://huggingface.co/spaces/Rockerleo/esrgan)

### Frontend Setup

1. **Navigate to the Frontend Directory:**

   ```bash
   cd ../frontend
   ```

2. **Install Frontend Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Frontend App:**

   ```bash
   python app.py
   ```

4. **Access the App:**
   - Open your browser and go to `<http://127.0.0.1:7860/>` (or the port specified in your app).

---

### Prerequisites for inference

- Python 3.8+
- pip (Python package manager)
- CUDA-enabled GPU for faster inference

### Replicate Results in your script

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/ParamAhuja/SpectraGAN.git
   cd SpectraGAN/backend
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pretrained Weights:**
   - Download the fine-tuned ESRGAN weights and place them in the `models/` directory. (Instructions or direct link can be provided here if available.)

4. **Run the Backend Server:**

   ```bash
   python test.py
   ```

   *(Or specify the actual backend entry point if different, e.g., `python test.py` or a FastAPI/Flask server script.)*

## License

- Dataset: MIT License
- Code: MIT License

---

## Team & Acknowledgements

- [ParamAhuja](https://github.com/ParamAhuja), [AngelGupta](https://github.com/AngelGupta13), [MohitGoyal](https://github.com/MohitGoyal)
