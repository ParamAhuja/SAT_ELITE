# SAT_ELITE: Blind Quality Assessment for Super-Resolved Satellite Images

## Problem Statement

Acquiring high-resolution data from satellite platforms is an expensive operation due to resource constraints. As a result, satellites often capture multiple low-resolution images, each shifted by half a pixel in both along and across track directions. These low-resolution images are then used to generate high-resolution images through super-resolution algorithms. However, the quality of these super-resolved images depends on various satellite system parameters, including the chosen super-resolution algorithm. Critically, assessing the quality of these generated images is challenging due to the absence of ground-truth references. This necessitates the use of blind (no-reference) quality assessment techniques that can evaluate both the perceptual realism and fidelity of super-resolved images.

---

## Project Description

### What the App Does

SAT_ELITE is a web-based application that allows users to upload or select super-resolved satellite images and receive a blind (no-reference) quality assessment. The app leverages a fine-tuned deep learning model to evaluate the perceptual quality and fidelity of super-resolved images, providing actionable feedback even in the absence of ground-truth high-resolution references.

### How the ML Model Was Developed and Trained

- **Model Architecture:**  
  The core of our solution is based on the ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) architecture, which is renowned for its ability to generate perceptually realistic high-resolution images from low-resolution inputs.
- **Dataset:**  
  We fine-tuned the ESRGAN model using the [SAT_ELITE_DATA dataset](http://huggingface.co/datasets/ParamDev/SAT_ELITE_DATA), which contains thousands of satellite images in both low and high resolutions. This dataset is specifically curated for satellite image super-resolution tasks.
- **Training Process:**  
  The model was initially pre-trained on generic image datasets and then fine-tuned on SAT_ELITE_DATA to adapt to the unique characteristics of satellite imagery. We employed data augmentation, perceptual loss, and adversarial training to ensure the model not only enhances resolution but also preserves critical features relevant to satellite analysis.
- **Quality Assessment:**  
  For blind quality assessment, we implemented a no-reference image quality evaluation module that leverages deep features to predict perceptual quality scores, enabling robust assessment without ground-truth images.

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

## Dataset

- **Name:** SAT_ELITE_DATA
- **Source:** [Hugging Face Dataset Link](http://huggingface.co/datasets/ParamDev/SAT_ELITE_DATA)
- **Description:** Contains thousands of satellite images for super-resolution tasks, including both low- and high-resolution pairs.

---

## Live Demo

- [Hugging Face Spaces Deployment](https://huggingface.co/spaces/Rockerleo/esrgan)

---

## Repository

- [GitHub Repository](https://github.com/ParamAhuja/SAT_ELITE)

---

## Setup and Deployment Instructions

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- (Optional) CUDA-enabled GPU for faster inference

### Backend Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/ParamAhuja/SAT_ELITE.git
   cd SAT_ELITE/backend
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pretrained Weights:**
   - Download the fine-tuned ESRGAN weights and place them in the `models/` directory. (Instructions or direct link can be provided here if available.)

4. **Run the Backend Server:**

   ```bash
   python app.py
   ```

   *(Or specify the actual backend entry point if different, e.g., `python test.py` or a FastAPI/Flask server script.)*

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
   - Open your browser and go to `http://localhost:5000` (or the port specified in your app).

### One-Click Deployment

A live demo is available at:  
[https://huggingface.co/spaces/Rockerleo/esrgan](https://huggingface.co/spaces/Rockerleo/esrgan)

---

## License

- Dataset: MIT License
- Code: [Specify your code license here, e.g., MIT, Apache 2.0, etc.]

---

## Team & Acknowledgements

- [List your team members, contributors, and any acknowledgements]

---

## References

- [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)
- [SAT_ELITE_DATA on Hugging Face](http://huggingface.co/datasets/ParamDev/SAT_ELITE_DATA)
- [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Rockerleo/esrgan)
- [GitHub Repository](https://github.com/ParamAhuja/SAT_ELITE)
