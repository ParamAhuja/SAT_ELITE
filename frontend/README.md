---
title: Real-ESRGAN Dual-Mode Image Upscaler
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
license: mit
---
# 🖼️ Real-ESRGAN Dual-Mode Image Upscaler

A lightweight Gradio web app to upscale any image using the Real-ESRGAN model. Simply upload your photo, choose either **Standard Upscale** (×4) or **Premium Upscale** (×8), and download the upscaled image.

---

## 📑 Table of Contents

1. [Features](#features)  
2. [Project Structure](#project-structure)  
3. [Prerequisites](#prerequisites)  
4. [Installation](#installation)  
5. [Running Locally](#running-locally)  
6. [Usage](#usage)  
7. [Contributing](#contributing)  
8. [License](#license)  
9. [Author & Credits](#author--credits)

---

## ✨ Features

- **Standard Upscale (×4)**  
  Enhance image resolution by 4x for clearer and larger images.

- **Premium Upscale (×8)**  
  Upscales first to 4x and then resizes using bicubic interpolation for even higher resolution (8x).

- **Live Preview**  
  See your original and upscaled images side by side before downloading.

- **Instant Download**  
  Export the upscaled image as a PNG and use it immediately.

---

## 📁 Project Structure

```
upscale-project/
├── app.py 
├── requirements.txt 
├── .gitattributes 
├── .gitignore 
└── README.md 
```

---

## ⚙️ Prerequisites

- Python 3.10 or higher  
- `git`  
- A terminal / command prompt  

---

## 🔧 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/ParamAhuja/SAT_ELITE.git
   cd SAT_ELITE/frontend
   ```

2. Create and activate a virtual environment:

   ```bash
    python -m venv .venv
    source .venv/bin/activate   # Linux/macOS
    .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running Locally

Launch the app on your machine:

   ```bash
   python app.py
   ```

By default, it will start on <http://127.0.0.1:7860/>. Open that URL in your browser to access the interface.

## 🎯 Usage

1. **Upload Photo** via the left panel.  
2. **Choose a Mode**:  
   - Click **Standard Upscale (×4)** for a 4x resolution increase.
   - Click **Premium Upscale (×8)** for an 8x resolution increase.
3. Preview your result on the right side.
4. Click **Download PNG** to save the upscaled image.
