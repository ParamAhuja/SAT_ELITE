import os
import math
import uuid
import numpy as np
import onnxruntime as ort
from PIL import Image
import gradio as gr
import tempfile
import requests


MODEL_DIR     = "model"
MODEL_X2_PATH = os.path.join(MODEL_DIR, "Real-ESRGAN_x2plus.onnx")
MODEL_X4_PATH = os.path.join(MODEL_DIR, "Real-ESRGAN-x4plus.onnx")


FILE_ID_X2 = "15xmXXZNH2wMyeQv4ie5hagT7eWK9MgP6"
FILE_ID_X4 = "1wDBHad9RCJgJDGsPdapLYl3cr8j-PMJ6"

def download_from_drive(file_id: str, dest_path: str):
  
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    
    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print(f"Model telah diunduh dan disimpan di {dest_path}")
    return dest_path


if not os.path.isfile(MODEL_X2_PATH):
    download_from_drive(FILE_ID_X2, MODEL_X2_PATH)

# Unduh model ×4
if not os.path.isfile(MODEL_X4_PATH):
    download_from_drive(FILE_ID_X4, MODEL_X4_PATH)


sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 2
sess_opts.inter_op_num_threads = 2

session_x2 = ort.InferenceSession(MODEL_X2_PATH, sess_options=sess_opts, providers=["CPUExecutionProvider"])
session_x4 = ort.InferenceSession(MODEL_X4_PATH, sess_options=sess_opts, providers=["CPUExecutionProvider"])


input_meta_x2 = session_x2.get_inputs()[0]
_, _, H_in_x2, W_in_x2 = tuple(input_meta_x2.shape)
H_in_x2, W_in_x2 = int(H_in_x2), int(W_in_x2)

input_meta_x4 = session_x4.get_inputs()[0]
_, _, H_in_x4, W_in_x4 = tuple(input_meta_x4.shape)
H_in_x4, W_in_x4 = int(H_in_x4), int(W_in_x4)


dummy_x2 = np.zeros((1, 3, H_in_x2, W_in_x2), dtype=np.float32)
dummy_out_x2 = session_x2.run(None, {input_meta_x2.name: dummy_x2})[0]
_, _, H_out_x2, W_out_x2 = dummy_out_x2.shape
SCALE_X2 = H_out_x2 // H_in_x2
if SCALE_X2 != 2:
    raise RuntimeError(f"Model ×2 menghasilkan scale = {SCALE_X2}, bukan 2")

dummy_x4 = np.zeros((1, 3, H_in_x4, W_in_x4), dtype=np.float32)
dummy_out_x4 = session_x4.run(None, {input_meta_x4.name: dummy_x4})[0]
_, _, H_out_x4, W_out_x4 = dummy_out_x4.shape
SCALE_X4 = H_out_x4 // H_in_x4
if SCALE_X4 != 4:
    raise RuntimeError(f"Model ×4 menghasilkan scale = {SCALE_X4}, bukan 4")


def run_tile_x2(tile_np: np.ndarray) -> np.ndarray:
    patch_nchw = np.transpose(tile_np, (2, 0, 1))[None, ...]
    out_nchw = session_x2.run(None, {input_meta_x2.name: patch_nchw})[0]
    out_nchw = np.squeeze(out_nchw, axis=0)
    out_hwc  = np.transpose(out_nchw, (1, 2, 0))
    return out_hwc

def run_tile_x4(tile_np: np.ndarray) -> np.ndarray:
    patch_nchw = np.transpose(tile_np, (2, 0, 1))[None, ...]
    out_nchw = session_x4.run(None, {input_meta_x4.name: patch_nchw})[0]
    out_nchw = np.squeeze(out_nchw, axis=0)
    out_hwc  = np.transpose(out_nchw, (1, 2, 0))
    return out_hwc


def tile_upscale(input_img: Image.Image, scale: int, max_dim=1024):
    if scale == 2:
        H_in, W_in, run_tile, SCALE = H_in_x2, W_in_x2, run_tile_x2, SCALE_X2
    else:
        H_in, W_in, run_tile, SCALE = H_in_x4, W_in_x4, run_tile_x4, SCALE_X4

    
    w, h = input_img.size
    if w > max_dim or h > max_dim:
        scale_factor = max_dim / float(max(w, h))
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        input_img = input_img.resize((new_w, new_h), Image.LANCZOS)

    img_rgb = input_img.convert("RGB")
    arr = np.array(img_rgb).astype(np.float32) / 255.0
    h_orig, w_orig, _ = arr.shape

    tiles_h = math.ceil(h_orig / H_in)
    tiles_w = math.ceil(w_orig / W_in)
    pad_h = tiles_h * H_in - h_orig
    pad_w = tiles_w * W_in - w_orig

    arr_padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    out_h = tiles_h * H_in * SCALE
    out_w = tiles_w * W_in * SCALE
    out_arr = np.zeros((out_h, out_w, 3), dtype=np.float32)

    for i in range(tiles_h):
        for j in range(tiles_w):
            y0, x0 = i * H_in, j * W_in
            tile = arr_padded[y0:y0+H_in, x0:x0+W_in, :]
            up_tile = run_tile(tile)
            oy0, ox0 = i * H_in * SCALE, j * W_in * SCALE
            out_arr[oy0:oy0 + H_in * SCALE, ox0:ox0 + W_in * SCALE, :] = up_tile

    final_arr = out_arr[0:h_orig * SCALE, 0:w_orig * SCALE, :]
    final_arr = np.clip(final_arr, 0.0, 1.0)
    final_uint8 = (final_arr * 255.0).round().astype(np.uint8)
    final_pil = Image.fromarray(final_uint8)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    final_pil.save(tmp.name, format="PNG")
    tmp.close()
    return final_pil, tmp.name


def upscale_x2(input_img: Image.Image):
    return tile_upscale(input_img, scale=2)

def standard_upscale(input_img: Image.Image):
    return tile_upscale(input_img, scale=4)

def premium_upscale(input_img: Image.Image):
    final_4x, _ = tile_upscale(input_img, scale=4)
    w_orig, h_orig = input_img.size
    target_size = (w_orig * 8, h_orig * 8)
    final_8x = final_4x.resize(target_size, resample=Image.LANCZOS)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    final_8x.save(tmp.name, format="PNG")
    tmp.close()
    return final_8x, tmp.name


css = """
#x2-btn {
    background-color: lightgreen !important;
    color: black !important;
}
#premium-btn {
    background-color: gold !important;
    color: black !important;
}
"""


with gr.Blocks(css=css, title="Real-ESRGAN Triple-Mode Upscaler") as demo:
    gr.Markdown(
        """
        # Real-ESRGAN Upscaler  
        **Upscale (×2)**, **Standard Upscale (×4)** atau **Premium Upscale 🚀 (×8)**.  
        
        """
    )

    with gr.Row():
        inp_image = gr.Image(type="pil", label="Upload Source Image")

    with gr.Row():
        btn_x2   = gr.Button("Upscale (×2)", elem_id="x2-btn")
        btn_std  = gr.Button("Standard Upscale (×4)", variant="primary", elem_id="std-btn")
        btn_prem = gr.Button("Premium Upscale 🚀 (×8)", elem_id="premium-btn")

    out_preview  = gr.Image(type="pil", label="Upscaled Preview")
    out_download = gr.DownloadButton("⬇️ Download PNG", visible=True)

    btn_x2.click(fn=upscale_x2, inputs=inp_image, outputs=[out_preview, out_download])
    btn_std.click(fn=standard_upscale, inputs=inp_image, outputs=[out_preview, out_download])
    btn_prem.click(fn=premium_upscale, inputs=inp_image, outputs=[out_preview, out_download])

demo.launch(server_name="0.0.0.0", server_port=7860)