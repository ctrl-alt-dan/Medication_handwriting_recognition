import json
import os
import numpy as np
import pandas as pd
import torch

import cv2
from PIL import Image, ImageOps
from PIL import ImageDraw


# -----------------------------
# IO helpers
# -----------------------------
def read_pil_from_streamlit_file(st_file) -> Image.Image:
    """
    Reads Streamlit UploadedFile / camera_input to PIL and fixes phone EXIF orientation.
    """
    img = Image.open(st_file)
    img = ImageOps.exif_transpose(img)  # crucial for phone photos
    return img.convert("RGB")


def load_labels(labels_path: str):
    """
    Accepts:
      - classes.json : ["label0","label1",...]
      - idx_to_class.json : {"0":"label0","1":"label1",...}
    Returns: list[str] ordered by class index.
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}\n"
            "Export classes.json from your training notebook and place it next to app.py."
        )

    with open(labels_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        # keys might be strings
        pairs = [(int(k), v) for k, v in obj.items()]
        pairs.sort(key=lambda x: x[0])
        return [v for _, v in pairs]

    raise ValueError("Labels JSON must be a list or dict.")


# -----------------------------
# Rotate / crop helpers
# -----------------------------
def apply_rotate(pil: Image.Image, degrees: int) -> Image.Image:
    if degrees == 0:
        return pil
    # expand so we don't clip corners; fill with white
    return pil.rotate(degrees, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))



from PIL import ImageDraw

def draw_crop_guidance_overlay(pil: Image.Image, target_aspect: float = 384/64, margin_frac: float = 0.08) -> Image.Image:
    """
    Returns a copy of the image with a centered rectangle hint showing the recommended crop region.
    - target_aspect: width/height aspect ratio of the model input (default 384/64).
    - margin_frac: fraction of min(image_w, image_h) used as margin around the rectangle.
    """
    img = pil.copy().convert("RGBA")
    w, h = img.size
    m = int(round(min(w, h) * margin_frac))

    # available region after margins
    aw, ah = max(w - 2*m, 1), max(h - 2*m, 1)
    avail_aspect = aw / ah

    if avail_aspect >= target_aspect:
        # limited by height
        rect_h = ah
        rect_w = int(round(rect_h * target_aspect))
    else:
        # limited by width
        rect_w = aw
        rect_h = int(round(rect_w / target_aspect))

    left = (w - rect_w) // 2
    top = (h - rect_h) // 2
    right = left + rect_w
    bottom = top + rect_h

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # darken outside area lightly
    shade = (0, 0, 0, 90)
    d.rectangle([0, 0, w, top], fill=shade)
    d.rectangle([0, bottom, w, h], fill=shade)
    d.rectangle([0, top, left, bottom], fill=shade)
    d.rectangle([right, top, w, bottom], fill=shade)

    # rectangle border + corner ticks
    border = (0, 191, 255, 255)  # deep sky blue
    d.rectangle([left, top, right, bottom], outline=border, width=max(3, w//300))

    tick = max(16, min(w, h)//25)
    width = max(4, w//250)

    # corners
    # top-left
    d.line([(left, top), (left+tick, top)], fill=border, width=width)
    d.line([(left, top), (left, top+tick)], fill=border, width=width)
    # top-right
    d.line([(right, top), (right-tick, top)], fill=border, width=width)
    d.line([(right, top), (right, top+tick)], fill=border, width=width)
    # bottom-left
    d.line([(left, bottom), (left+tick, bottom)], fill=border, width=width)
    d.line([(left, bottom), (left, bottom-tick)], fill=border, width=width)
    # bottom-right
    d.line([(right, bottom), (right-tick, bottom)], fill=border, width=width)
    d.line([(right, bottom), (right, bottom-tick)], fill=border, width=width)

    # composite and return RGB
    out = Image.alpha_composite(img, overlay).convert("RGB")
    return out


# -----------------------------
# Preprocessing (DL-2 style)
# -----------------------------
def preprocess_fast(pil: Image.Image, target_h=64, target_w=384) -> np.ndarray:
    """
    Mirrors your DL-2 preprocessing pipeline:
      grayscale -> gaussian denoise -> CLAHE -> sharpen
      -> resize by height -> pad/crop width -> normalize to [0,1]
    Returns: float32 array shape (64,384)
    """
    rgb = np.array(pil.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    gray = cv2.filter2D(gray, -1, kernel)

    # resize by height preserving aspect ratio
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Invalid image shape after grayscale conversion.")

    scale = target_h / float(h)
    new_w = int(round(w * scale))
    resized = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_AREA)

    # pad or crop to target_w
    if new_w < target_w:
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        out = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=255)
    else:
        start = (new_w - target_w) // 2
        out = resized[:, start:start + target_w]

    out = out.astype(np.float32) / 255.0
    return out


def preprocess_for_model(pil: Image.Image) -> torch.Tensor:
    """
    Returns torch tensor shape [1,1,64,384]
    """
    arr = preprocess_fast(pil, target_h=64, target_w=384)
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return x


def preprocess_debug_image(pil: Image.Image) -> Image.Image:
    """
    Returns a visible preview of the preprocessed input (scaled back to 0..255).
    """
    arr = preprocess_fast(pil, target_h=64, target_w=384)
    img = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="L")


# -----------------------------
# Output formatting
# -----------------------------
def format_topk(topk, labels):
    """
    topk: list[(idx, prob)]
    labels: list[str]
    """
    meds = []
    for idx, p in topk:
        label = labels[idx] if 0 <= idx < len(labels) else f"class_{idx}"
        meds.append({
            "rank": len(meds) + 1,
            "medication": label,
            "probability_pct": f"{p * 100:.2f}%",
            "probability_float": float(p)
        })
    return pd.DataFrame(meds)
