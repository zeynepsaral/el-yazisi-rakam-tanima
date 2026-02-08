import os
import numpy as np
import cv2 as cv
from PIL import Image, ImageOps
import tensorflow as tf

MODEL_PATH = os.path.join("artifacts", "mnist_cnn_final.keras")
IMAGES_DIR = "my_digits"
OUT_DIR = os.path.join("artifacts", "processed")
os.makedirs(OUT_DIR, exist_ok=True)


def _exif_fix(pil_img: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(pil_img)

def _center_of_mass_shift(img28: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(img28)
    if len(xs) == 0:
        return img28
    cy, cx = ys.mean(), xs.mean()
    shiftx = int(round(14 - cx))
    shifty = int(round(14 - cy))
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    return cv.warpAffine(img28, M, (28, 28))

def _largest_component_crop(bw: np.ndarray, pad: int = 2) -> np.ndarray:
   
    cnts, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros((28, 28), dtype=np.uint8)
    c = max(cnts, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(c)
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(bw.shape[1], x + w + pad); y1 = min(bw.shape[0], y + h + pad)
    return bw[y0:y1, x0:x1]


def preprocess_to_mnist28(pil_img: Image.Image) -> np.ndarray:
    """
    Çıktı: (28,28) float32 [0,1], siyah zemin / beyaz rakam
    1) Kırmızı kalem maskeleme (HSV) -> en iyi sonuç
    2) Yetersizse gri + adaptif threshold'a düş
    3) 20x20 ölçekle, 28x28 ortala, kütle merkezine hizala
    """
    pil_img = _exif_fix(pil_img)
    rgb = np.array(pil_img.convert("RGB"))
    hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)

    
    lower1 = np.array([0, 60, 40], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 60, 40], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv.inRange(hsv, lower1, upper1)
    mask2 = cv.inRange(hsv, lower2, upper2)
    red_mask = cv.bitwise_or(mask1, mask2)

    
    use_color = (red_mask > 0).sum() > 100

    if use_color:
        bw = red_mask  # beyaz vuruş = kırmızı pikseller
        
        bw = cv.medianBlur(bw, 3)
        bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    else:
       
        gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
        bg = cv.medianBlur(gray, 31)
        norm = cv.divide(gray, bg, scale=255)
        bw = cv.adaptiveThreshold(
            norm, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 2
        )
        bw = cv.medianBlur(bw, 3)

    
    digit = _largest_component_crop(bw, pad=2)

    
    h, w = digit.shape
    if max(h, w) == 0:
        return np.zeros((28, 28), dtype=np.float32)
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    digit = cv.resize(digit, (new_w, new_h), interpolation=cv.INTER_AREA)

    
    canvas = np.zeros((28, 28), dtype=np.uint8)
    yy = (28 - new_h) // 2
    xx = (28 - new_w) // 2
    canvas[yy:yy + new_h, xx:xx + new_w] = digit

    
    canvas = _center_of_mass_shift(canvas)
    canvas = cv.dilate(canvas, np.ones((3,3), np.uint8), iterations=1)

    return (canvas / 255.0).astype("float32")

def _topk(probs, k=3):
    idx = np.argsort(probs)[::-1][:k]
    return [(int(i), float(probs[i])) for i in idx]

def predict_one(model, pil: Image.Image, save_name: str):
    """Normal ve ters versiyonları dener; en güvenileni seçer. Top-3 de döndürür."""
    arr = preprocess_to_mnist28(pil)           # (28,28) [0,1]
    x1 = arr[np.newaxis, ..., np.newaxis]
    x2 = (1.0 - arr)[np.newaxis, ..., np.newaxis]

    p1 = model.predict(x1, verbose=0)[0]
    p2 = model.predict(x2, verbose=0)[0]

    if p1.max() >= p2.max():
        pred, conf, used, vis, probs = int(np.argmax(p1)), float(p1.max()), "normal", arr, p1
    else:
        pred, conf, used, vis, probs = int(np.argmax(p2)), float(p2.max()), "inverted", 1.0 - arr, p2

    out_path = os.path.join(OUT_DIR, save_name)
    Image.fromarray((vis * 255).astype("uint8")).save(out_path)

    top3 = _topk(probs, k=3)
    return pred, conf, used, out_path, vis, top3

def main():
    print(f"Model yükleniyor: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    files = [f for f in os.listdir(IMAGES_DIR)
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    if not files:
        print("my_digits klasöründe görsel yok.")
        return

    for fname in sorted(files):
        path = os.path.join(IMAGES_DIR, fname)
        with Image.open(path) as pil:
            pred, conf, used, outp, _, top3 = predict_one(model, pil, save_name=f"proc_{fname}.png")
        top3_str = ", ".join([f"{i}:{p:.2f}" for i, p in top3])
        print(f"{fname:12s} -> Tahmin: {pred} | Güven: {conf:.2f} | versiyon: {used} | top3: [{top3_str}] | çıktı: {outp}")

if __name__ == "__main__":
    main()
