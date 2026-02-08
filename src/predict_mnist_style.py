
import os, re, math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


from predict_my_images import MODEL_PATH, IMAGES_DIR, predict_one

SAVE_DIR = "artifacts"
os.makedirs(SAVE_DIR, exist_ok=True)

def parse_true_from_name(name: str):
    """
    Dosya adında geçen ilk rakamı 'T' (gerçek etiket) olarak alır.
    Örn: '8.png', '7_kendi.jpg', 'label-4.png' -> 8, 7, 4
    Hiç rakam yoksa None döner.
    """
    m = re.search(r'([0-9])', name)
    return int(m.group(1)) if m else None

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    files = [f for f in os.listdir(IMAGES_DIR)
             if f.lower().endswith((".png",".jpg",".jpeg",".bmp"))]
    files.sort()
    if not files:
        print("my_digits klasöründe görsel yok.")
        return

    n = len(files)
    cols = min(10, n)
    rows = math.ceil(n / cols)

    plt.figure(figsize=(2.6*cols, 2.6*rows))
    for i, fname in enumerate(files, start=1):
        path = os.path.join(IMAGES_DIR, fname)
        with Image.open(path) as pil:
            
            pred, conf, used, _, vis, _ = predict_one(model, pil, save_name=f"style_{fname}.png")

        true_label = parse_true_from_name(fname)
        title = f"T:{true_label if true_label is not None else '-'}  P:{pred}"

        plt.subplot(rows, cols, i)
        plt.imshow((vis * 255).astype("uint8"), cmap="gray")
        plt.title(title, fontsize=10)
        plt.axis("off")

    out = os.path.join(SAVE_DIR, "my_digits_mnist_style.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print("MNIST tarzı görsel kaydedildi:", out)

if __name__ == "__main__":
    main()
