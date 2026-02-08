import os, math
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow as tf

from predict_my_images import MODEL_PATH, IMAGES_DIR, predict_one

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    files = [f for f in os.listdir(IMAGES_DIR)
             if f.lower().endswith((".png",".jpg",".jpeg",".bmp"))]
    if not files:
        print("my_digits klasöründe görsel yok.")
        return

    cols = min(5, len(files))
    rows = 2
    plt.figure(figsize=(3.2*cols, 6))

    for i, fname in enumerate(sorted(files), start=1):
        path = os.path.join(IMAGES_DIR, fname)
        with Image.open(path) as pil:
            pred, conf, used, _, vis, top3 = predict_one(model, pil, save_name=f"vis_{fname}.png")
            title = f"{fname} | P:{pred} ({conf:.2f}) • {used} • top3:{' '.join([f'{i}:{p:.2f}' for i,p in top3])}"

            
            plt.subplot(rows, cols, i)
            plt.imshow(ImageOps.exif_transpose(pil))
            plt.title(title, fontsize=9)
            plt.axis("off")

            
            plt.subplot(rows, cols, cols + i)
            plt.imshow((vis * 255).astype("uint8"), cmap="gray")
            plt.title("işlenmiş (28×28)", fontsize=9)
            plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
