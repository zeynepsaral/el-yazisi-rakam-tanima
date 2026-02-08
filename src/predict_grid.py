import os, math, numpy as np, matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf


from predict_my_images import (
    MODEL_PATH, IMAGES_DIR, preprocess_to_mnist28, predict_one
)

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    files = [f for f in os.listdir(IMAGES_DIR)
             if f.lower().endswith((".png",".jpg",".jpeg",".bmp"))]
    if not files:
        print("my_digits klasöründe görsel yok.")
        return

    imgs, titles = [], []
    for fname in files:
        path = os.path.join(IMAGES_DIR, fname)
        with Image.open(path) as pil:
            
            pred, conf, used = predict_one(model, pil, save_name=f"grid_{fname}.png")

            
            arr = preprocess_to_mnist28(pil)
            if used == "inverted":
                arr = 1.0 - arr   

            imgs.append(arr)
            titles.append(f"{fname}\nP:{pred} ({conf:.2f}) • {used}")

    cols = min(4, len(imgs))
    rows = math.ceil(len(imgs)/cols)
    plt.figure(figsize=(3*cols, 3*rows))
    for i, (arr, title) in enumerate(zip(imgs, titles), start=1):
        plt.subplot(rows, cols, i)
        plt.imshow(arr, cmap="gray")
        plt.title(title, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
