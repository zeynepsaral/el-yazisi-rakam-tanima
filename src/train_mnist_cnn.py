import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

SAVE_DIR = "artifacts"
os.makedirs(SAVE_DIR, exist_ok=True)



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
x_test  = (x_test.astype("float32")  / 255.0)[..., np.newaxis]


val_ratio = 0.1
val_size = int(len(x_train) * val_ratio)
x_val, y_val = x_train[:val_size], y_train[:val_size]
x_train, y_train = x_train[val_size:], y_train[val_size:]

print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")


def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(10, activation="softmax")
    ])
    return model

model = build_model()
model.summary()


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",  
    metrics=["accuracy"]
)


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(SAVE_DIR, "mnist_cnn.keras"),
        monitor="val_accuracy",
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1
    ),
]


history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=12,
    batch_size=128,
    callbacks=callbacks,
    verbose=2
)


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n[TEST] Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")


y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix shape:", cm.shape)


def show_predictions(images, labels, preds, n=8):
    idx = np.random.choice(len(images), size=n, replace=False)
    plt.figure(figsize=(12, 3))
    for i, k in enumerate(idx):
        plt.subplot(1, n, i+1)
        plt.imshow(images[k].squeeze(), cmap="gray")
        title = f"T:{labels[k]} P:{preds[k]}"
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

show_predictions(x_test, y_test, y_pred, n=10)


model.save(os.path.join(SAVE_DIR, "mnist_cnn_final.keras"))
print(f"Model kaydedildi: {os.path.join(SAVE_DIR, 'mnist_cnn_final.keras')}")
