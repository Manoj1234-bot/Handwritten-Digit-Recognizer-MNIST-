"""
Day 11 - Handwritten Digit Recognizer (MNIST)
Core Skills: Neural networks (CNN), TensorFlow/Keras
Author: Manoj S

Trains a Convolutional Neural Network (CNN) on the MNIST dataset
(70,000 images of handwritten digits 0-9) and saves the trained model
so it can be used later by the Flask web app (app.py) for live predictions.

NOTE: Running this script downloads the MNIST dataset (~11 MB) from
Google's servers the first time, so you need an internet connection.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def load_and_prepare_data():
    """Load MNIST and normalize pixel values to the 0-1 range."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values (0-255 -> 0-1) - helps the network train faster
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add a channel dimension: (28, 28) -> (28, 28, 1) since CNNs expect
    # a channel axis (1 for grayscale, 3 for RGB)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)


def build_model():
    """
    Build a simple but effective CNN:
    Conv -> Pool -> Conv -> Pool -> Flatten -> Dense -> Output(10 classes)
    """
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(64, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),

        layers.Flatten(),
        layers.Dropout(0.5),  # helps prevent overfitting
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),  # 10 digits: 0-9
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_training_history(history):
    """Save a plot of training/validation accuracy and loss over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    print("Saved training plot to training_history.png")


def main():
    print("=" * 55)
    print("     🔢  MNIST DIGIT RECOGNIZER - TRAINING")
    print("=" * 55)

    print("\nLoading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_prepare_data()
    print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")

    model = build_model()
    model.summary()

    print("\nTraining model (5 epochs)...")
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1,
    )

    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    plot_training_history(history)

    model.save("digit_model.keras")
    print("\nModel saved as digit_model.keras")
    print("You can now run app.py to use the model in the web app.")


if __name__ == "__main__":
    main()