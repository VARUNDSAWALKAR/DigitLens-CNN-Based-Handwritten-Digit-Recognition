"""DigitLens: MNIST CNN classification + Streamlit application (UNLOX minor project).

PART A: CNN training, validation, testing, plots, and correct/incorrect predictions.
PART B: Streamlit interactive handwritten-digit recognition application.

INSTALL (Python 3.10+ recommended):
    python -m pip install torch torchvision streamlit numpy pillow matplotlib reportlab

TRAIN (downloads MNIST on first run; Internet is required only for first download):
    python Varun_CNN_Streamlit_Project.py --train --epochs 3

RUN WEB APP:
    python -m streamlit run Varun_CNN_Streamlit_Project.py

OPTIONAL SELF-TEST (no MNIST download needed):
    python Varun_CNN_Streamlit_Project.py --smoke-test

Generated files stay in digitlens_runtime/ and are not part of the two-file submission.
"""

# ============================ IMPORTS / SETTINGS ==============================
import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = PROJECT_ROOT / "digitlens_runtime"
DATA_DIR = RUNTIME_DIR / "data"
MODEL_PATH = RUNTIME_DIR / "mnist_cnn.pt"
METRICS_PATH = RUNTIME_DIR / "metrics.json"
CLASS_NAMES = [str(i) for i in range(10)]
MEAN = 0.1307
STD = 0.3081
SEED = 42


def set_seed(seed=SEED):
    """Make the train/validation split and weight initialization reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 2))


# ============================== PART A: CNN ===================================
class DigitCNN(nn.Module):
    """A small convolutional network designed for 28 x 28 grayscale digits."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),       # [B, 16, 14, 14]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),       # [B, 32, 7, 7]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(64, 10),                 # logits for digits 0 through 9
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def model_summary():
    """Return architecture and real parameter count (computed, not estimated)."""
    network = DigitCNN()
    n_total = sum(p.numel() for p in network.parameters())
    n_trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
    return f"{network}\n\nTotal parameters: {n_total:,}\nTrainable parameters: {n_trainable:,}\nInput: [batch, 1, 28, 28]\nOutput: [batch, 10] logits"


def mnist_datasets(download=True):
    """Load the allowed MNIST dataset; fit preprocessing on its grayscale pixels."""
    from torchvision import datasets, transforms

    # Google's CVDF mirror hosts the standard MNIST archives and is often more
    # accessible than the two legacy torchvision mirror URLs.
    google_mirror = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    if google_mirror not in datasets.MNIST.mirrors:
        datasets.MNIST.mirrors = [google_mirror] + list(datasets.MNIST.mirrors)

    preprocessor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    try:
        train = datasets.MNIST(root=str(DATA_DIR), train=True,
                               download=download, transform=preprocessor)
        test = datasets.MNIST(root=str(DATA_DIR), train=False,
                              download=download, transform=preprocessor)
    except Exception as exc:
        raise RuntimeError(
            "MNIST could not be loaded. Your first run needs an internet connection "
            "to download the 60,000 training and 10,000 test images. "
            "Try again on a network that permits torchvision's MNIST mirrors. "
            f"Original error: {exc}"
        ) from exc
    return train, test


def one_pass(model, loader, device, loss_fn, optimizer=None, collect=False):
    """One epoch: back-propagate if optimizer is given; otherwise evaluate."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    true_labels, predicted_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((predictions == labels).sum().item())
        count += batch_size
        if collect:
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())
    return {
        "loss": total_loss / count,
        "accuracy": correct / count,
        "labels": true_labels,
        "predictions": predicted_labels,
    }


def _plot_results(history, test_data, actual, predicted):
    """Save genuine training curves, MNIST samples, and test predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in [
        (axes[0], "accuracy", "Classification accuracy"),
        (axes[1], "loss", "Cross-entropy loss"),
    ]:
        ax.plot(history["epoch"], history[f"train_{metric}"],
                marker="o", label="Train")
        ax.plot(history["epoch"], history[f"val_{metric}"],
                marker="s", label="Validation")
        ax.set(xlabel="Epoch", title=title)
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Accuracy (fraction)")
    axes[1].set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(RUNTIME_DIR / "training_curves.png", dpi=170)
    plt.close(fig)

    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (np.asarray(actual), np.asarray(predicted)), 1)
    fig, ax = plt.subplots(figsize=(7.3, 6.3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(10), yticks=range(10),
           xlabel="Predicted digit", ylabel="Actual digit", title="MNIST test confusion matrix")
    for row in range(10):
        for col in range(10):
            val = cm[row, col]
            ax.text(col, row, str(val), ha="center", va="center", fontsize=7,
                    color="white" if val > cm.max() / 2 else "#1a2340")
    fig.colorbar(im, ax=ax, fraction=0.047, pad=0.03)
    fig.tight_layout()
    fig.savefig(RUNTIME_DIR / "confusion_matrix.png", dpi=170)
    plt.close(fig)

    correct_ids = [i for i, (a, p) in enumerate(zip(actual, predicted)) if a == p][:8]
    wrong_ids = [i for i, (a, p) in enumerate(zip(actual, predicted)) if a != p][:8]
    chosen = correct_ids + wrong_ids
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for axis, index in zip(axes.flat, chosen):
        tensor, _ = test_data[index]
        raw = (tensor[0].numpy() * STD + MEAN).clip(0, 1)
        axis.imshow(raw, cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"True {actual[index]} | Pred {predicted[index]}",
                       color="#16865a" if actual[index] == predicted[index] else "#c84c4c",
                       fontsize=10)
        axis.axis("off")
    for axis in list(axes.flat)[len(chosen):]:
        axis.axis("off")
    fig.suptitle("Correct predictions (top) and mistakes (bottom)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(RUNTIME_DIR / "sample_predictions.png", dpi=160)
    plt.close(fig)

    # Use actual held-out MNIST images, not drawings or invented examples.
    fig, axes = plt.subplots(2, 5, figsize=(9, 4))
    for digit in range(10):
        index = next(i for i, target in enumerate(actual) if target == digit)
        tensor, _ = test_data[index]
        axes.flat[digit].imshow((tensor[0].numpy() * STD + MEAN).clip(0, 1), cmap="gray")
        axes.flat[digit].set_title(f"Digit {digit}")
        axes.flat[digit].axis("off")
    fig.suptitle("Actual MNIST test set examples")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(RUNTIME_DIR / "mnist_samples.png", dpi=160)
    plt.close(fig)
    return cm


def train_project(epochs=3, batch_size=128, callback=None):
    """Train CNN from scratch; keep test set separate from model selection."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    set_seed()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    train_full, test_data = mnist_datasets(download=True)
    print(f"Dataset: MNIST | train={len(train_full):,} | test={len(test_data):,} "
          f"| image shape={tuple(train_full[0][0].shape)} | classes={CLASS_NAMES}", flush=True)
    n_val = 6000
    training, validation = random_split(
        train_full, [len(train_full) - n_val, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = {"epoch": [], "train_accuracy": [], "val_accuracy": [],
               "train_loss": [], "val_loss": []}
    best_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        train_metrics = one_pass(model, train_loader, device, criterion, optimizer=optimizer)
        val_metrics = one_pass(model, val_loader, device, criterion)
        history["epoch"].append(epoch)
        for key in ("accuracy", "loss"):
            history[f"train_{key}"].append(train_metrics[key])
            history[f"val_{key}"].append(val_metrics[key])
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_weights = {key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()}
        message = (f"Epoch {epoch}/{epochs} | train acc {train_metrics['accuracy']:.2%} "
                   f"| val acc {val_metrics['accuracy']:.2%} "
                   f"| train loss {train_metrics['loss']:.4f} "
                   f"| val loss {val_metrics['loss']:.4f}")
        print(message, flush=True)
        if callback is not None:
            callback(epoch, epochs, message)

    model.load_state_dict(best_weights)
    model.to(device)
    test_metrics = one_pass(model, test_loader, device, criterion, collect=True)
    confusion = _plot_results(history, test_data,
                              test_metrics["labels"], test_metrics["predictions"])
    checkpoint = {
        "model_state_dict": best_weights,
        "architecture": "DigitCNN-v1",
        "classes": CLASS_NAMES,
        "mean": MEAN,
        "std": STD,
    }
    torch.save(checkpoint, MODEL_PATH)
    metadata = {
        "project": "DigitLens - CNN Image Classification and Streamlit App",
        "dataset": "MNIST",
        "class_names": CLASS_NAMES,
        "input_shape": [1, 28, 28],
        "samples": {"training": 54000, "validation": 6000, "testing": 10000},
        "seed": SEED,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "Adam", "learning_rate": 0.001,
        "loss_function": "CrossEntropyLoss",
        "architecture": "DigitCNN-v1",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "test_correct": int(round(test_metrics["accuracy"] * len(test_data))),
        "test_total": len(test_data),
        "history": history,
        "per_class_accuracy": {
            str(i): (int(confusion[i, i]) / int(confusion[i].sum()))
            for i in range(10)
        },
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    METRICS_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (RUNTIME_DIR / "model_summary.txt").write_text(model_summary(), encoding="utf-8")
    print(f"\nTest accuracy: {test_metrics['accuracy']:.2%} | Test loss: {test_metrics['loss']:.4f}")
    print(f"Model: {MODEL_PATH}\nCharts and metrics: {RUNTIME_DIR}")
    return metadata


def load_model():
    """Safe state-dict loading; never unpickle an arbitrary user-supplied model."""
    if not MODEL_PATH.exists():
        return None
    payload = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if payload.get("architecture") != "DigitCNN-v1":
        raise RuntimeError("Model architecture mismatch; please retrain.")
    model = DigitCNN()
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def read_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def preprocess_uploaded_image(image):
    """Convert photos / drawings into MNIST-like centered 28x28 white-on-black digits.

    Blank images are rejected instead of producing a meaningless prediction.
    """
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba).convert("RGB")
    image = image.convert("L")
    if max(image.size) > 1200:
        image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    gray = np.asarray(image, dtype=np.uint8)
    if gray.size == 0:
        raise ValueError("The uploaded image is empty.")
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    if np.median(border) > 127:
        gray = 255 - gray  # accept black-on-white photos and white-on-black MNIST
    # Filter out weak background artifacts and detect the main foreground bounds.
    threshold = max(40, int(gray.max() * 0.18))
    foreground = gray > threshold
    if int(foreground.sum()) < 8 or int(gray.max()) < 45:
        raise ValueError("No clear digit detected. Upload an image with one visible digit.")
    yy, xx = np.where(foreground)
    cropped = gray[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    height, width = cropped.shape
    scale = 20.0 / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = Image.fromarray(cropped).resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    canvas.paste(resized, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))
    centered = np.asarray(canvas, dtype=np.uint8)
    # Shift the image toward the center of mass, as in common MNIST processing.
    mass = centered.astype(np.float64)
    total = float(mass.sum())
    if total <= 0:
        raise ValueError("Image could not be normalized to a handwritten digit.")
    y_coord, x_coord = np.indices(mass.shape)
    dx = int(round(13.5 - float((x_coord * mass).sum()) / total))
    dy = int(round(13.5 - float((y_coord * mass).sum()) / total))
    result = Image.new("L", (28, 28), 0)
    result.paste(canvas, (dx, dy))
    return np.asarray(result, dtype=np.uint8)


def predict_digit(model, pixels):
    if pixels.shape != (28, 28):
        raise ValueError("Prediction requires a 28 x 28 image.")
    tensor = torch.from_numpy(pixels.copy()).float()[None, None] / 255.0
    tensor = (tensor - MEAN) / STD
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return int(np.argmax(probabilities)), probabilities


def smoke_test():
    """Architecture, gradient, shape, and upload-transform checks without a dataset."""
    set_seed()
    model = DigitCNN()
    x = torch.rand((4, 1, 28, 28))
    y = torch.tensor([0, 3, 6, 9], dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logits = model(x)
    assert logits.shape == (4, 10), f"Unexpected model output: {logits.shape}"
    loss = nn.CrossEntropyLoss()(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert float(loss.detach().item()) > 0
    # A white-background black glyph tests polarity conversion and centering.
    canvas = Image.new("L", (80, 80), 255)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 56)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 6), "5", font=font, fill=0)
    processed = preprocess_uploaded_image(canvas)
    assert processed.shape == (28, 28) and processed.max() > 0
    digit, scores = predict_digit(model.eval(), processed)
    assert 0 <= digit <= 9 and abs(float(scores.sum()) - 1.0) < 1e-5
    try:
        preprocess_uploaded_image(Image.new("L", (50, 50), 255))
    except ValueError:
        pass
    else:
        raise AssertionError("Blank images must be rejected.")
    print("PASS: CNN forward pass [4,1,28,28] -> [4,10]")
    print("PASS: loss computation, back-propagation, optimizer step")
    print("PASS: upload preprocessing and normalized prediction")
    print("PASS: blank-image handling")
    print("NOTE: this is a synthetic smoke test, not an MNIST accuracy evaluation.")


# ========================= PART B: STREAMLIT APP ===============================
def streamlit_app():
    import streamlit as st
    import matplotlib.pyplot as plt

    st.set_page_config(page_title="DigitLens | Handwriting AI", page_icon="🔢",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
      .block-container{padding-top:2rem;max-width:1200px;}
      h1,h2,h3{letter-spacing:-.02em;}
      div[data-testid="stMetric"]{border:1px solid #e3e9f4;padding:16px;
                                    border-radius:14px;}
    </style>""", unsafe_allow_html=True)
    st.caption("UNLOX  /  AIML MINOR PROJECT  /  MNIST CNN")
    st.title("🔢 DigitLens: Handwriting AI")
    st.write("Upload a handwritten digit or explore the MNIST test set. "
             "A convolutional neural network classifies digits **0–9** and displays confidence estimates.")
    with st.sidebar:
        st.header("Project overview")
        st.write("**Dataset:** MNIST · 28 × 28 grayscale")
        st.write("**Classes:** 0–9 · **Network:** 2 Conv blocks + dense layers")
        st.write("**Framework:** PyTorch · **UI:** Streamlit")
        st.divider()
        st.caption("To train from the terminal:")
        st.code("python Varun_CNN_Streamlit_Project.py --train --epochs 3", language="bash")
        st.caption("A first download requires internet. Model artifacts are generated locally.")

    tab_predict, tab_metrics, tab_train = st.tabs(
        ["🎯 Predict digit", "📈 Model results", "⚙️ Train & setup"])
    model = None
    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Saved model could not be loaded: {exc}")

    with tab_predict:
        if model is None:
            st.info("No trained model found yet. Open **Train & setup** to train the CNN "
                    "(or run the training command shown in the sidebar).")
        else:
            input_source = st.radio("Choose image source",
                                    ("Upload your image", "Try an MNIST test image"),
                                    horizontal=True)
            chosen_image = None
            true_label = None
            if input_source == "Upload your image":
                uploaded = st.file_uploader(
                    "Upload one handwritten digit (PNG, JPG or JPEG)",
                    type=["png", "jpg", "jpeg"], help="One digit, high contrast, simple background.")
                if uploaded is not None:
                    if uploaded.size > 8 * 1024 * 1024:
                        st.error("The image is too large. Use a file under 8 MB.")
                    else:
                        try:
                            chosen_image = Image.open(uploaded)
                            chosen_image.load()
                        except Exception:
                            st.error("Cannot read this image. Upload a valid PNG or JPEG.")
            else:
                try:
                    _, testing = mnist_datasets(download=False)
                    index = st.slider("Test sample index", 0, len(testing) - 1, 23)
                    image_tensor, true_label = testing[index]
                    img_array = ((image_tensor[0].numpy() * STD + MEAN) * 255).clip(0, 255)
                    chosen_image = Image.fromarray(img_array.astype(np.uint8), mode="L")
                except Exception as exc:
                    st.warning(f"MNIST test set unavailable locally: {exc}")
            if chosen_image is not None:
                try:
                    normalized = preprocess_uploaded_image(chosen_image)
                    predicted, probabilities = predict_digit(model, normalized)
                    left, middle, right = st.columns([1.2, 1.2, 1.6])
                    with left:
                        st.markdown("**Original image**")
                        st.image(chosen_image, width=235)
                    with middle:
                        st.markdown("**CNN input · 28 × 28**")
                        st.image(Image.fromarray(normalized).resize((224, 224),
                                 Image.Resampling.NEAREST), width=224)
                    with right:
                        st.metric("Predicted digit", str(predicted))
                        st.metric("Model confidence", f"{float(probabilities[predicted]):.1%}")
                        if true_label is not None:
                            st.write(f"**MNIST actual label:** {true_label}")
                            if predicted == true_label:
                                st.success("Prediction matches the test label.")
                            else:
                                st.warning("Prediction differs from the test label.")
                        if float(probabilities[predicted]) < .70:
                            st.warning("Low confidence. Try a clearer, centered image.")
                    fig, ax = plt.subplots(figsize=(9, 2.8))
                    colors = ["#2563eb" if i == predicted else "#9facc6" for i in range(10)]
                    ax.bar(range(10), probabilities * 100, color=colors)
                    ax.set(xticks=range(10), xlabel="Digit", ylabel="Model score (%)",
                           ylim=(0, 105), title="Class confidence scores")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption("Confidence is the model's softmax score, not a guarantee of correctness. "
                               "Photos and unusual handwriting can differ from MNIST training data.")
                except ValueError as exc:
                    st.error(str(exc))

    with tab_metrics:
        metrics = read_metrics()
        if metrics is None:
            st.info("Evaluation charts will be available after CNN training finishes.")
        else:
            a, b, c = st.columns(3)
            a.metric("Held-out test accuracy", f"{metrics['test_accuracy']:.2%}")
            b.metric("Test cross-entropy", f"{metrics['test_loss']:.4f}")
            c.metric("Test images evaluated", f"{metrics['test_total']:,}")
            st.caption("Training / validation split: 54,000 / 6,000. Separate test split: 10,000.")
            for label, name in [
                ("Learning curves", "training_curves.png"),
                ("Example images", "mnist_samples.png"),
                ("Correct and incorrect predictions", "sample_predictions.png"),
                ("Test-set confusion matrix", "confusion_matrix.png"),
            ]:
                picture = RUNTIME_DIR / name
                if picture.exists():
                    with st.expander(label, expanded=(name == "training_curves.png")):
                        st.image(str(picture), use_container_width=True)
            st.markdown("**Per-digit accuracy on held-out test images**")
            st.json(metrics["per_class_accuracy"], expanded=False)
            st.caption("All displayed results are loaded from the latest local training run.")

    with tab_train:
        st.subheader("Train your CNN")
        st.write("This step downloads **MNIST** once, trains the CNN, "
                 "evaluates it on held-out data, and saves an actual model and charts.")
        st.code("python -m pip install torch torchvision streamlit numpy pillow matplotlib reportlab\n"
                "python Varun_CNN_Streamlit_Project.py --train --epochs 3\n"
                "python -m streamlit run Varun_CNN_Streamlit_Project.py", language="bash")
        st.write("**Neural network summary**")
        st.code(model_summary(), language="text")
        epoch_count = st.number_input("Training epochs", min_value=1,
                                      max_value=15, value=3, step=1)
        if st.button("🚀 Start training on MNIST", type="primary"):
            bar = st.progress(0, text="Preparing data")
            lines = st.empty()

            def update_progress(current, total, message):
                bar.progress(current / total, text=f"Epoch {current} of {total}")
                lines.info(message)

            try:
                with st.spinner("Training and testing CNN. First run may download the dataset..."):
                    outcome = train_project(epochs=int(epoch_count), callback=update_progress)
                st.success(f"Training complete. Test accuracy: {outcome['test_accuracy']:.2%}. "
                           "Go to Model results or Predict digit; refresh if needed.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.info("Check your internet connection for the first MNIST download, "
                        "package versions and disk space, then retry.")
        st.divider()
        st.markdown("**Input validation built into the app**")
        st.write("Unreadable uploads, blank images, large files, missing model files, "
                 "and unavailable MNIST data produce helpful feedback instead of a crash.")



# ==================== PDF REPORT (OPTIONAL EXPORT) ============================
def export_pdf_report(student_name="Varun D. Sawalkar", batch="[Enter batch]",
                      app_screenshot=None):
    """Build the required 6-page report from ACTUAL metrics/plots when available.

    Optional dependency: python -m pip install reportlab
    For the app screenshot, save a real browser capture as:
        digitlens_runtime/app_screenshot.png
    Then run: python Varun_CNN_Streamlit_Project.py --export-report
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.utils import ImageReader
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                        TableStyle, Image as PDFImage, PageBreak,
                                        KeepTogether, Preformatted)
    except ImportError as exc:
        raise RuntimeError("Install reportlab first: python -m pip install reportlab") from exc
    from xml.sax.saxutils import escape

    pdf_path = PROJECT_ROOT / "Varun_CNN_Streamlit_Project_Report.pdf"
    metrics = read_metrics()
    dark = colors.HexColor("#13263E")
    blue = colors.HexColor("#1E63DC")
    soft = colors.HexColor("#EDF4FE")
    teal = colors.HexColor("#0F9BA8")
    muted = colors.HexColor("#59718C")
    line = colors.HexColor("#D9E4F3")
    W, H = A4
    width = W - 92
    styles = {
        "title": ParagraphStyle("dl_title", fontName="Helvetica-Bold", fontSize=31,
                                leading=37, textColor=dark, spaceAfter=14),
        "heading": ParagraphStyle("dl_heading", fontName="Helvetica-Bold", fontSize=20,
                                  leading=25, textColor=dark, spaceAfter=13),
        "sub": ParagraphStyle("dl_sub", fontName="Helvetica-Bold", fontSize=11,
                              leading=15, textColor=blue, spaceBefore=8, spaceAfter=6),
        "body": ParagraphStyle("dl_body", fontName="Helvetica", fontSize=9.4,
                               leading=14.6, textColor=dark, spaceAfter=7),
        "small": ParagraphStyle("dl_small", fontName="Helvetica", fontSize=8,
                                leading=12, textColor=muted, spaceAfter=4),
        "card": ParagraphStyle("dl_card", fontName="Helvetica-Bold", fontSize=10.2,
                               leading=15.3, textColor=dark, alignment=TA_CENTER),
        "tiny": ParagraphStyle("dl_tiny", fontName="Helvetica", fontSize=7.3,
                               leading=10, textColor=muted),
    }
    def para(value, style="body"):
        return Paragraph(value, styles[style])

    def simple_table(rows, widths=None, header=True, padd=8):
        table = Table(rows, colWidths=widths, hAlign="LEFT")
        cmds = [
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), padd),
            ("RIGHTPADDING", (0, 0), (-1, -1), padd),
            ("TOPPADDING", (0, 0), (-1, -1), padd),
            ("BOTTOMPADDING", (0, 0), (-1, -1), padd),
            ("LINEBELOW", (0, 0), (-1, -2), .5, line),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ]
        if header:
            cmds += [("BACKGROUND", (0, 0), (-1, 0), soft)]
        table.setStyle(TableStyle(cmds))
        return table

    def image_or_placeholder(filename, w, h, title, message):
        path = RUNTIME_DIR / filename
        if path.exists():
            from PIL import Image as PILImage
            with PILImage.open(path) as original:
                iw, ih = original.size
            scale = min(w / iw, h / ih)
            img = PDFImage(str(path), width=iw * scale, height=ih * scale)
            img.hAlign = "CENTER"
            return img
        data = [
            [para(f"<b>{escape(title)}</b>", "body")],
            [para(escape(message), "small")],
        ]
        block = Table(data, colWidths=[w], rowHeights=[h * .50, h * .50])
        block.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), soft),
            ("BOX", (0, 0), (-1, -1), .8, line),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return block

    def frame(canvas, document):
        page = document.page
        canvas.saveState()
        canvas.setFillColor(dark)
        canvas.rect(0, H - 12, W, 12, fill=1, stroke=0)
        canvas.setFillColor(blue)
        canvas.roundRect(46, H - 42, 9, 9, 3, fill=1, stroke=0)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(dark)
        canvas.drawString(62, H - 40, "DIGITLENS  /  AIML MINOR PROJECT")
        canvas.setStrokeColor(line)
        canvas.line(46, 42, W - 46, 42)
        canvas.setFillColor(muted)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(46, 28, "CNN IMAGE CLASSIFICATION + STREAMLIT")
        canvas.drawRightString(W - 46, 28, f"PAGE {page} / 6")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=46, rightMargin=46, topMargin=68, bottomMargin=58,
        title="DigitLens | CNN Image Classification and Streamlit Application",
        author=student_name, subject="UNLOX AIML Minor Project Report",
    )
    story = []
    # PAGE 1 - COVER
    story += [Spacer(1, 40), para("AI / COMPUTER VISION / INTERACTIVE WEB", "sub"),
              para("DigitLens", "title"),
              para("CNN-Based Handwritten Digit Recognition<br/>and Streamlit Application", "heading"),
              Spacer(1, 21)]
    box_rows = [
        [para("DATASET", "small"), para("MODEL", "small"), para("DELIVERABLES", "small")],
        [para("<b>MNIST: digits 0-9</b>", "card"),
         para("<b>PyTorch CNN</b>", "card"),
         para("<b>1 Python + 1 PDF</b>", "card")],
    ]
    box = Table(box_rows, colWidths=[width / 3] * 3, rowHeights=[30, 52])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), soft),
        ("GRID", (0, 0), (-1, -1), .5, line),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story += [box, Spacer(1, 34),
              para("<b>Student:</b> " + escape(student_name)),
              para("<b>Batch:</b> " + escape(batch)),
              para("<b>Date:</b> " + datetime.now().strftime("%d %B %Y")),
              para("<b>App topic:</b> Interactive handwritten-digit image classifier"),
              Spacer(1, 28),
              para("Project overview", "sub"),
              para("A small convolutional neural network learns spatial patterns in "
                   "28 x 28 grayscale images, predicts one of ten digit classes, and "
                   "powers a Streamlit dashboard for user-uploaded images. The application "
                   "also includes training and evaluation views."),
              Spacer(1, 18)]
    if metrics:
        story.append(para(f"<b>Execution status:</b> Full MNIST run recorded. "
                          f"Measured test accuracy: {metrics['test_accuracy']:.2%}.", "small"))
    else:
        story.append(para("<b>Execution status:</b> Code passed synthetic offline checks. "
                          "MNIST training, results, and live app screenshot must be "
                          "captured on a machine with dataset access before final submission.", "small"))
    story.append(PageBreak())

    # PAGE 2 - REQUIREMENTS + DATASET + PREPROCESSING
    story += [para("01  Project objective and dataset", "heading"),
              para("<b>Part A:</b> Classify digit images using a trained CNN, quantify "
                   "generalization on held-out images, and explain prediction successes and failures."),
              para("<b>Part B:</b> Provide a usable Streamlit interface with clear instructions, "
                   "an image upload or sample selector, prediction processing, visible outputs, "
                   "and error handling."),
              para("Dataset specification", "sub")]
    story.append(simple_table([
        [para("Property", "body"), para("Value", "body")],
        [para("Source / classes", "small"), para("MNIST, handwritten numerals 0 to 9", "small")],
        [para("Image dimensions", "small"), para("28 x 28 pixels; one grayscale channel", "small")],
        [para("Original split", "small"), para("60,000 training images; 10,000 held-out test images", "small")],
        [para("Project train / validation", "small"), para("54,000 train; 6,000 validation (seed = 42)", "small")],
    ], widths=[170, width - 170]))
    story += [Spacer(1, 12), para("Preprocessing workflow", "sub")]
    steps = ["Image -> grayscale", "Center / crop digit", "Resize to 28 x 28", "Normalize pixels", "CNN prediction"]
    flow_cells = [para(f"<b>{escape(text)}</b>", "tiny") for text in steps]
    flow = Table([flow_cells], colWidths=[width / len(flow_cells)] * len(flow_cells), rowHeights=[54])
    flow.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), .6, line),
        ("INNERGRID", (0, 0), (-1, -1), .6, line),
        ("BACKGROUND", (0, 0), (-1, -1), soft),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [flow, Spacer(1, 15),
              para("MNIST image examples (digits 0-9)", "sub"),
              image_or_placeholder(
                  "mnist_samples.png", width, 170, "Image evidence is pending",
                  "Run the MNIST training command to generate genuine digit examples. "
                  "Then use --export-report again."),
              Spacer(1, 10),
              para("The original MNIST pixels are converted to tensors in [0, 1] and "
                   "normalized with mean 0.1307 and standard deviation 0.3081. "
                   "User uploads additionally handle background polarity, centering, "
                   "and blank-image rejection.", "small"),
              PageBreak()]

    # PAGE 3 - MODEL AND TRAINING METHOD
    story += [para("02  Model architecture and training", "heading"),
              para("CNN design", "sub")]
    layers = [
        ("Input", "Grayscale image [1, 28, 28]"),
        ("Convolution block 1", "Conv2d(1, 16, kernel=3, padding=1) + ReLU + MaxPool2d(2)"),
        ("Convolution block 2", "Conv2d(16, 32, kernel=3, padding=1) + ReLU + MaxPool2d(2)"),
        ("Classifier", "Flatten 1,568 units -> Dense 64 -> ReLU -> Dropout(0.25)"),
        ("Output", "Dense 10 logits, predicted class = argmax(logits)"),
    ]
    story.append(simple_table(
        [[para("Stage", "body"), para("Implementation", "body")]] +
        [[para(escape(a), "small"), para(escape(b), "small")] for a, b in layers],
        widths=[145, width - 145], padd=8))
    story += [Spacer(1, 10),
              para(f"<b>Trainable parameters:</b> {sum(p.numel() for p in DigitCNN().parameters()):,} "
                   "(computed from the defined PyTorch network).", "body"),
              para("Training protocol", "sub")]
    config = [
        ("Loss", "Categorical cross-entropy over raw class logits"),
        ("Optimizer", "Adam, learning rate 0.001"),
        ("Validation", "Fixed 54k/6k split; keep checkpoint with lowest validation loss"),
        ("Test", "Evaluate once after model selection on separate 10k examples"),
        ("Reproducibility", "Seed = 42; batch size = 128; default = 3 epochs"),
    ]
    story.append(simple_table(
        [[para("Setting", "body"), para("Choice", "body")]] +
        [[para(escape(a), "small"), para(escape(b), "small")] for a, b in config],
        widths=[125, width - 125], padd=6))
    story += [Spacer(1, 12), para("How the CNN learns", "sub"),
              para("Convolution filters detect local edges and strokes; pooling reduces "
                   "spatial dimensions; the dense layers combine these features into ten "
                   "digit scores. Backpropagation updates filter weights to minimize loss."),
              para("For the required model-summary screenshot: open the Streamlit "
                   "'Train &amp; setup' tab, where the full architecture is displayed.", "small"),
              PageBreak()]

    # PAGE 4 - RESULTS: NO FAKE ACCURACY
    story += [para("03  CNN evaluation and findings", "heading")]
    if metrics:
        result_values = [
            ("TEST ACCURACY", f"{metrics['test_accuracy']:.2%}"),
            ("TEST LOSS", f"{metrics['test_loss']:.4f}"),
            ("CORRECT / 10,000", f"{metrics['test_correct']:,}"),
        ]
    else:
        result_values = [
            ("TEST ACCURACY", "Not yet measured"),
            ("TEST LOSS", "Not yet measured"),
            ("CORRECT / 10,000", "Not yet measured"),
        ]
    metrics_table = Table([
        [para(escape(x[0]), "tiny") for x in result_values],
        [para(f"<b>{escape(x[1])}</b>", "card") for x in result_values],
    ], colWidths=[width / 3] * 3, rowHeights=[33, 38])
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), soft),
        ("GRID", (0, 0), (-1, -1), .6, line),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [metrics_table, Spacer(1, 9),
              para("Training and validation curves", "sub"),
              image_or_placeholder(
                  "training_curves.png", width, 205, "Training / validation evidence",
                  "Generated after real training: epoch accuracy and cross-entropy loss "
                  "for training and validation splits."),
              Spacer(1, 8),
              para("Test errors and correct predictions", "sub")]
    pair = Table([[
        image_or_placeholder("confusion_matrix.png", 242, 215,
                             "Confusion matrix pending",
                             "10 x 10 matrix created from 10,000 test predictions."),
        image_or_placeholder("sample_predictions.png", 242, 215,
                             "Predictions pending",
                             "8 correct and 8 incorrect genuine MNIST predictions."),
    ]], colWidths=[width / 2, width / 2])
    pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 2),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    story += [pair, Spacer(1, 10)]
    if metrics:
        best = max(metrics["per_class_accuracy"].items(), key=lambda item: item[1])
        worst = min(metrics["per_class_accuracy"].items(), key=lambda item: item[1])
        story.append(para("<b>Observation:</b> Digit " + escape(best[0]) +
                          " is best recognized (" + f"{best[1]:.2%}" +
                          "); digit " + escape(worst[0]) + " is hardest (" +
                          f"{worst[1]:.2%}" + ") on this run's test images.", "small"))
    else:
        story.append(para("<b>Observation pending:</b> Report actual test accuracy and "
                          "which digit pairs the confusion matrix shows are most often "
                          "mixed up. Do not replace this with an estimated score.", "small"))
    story.append(PageBreak())

    # PAGE 5 - STREAMLIT
    story += [para("04  Streamlit application", "heading"),
              para("DigitLens is a single-page, three-tab web dashboard. Users can "
                   "upload a photograph or test a real MNIST example, compare the "
                   "original image with its 28 x 28 input, and inspect ten class scores."),
              para("Live application screenshot", "sub")]
    screenshot_path = Path(app_screenshot) if app_screenshot else RUNTIME_DIR / "app_screenshot.png"
    if screenshot_path.is_file():
        with Image.open(screenshot_path) as source:
            img_w, img_h = source.size
        sf = min(width / img_w, 310 / img_h)
        proof = PDFImage(str(screenshot_path), width=img_w * sf, height=img_h * sf)
        proof.hAlign = "CENTER"
        story.append(proof)
    else:
        story.append(image_or_placeholder("app_screenshot.png", width, 285,
                "Actual Streamlit browser screenshot required",
                "Run the web app, predict a digit, capture the browser screen, "
                "save as digitlens_runtime/app_screenshot.png, then regenerate this report."))
    story += [Spacer(1, 14), para("Features and application flow", "sub")]
    app_rows = [
        ("Predict digit", "Upload PNG/JPEG or use a MNIST test sample; show image, class and scores"),
        ("Model results", "Display saved metrics, training curves, example predictions and confusion matrix"),
        ("Train & setup", "CNN summary, instructions, configurable epochs and in-app training button"),
        ("Input handling", "Reject empty/invalid/oversized image; handle untrained or missing model"),
    ]
    story.append(simple_table(
        [[para("Tab / capability", "body"), para("User-visible output", "body")]] +
        [[para(escape(a), "small"), para(escape(b), "small")] for a, b in app_rows],
        widths=[140, width - 140], padd=6))
    story += [Spacer(1, 12),
              para("The interface explains model uncertainty, explicitly distinguishes "
                   "softmax confidence from a guarantee, and warns that uploaded photos "
                   "can differ from MNIST's clean training distribution.", "small"),
              PageBreak()]

    # PAGE 6 - CONCLUSION, RUNBOOK, SOURCES
    story += [para("05  Conclusion and reproducibility", "heading"),
              para("Results summary", "sub")]
    if metrics:
        story.append(para(
            f"The network was trained for {metrics['epochs']} epoch(s) on 54,000 MNIST "
            f"training images and selected against 6,000 validation images. "
            f"It achieved {metrics['test_accuracy']:.2%} accuracy on 10,000 "
            "held-out test images. This is the measured result of the recorded run."))
    else:
        story.append(para(
            "The model, optimizer, image preprocessing and empty-input handling "
            "have passed an offline synthetic smoke test. The genuine MNIST train/test "
            "experiment cannot be certified until the real dataset has been downloaded "
            "and training, metrics and screenshots have been recorded."))
    story += [para("Challenges and improvements", "sub"),
              para("<b>Potential challenges:</b> ambiguity in handwritten strokes, "
                   "out-of-distribution phone photos, white/black background variation, "
                   "and computing time. Some digit pairs may be visually similar."),
              para("<b>Improvements:</b> collect real user handwriting, use controlled "
                   "rotation/translation augmentation on training data only, add a "
                   "confidence-based retake prompt, and calibrate probabilities."),
              para("How to run this project", "sub")]
    commands = (
        "python -m pip install torch torchvision streamlit numpy pillow matplotlib reportlab\n"
        "python Varun_CNN_Streamlit_Project.py --smoke-test\n"
        "python Varun_CNN_Streamlit_Project.py --train --epochs 3\n"
        "python -m streamlit run Varun_CNN_Streamlit_Project.py\n"
        "# Save browser screenshot: digitlens_runtime/app_screenshot.png\n"
        "python Varun_CNN_Streamlit_Project.py --export-report"
    )
    box = Table([[Preformatted(commands, ParagraphStyle("commands", fontName="Courier",
                      fontSize=7.7, leading=12.3, textColor=dark))]], colWidths=[width])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), soft),
                              ("BOX", (0, 0), (-1, -1), .7, line),
                              ("LEFTPADDING", (0, 0), (-1, -1), 12),
                              ("TOPPADDING", (0, 0), (-1, -1), 10),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    story += [box, Spacer(1, 12),
              para("Submission checklist", "sub"),
              para("[ ] One Python file with both parts  [ ] MNIST test metrics  "
                   "[ ] Accuracy/loss plots  [ ] Correct and incorrect samples  "
                   "[ ] Actual Streamlit screenshot  [ ] PDF 4-6 pages", "small"),
              para("References", "sub"),
              para("1. LeCun et al., MNIST database of handwritten digits: "
                   "https://yann.lecun.com/exdb/mnist/<br/>"
                   "2. PyTorch and TorchVision documentation: https://pytorch.org/vision/<br/>"
                   "3. Streamlit documentation: https://docs.streamlit.io/", "tiny"),
              Spacer(1, 6),
              para("<b>Integrity note:</b> Illustrative/test-only checks are clearly "
                   "distinguished from measured MNIST results. This report is regenerated "
                   "from real JSON metrics and saved plots after execution.", "small")]

    doc.build(story, onFirstPage=frame, onLaterPages=frame)
    print(f"Report created: {pdf_path}")
    if not metrics:
        print("PENDING: real MNIST results; run --train then --export-report again.")
    if not screenshot_path.is_file():
        print("PENDING: real Streamlit screenshot digitlens_runtime/app_screenshot.png")
    return pdf_path

def is_streamlit_runtime():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except (ImportError, RuntimeError, TypeError):
        return False


def main():
    if is_streamlit_runtime():
        streamlit_app()
        return
    parser = argparse.ArgumentParser(
        description="DigitLens — MNIST CNN + Streamlit in one Python file")
    parser.add_argument("--train", action="store_true", help="Download MNIST, train and evaluate")
    parser.add_argument("--smoke-test", action="store_true", help="Run offline functional checks")
    parser.add_argument("--summary", action="store_true", help="Print CNN architecture")
    parser.add_argument("--export-report", action="store_true", help="Regenerate six-page PDF")
    parser.add_argument("--student-name", default="Varun D. Sawalkar")
    parser.add_argument("--batch", default="September Batch")
    parser.add_argument("--app-screenshot", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
    elif args.train:
        train_project(epochs=args.epochs, batch_size=args.batch_size)
    elif args.summary:
        print(model_summary())
    elif args.export_report:
        export_pdf_report(student_name=args.student_name,
                          batch=args.batch, app_screenshot=args.app_screenshot)
    else:
        parser.print_help()
        print("\nLaunch dashboard: python -m streamlit run " + Path(__file__).name)


if __name__ == "__main__":
    main()
