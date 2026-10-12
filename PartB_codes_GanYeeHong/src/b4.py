"""Question B4: Grad-CAM on the B1-B3 architectures.

Five pneumonia X-rays are drawn at random from the test set and explained with
Captum's LayerGradCam, taken at the last convolutional stage of each network
and upsampled to the input resolution. Alongside the overlays, the pairwise
correlation between the models' maps is reported so the visual comparison has
a number behind it.

    uv run python src/b4.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import LayerAttribution, LayerGradCam

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import load_data
from models import AlexNetLike, ResNetLike, VGGLike

OUTPUT_DIR = (SRC_DIR.parent / "outputs" / "b4").resolve()
OUTPUTS = SRC_DIR.parent / "outputs"
SEED = 42
SAMPLE_COUNT = 5


def load_trained():
    """Load the exact models reported in B1-B3 and their Grad-CAM target layer."""
    b1 = AlexNetLike()
    b1.load_state_dict(torch.load(OUTPUTS / "b1" / "b1_model.pt"))
    b2 = VGGLike()
    b2.load_state_dict(torch.load(OUTPUTS / "b2" / "b2_model.pt"))
    b3 = ResNetLike()
    b3.load_state_dict(torch.load(OUTPUTS / "b3" / "b3_model.pt"))
    return {
        "B1 AlexNet-like": (b1, b1.features[4]),
        "B2 VGG-like": (b2, b2.features[7]),
        "B3 ResNet-like": (b3, b3.features[4]),
    }


def grad_cam(model, layer, images):
    """Upsampled, per-image normalised Grad-CAM maps."""
    model.eval()

    def forward(x):
        return model(x).unsqueeze(1)

    explainer = LayerGradCam(forward, layer)
    attribution = explainer.attribute(images, target=0, relu_attributions=True)
    upsampled = LayerAttribution.interpolate(
        attribution, (images.shape[2], images.shape[3]), interpolate_mode="bilinear"
    )
    maps = upsampled.squeeze(1).detach().numpy()
    normalised = []
    for single in maps:
        peak = single.max()
        normalised.append(single / peak if peak > 0 else single)
    return np.stack(normalised), tuple(attribution.shape[2:])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    images, labels = data["test"].tensors

    generator = torch.Generator().manual_seed(SEED)
    pneumonia = torch.nonzero(labels == 1).squeeze(1)
    chosen = pneumonia[torch.randperm(len(pneumonia), generator=generator)[:SAMPLE_COUNT]]
    chosen, _ = torch.sort(chosen)
    selected = images[chosen]
    print(f"selected pneumonia test indices: {chosen.tolist()}")

    display = (selected.squeeze(1).numpy() * data["std"] + data["mean"]).clip(0, 1)

    models = load_trained()
    cams = {}
    resolutions = {}
    for name, (model, layer) in models.items():
        cams[name], resolutions[name] = grad_cam(model, layer, selected)
        with torch.no_grad():
            probability = torch.sigmoid(model(selected)).numpy()
        print(
            f"{name:<18} CAM grid {resolutions[name][0]}x{resolutions[name][1]} | "
            f"p(pneumonia) " + " ".join(f"{p:.3f}" for p in probability)
        )

    names = list(models)
    correlations = {}
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            values = [
                float(np.corrcoef(cams[first][k].ravel(), cams[second][k].ravel())[0, 1])
                for k in range(SAMPLE_COUNT)
            ]
            correlations[f"{first} vs {second}"] = float(np.mean(values))
    print("\nmean pairwise Grad-CAM correlation across the five images:")
    for pair, value in correlations.items():
        print(f"  {pair:<42} {value:+.3f}")

    figure, axes = plt.subplots(
        4, SAMPLE_COUNT, figsize=(1.75 * SAMPLE_COUNT, 7.2)
    )
    for column in range(SAMPLE_COUNT):
        axes[0][column].imshow(display[column], cmap="gray", vmin=0, vmax=1)
        axes[0][column].set_title(f"test #{chosen[column].item()}", fontsize=8)
        for row, name in enumerate(names, start=1):
            axes[row][column].imshow(display[column], cmap="gray", vmin=0, vmax=1)
            axes[row][column].imshow(cams[name][column], cmap="jet", alpha=0.45,
                                     vmin=0, vmax=1)
    for row, label in enumerate(["X-ray"] + names):
        axes[row][0].set_ylabel(label, fontsize=8)
    for row in axes:
        for axis in row:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "fig_b4_gradcam.png", dpi=200)
    plt.close(figure)

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "indices": chosen.tolist(),
                "cam_resolution": {k: list(v) for k, v in resolutions.items()},
                "mean_pairwise_correlation": correlations,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
