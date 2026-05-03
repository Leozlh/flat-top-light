from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def save_field_visualizations(
    output_dir: Path,
    target_amp: torch.Tensor,
    target_phase: torch.Tensor,
    out_amp: torch.Tensor,
    out_phase: torch.Tensor,
    slm_phase: torch.Tensor,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ta = _to_numpy(target_amp)
    tp = _to_numpy(target_phase)
    oa = _to_numpy(out_amp)
    op = _to_numpy(out_phase)
    sp = _to_numpy(slm_phase)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    items = [
        (ta**2, "Target Intensity", "viridis"),
        (tp, "Target Phase", "twilight"),
        (oa**2, "Output Intensity", "viridis"),
        (op, "Output Phase", "twilight"),
        (sp, "SLM Phase", "twilight"),
        ((ta**2) - (oa**2), "Intensity Error", "coolwarm"),
    ]
    for ax, (img, title, cmap) in zip(axes.flat, items):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "summary.png", dpi=180)
    plt.close(fig)


def save_linecuts(
    output_dir: Path,
    target_amp: torch.Tensor,
    out_amp: torch.Tensor,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ta = _to_numpy(target_amp**2)
    oa = _to_numpy(out_amp**2)

    cy = ta.shape[0] // 2
    cx = ta.shape[1] // 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ta[cy, :], label="target")
    axes[0].plot(oa[cy, :], label="output")
    axes[0].set_title("Horizontal Linecut")
    axes[0].legend()

    axes[1].plot(ta[:, cx], label="target")
    axes[1].plot(oa[:, cx], label="output")
    axes[1].set_title("Vertical Linecut")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "linecuts.png", dpi=180)
    plt.close(fig)

