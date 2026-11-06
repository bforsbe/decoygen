"""Slice views of symmetric envelopes made by `decoygen envelope --symmetry ...`.

For each NAME: the subunit labels in the slice through the symmetry centre perpendicular to the axis
(top), and in the slice containing the axis (bottom). The axis is the red line or cross. Oblique axes
are handled by resampling the label map in a frame aligned with the axis.

usage: python slices3d.py DIR OUT.png NAME [NAME ...]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import ndimage

from decoygen.symmetry import COLOURS, rotation_from_z

SOLVENT = "#f6f6f4"
HALF = 50                                   # half-width of the views (A)


def frame_slices(labels, centre, axis, apix):
    """Labels on two planes through `centre` in A: perpendicular to the axis and containing it."""
    P = rotation_from_z(axis)                                       # columns: axis, e1, e2 in [z, y, x]
    t = np.arange(-HALF, HALF + 1, 1.0)
    a, b = np.meshgrid(t, t, indexing="ij")
    perp = centre + a[..., None] * P[:, 1] + b[..., None] * P[:, 2]   # rows e1, columns e2
    along = centre + a[..., None] * P[:, 0] + b[..., None] * P[:, 1]  # rows axis, columns e1

    def sample(pts):
        return ndimage.map_coordinates(labels, (pts / apix).transpose(2, 0, 1), order=0, mode="constant")

    return sample(perp), sample(along)


def main(directory, out, names):
    d = Path(directory)
    fig, axs = plt.subplots(2, len(names), figsize=(2.6 * len(names), 5.6), layout="constrained")
    axs = np.atleast_2d(axs).reshape(2, -1)
    for j, name in enumerate(names):
        meta = json.loads((d / f"{name}.json").read_text())["symmetry"]
        with mrcfile.open(d / f"{name}_subunits.mrc") as m:
            labels = m.data.astype(np.int64)
            apix = float(m.voxel_size.x)
        n = meta["n_subunits"]
        perp, along = frame_slices(labels, np.array(meta["centre_zyx_A"]), np.array(meta["axis_zyx"]), apix)
        cmap = ListedColormap([SOLVENT] + COLOURS[:n])
        title = (f"{name}\n{meta['symmetry']} contact {meta['contact']}"
                 + (f" swap {meta['swap']}" if meta["swap"] else "")
                 + f"\n{meta['interface_A2_per_subunit']:.0f} A² faces/subunit")
        for i, (img, ttl) in enumerate([(perp, title), (along, "slice containing the axis")]):
            ax = axs[i, j]
            ax.imshow(img, cmap=cmap, vmin=-0.5, vmax=n + 0.5, origin="lower", interpolation="nearest",
                      extent=(-HALF, HALF, -HALF, HALF))
            for k in range(n):
                if (img == k + 1).any():
                    ax.contour(img == k + 1, levels=[0.5], colors="#1f2328", linewidths=0.7, origin="lower",
                               extent=(-HALF, HALF, -HALF, HALF))
            if i == 0:
                ax.plot(0, 0, "+", color="#d62728", ms=10, mew=2)
            else:
                ax.axvline(0, color="#d62728", lw=1.2)              # columns = e1, rows = axis: axis vertical
            ax.set_title(ttl, fontsize=8.5, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
