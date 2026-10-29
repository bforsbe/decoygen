"""Per-decoy diagnostics figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import FSC_THRESHOLD


def summary_figure(decoy, path) -> None:
    maps, c, dg, p = decoy.maps, decoy.curves, decoy.diagnostics, decoy.params
    n = p.box
    fig, ax = plt.subplots(2, 4, figsize=(18, 8.5))

    for a, (key, title) in zip(ax[0], [("clean", "clean (central z slice)"), ("full", "full map"),
                                       ("turing", "Turing field"), ("mask", "envelope")]):
        a.imshow(maps[key][n // 2], cmap="gray", origin="lower")
        a.set_title(title)
        a.set_xticks([])
        a.set_yticks([])

    s = c["shell_s"]
    nyq = n // 2 + 1
    a = ax[1, 0]
    a.semilogy(s[1:nyq], c["power_target"][1:nyq], "k-", label="target")
    a.semilogy(s[1:nyq], c["power_reference"][1:nyq], "C0:", label="Poisson gas reference")
    a.semilogy(s[1:nyq], c["power_signal"][1:nyq], "C1-", alpha=0.8, label="decoy signal")
    a.axvline(1 / p.resolution, color="gray", lw=0.8)
    a.set_xlabel("s (1/Å)")
    a.set_title(f"radial power (log10 rms err {dg['spectrum_log10_rms_error']:.3f})")
    a.legend(fontsize=8)

    a = ax[1, 1]
    sig = maps["clean"][maps["mask"] > 0]
    a.hist(sig, bins=100, density=True, color="C1", alpha=0.7)
    a.set_title("clean-map histogram inside envelope")

    a = ax[1, 2]
    r = c["richardson"]
    ratio = np.array(r["areas"]) / np.array(r["ball_areas"])
    a.loglog(r["sigmas"], ratio, "o")
    fit = np.exp(np.polyval(r["fit"], np.log(r["sigmas"])))
    a.loglog(r["sigmas"], fit, "k-")
    a.set_xlabel("blur σ (Å)")
    a.set_ylabel("area / equal-volume ball area")
    a.set_title(f"Richardson D = {dg['richardson_D']:.3f} (α = {decoy.info['alpha']:.2f})")

    a = ax[1, 3]
    a.plot(s[1:nyq], c["fsc"][1:nyq])
    a.axhline(FSC_THRESHOLD, color="gray", lw=0.8)
    a.axvline(1 / p.resolution, color="gray", lw=0.8)
    a.set_xlabel("s (1/Å)")
    a.set_title(f"half-map FSC: {dg['halfmap_fsc_resolution_A']:.2f} Å")

    fig.suptitle(f"decoy seed {p.seed}: {p.mass_kda} kDa, D={p.roughness_D}, "
                 f"meso λ={p.meso_wavelength} Å strength {p.meso_strength}, "
                 f"fill {p.fill_fraction}, target {p.target_spectrum}")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
