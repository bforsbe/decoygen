"""Evaluation maps for one rung, exactly as experiments/deepemhancer/prepare.py makes them.

    python experiments/ladder/maps.py runs/ladder/main/3_threaded_chain

Renders atoms.* (B = 40 A^2, 1.0 A/px, 144^3) and adds two independent white-noise half-maps
whose FSC crosses 0.143 at 3.5 A. Writes maps/half1, half2, full (their mean), truth, support.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "deepemhancer"))

from prepare import APIX, BFACTOR, BOX, RESOLUTION, load_atoms

from decoygen import atoms, io, noise
from decoygen.fourier import Grid


def main(d: Path, seed: int = 0):
    src = min(d.glob("atoms.*"))
    grid = Grid(BOX, APIX, "cpu")
    pos, el = load_atoms(src)
    signal = atoms.render(grid, pos, el, BFACTOR)
    h1, h2, info = noise.half_maps(grid, signal, RESOLUTION, 1.0, "white", seed)
    support = grid.gaussian_filter(signal, 2.0)
    support = support > 0.2 * support.max()
    (d / "maps").mkdir(exist_ok=True)
    for name, m in (("half1", h1), ("half2", h2), ("full", (h1 + h2) / 2), ("truth", signal),
                    ("support", support.float())):
        io.write_mrc(d / "maps" / f"{name}.mrc", m.numpy(), APIX)
    print(f"{d.name}: {len(pos)} atoms, noise sd {info['noise_sd_real']:.4g}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
