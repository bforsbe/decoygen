"""Load decoygen output into ChimeraX, one tile per decoy.

Inside ChimeraX:
    runscript /path/to/chimerax_decoys.py /path/to/decoys [--halves] [--fraction 1.0] [--columns 4]
From a shell:
    chimerax --cmd "runscript /path/to/chimerax_decoys.py /path/to/decoys"

`/path/to/decoys` is searched recursively for decoy folders (a folder NAME holding NAME.json).
Per decoy, the noise-free map is shown as a surface contoured at the level that encloses
`fraction` x the envelope volume (from the JSON); the full noisy map, envelope mask, Turing
field (if written) and, with --halves, the half-maps are opened hidden at the same position.
Toggle them in the Models panel.
"""

import argparse
import json
import os
import sys

import numpy as np
from chimerax.core.commands import run
from chimerax.geometry import translation

COLORS = {"clean": "#b4a0e6", "full": "#808080", "mask": "#d0d0d0", "turing": "#e6b450",
          "half_1": "#7fb3d5", "half_2": "#f1948a"}


def find_decoys(root):
    found = []
    for d, _, files in sorted(os.walk(root)):
        name = os.path.basename(d)
        if f"{name}.json" in files:
            with open(os.path.join(d, f"{name}.json")) as fh:
                found.append((name, d, json.load(fh)))
    return found


def level_enclosing(v, volume_A3):
    m = v.full_matrix()
    apix = v.data.step[0]
    n = int(np.rint(volume_A3 / apix**3))   # step may be numpy floats
    flat = np.sort(m.ravel())[::-1]
    return float(flat[min(max(n, 0), flat.size - 1)])


def open_map(session, path, name):
    v = run(session, f'open "{path}"', log=False)[0]
    run(session, f'rename #{v.id_string} "{name}"', log=False)
    return v


def main(session, argv):
    ap = argparse.ArgumentParser(prog="chimerax_decoys.py")
    ap.add_argument("root", nargs="?", default=os.getcwd())
    ap.add_argument("--halves", action="store_true", help="also open the half-maps")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="contour the clean map to enclose this fraction of the envelope volume")
    ap.add_argument("--columns", type=int, default=4)
    args = ap.parse_args(argv)

    decoys = find_decoys(args.root)
    if not decoys:
        session.logger.warning(f"no decoy folders found under {args.root}")
        return
    models = []          # one list of volumes per decoy, clean map first
    for name, d, meta in decoys:
        files = meta["files"]
        clean = open_map(session, os.path.join(d, files["clean"]), f"{name} clean")
        vol = meta.get("diagnostics", {}).get("volume_A3") or meta["params"]["mass_kda"] * 1210.0
        level = level_enclosing(clean, args.fraction * vol)
        run(session, f"volume #{clean.id_string} style surface level {level:.6g} "
                     f"color {COLORS['clean']} step 1", log=False)
        group = [clean]
        keys = ["full", "mask", "turing"] + (["half_1", "half_2"] if args.halves else [])
        for key in keys:
            if key not in files:
                continue
            v = open_map(session, os.path.join(d, files[key]), f"{name} {key}")
            if key in ("mask", "turing"):
                run(session, f"volume #{v.id_string} style mesh level 0.5 color {COLORS[key]} step 1", log=False)
            else:
                run(session, f"volume #{v.id_string} style surface rmsLevel 3 color {COLORS[key]} step 1",
                    log=False)
            run(session, f"hide #{v.id_string} models", log=False)
            group.append(v)
        models.append(group)
        diag = meta.get("diagnostics", {})
        session.logger.info(
            f"{name}: clean level {level:.4g}; D={diag.get('richardson_D', float('nan')):.3f}, "
            f"FSC {diag.get('halfmap_fsc_resolution_A', float('nan')):.2f} A, "
            f"linearity {diag.get('structure_tensor_linearity_mean', float('nan')):.3f}")

    # Grid layout by box size (the `tile` command needs the GUI's mouse modes).
    for i, group in enumerate(models):
        clean = group[0]
        extent = max(s * st for s, st in zip(clean.data.size, clean.data.step))
        row, col = divmod(i, args.columns)
        place = translation((1.15 * extent * col, -1.15 * extent * row, 0.0))
        for v in group:
            v.position = place
    run(session, "view; lighting soft", log=False)


main(globals()["session"], sys.argv[1:])
