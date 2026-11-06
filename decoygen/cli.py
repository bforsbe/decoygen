"""Command line.

Decoy maps (v0):   decoygen generate | batch | calibrate
Envelopes:         decoygen envelope (optionally symmetric: --symmetry C3, divided into subunits)
Atom-model ladder: decoygen gas | hardcore | polymer | thread | peptide | sidechains | mirror | render |
                   info | ladder | expand
                   (each model step makes the next rung of the protein-likeness ladder from the one
                   below and refuses any other input; see decoygen/model.py)
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import types
import typing
from pathlib import Path

import numpy as np

from .params import DecoyParams


def _field_type(f: dataclasses.Field):
    hints = typing.get_type_hints(DecoyParams)
    t = hints[f.name]
    args = [a for a in typing.get_args(t) if a is not type(None)]
    if typing.get_origin(t) in (typing.Union, types.UnionType):
        t = args[0]
    return t


def add_param_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("decoy parameters (see DecoyParams)")
    for f in dataclasses.fields(DecoyParams):
        flag = "--" + f.name.replace("_", "-")
        t = _field_type(f)
        if t is bool:
            g.add_argument(flag, action="store_true", default=None)
        elif typing.get_origin(t) is tuple:
            g.add_argument(flag, type=float, nargs="+", default=None)
        else:
            g.add_argument(flag, type=t, default=None)
    p.add_argument("--params-json", help="JSON file with parameters (CLI flags override)")


def params_from_args(args) -> DecoyParams:
    d = {}
    if args.params_json:
        d.update(json.loads(Path(args.params_json).read_text()))
    for f in dataclasses.fields(DecoyParams):
        v = getattr(args, f.name)
        if v is not None:
            d[f.name] = v
    return DecoyParams.from_dict(d)


def default_device() -> str:
    import torch

    # is_available() can be True on nodes whose GPU is reserved (e.g. login nodes); probe it.
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return "cuda"
        except RuntimeError:
            pass
    return "cpu"


def cmd_generate(args) -> None:
    from .pipeline import generate, save

    params = params_from_args(args)
    decoy = generate(params, device=args.device or default_device())
    d = save(decoy, args.out, name=args.name, plots=args.plots, extra_maps=args.extra_maps)
    print(f"wrote {d}")
    print(json.dumps({k: v for k, v in decoy.diagnostics.items() if not isinstance(v, dict)}, indent=1))


def cmd_envelope(args) -> None:
    from .fourier import Grid
    from .roughness import roughness_parameters
    from .symmetry import make_symmetric_envelope, save

    params = params_from_args(args)
    rough = roughness_parameters(params.roughness_D, params.alpha, params.roughness_amplitude)
    env = make_symmetric_envelope(
        Grid(params.box, params.apix, args.device or default_device()), volume=params.volume,
        alpha=rough["alpha"], roughness_amplitude=rough["roughness_amplitude"], compactness=params.compactness,
        bias=params.envelope_bias, clearance=params.edge_clearance, seed=params.seed,
        **_symmetry_kw(args, params.seed))
    name = args.name or f"{args.symmetry.lower()}_{params.seed:06d}"
    d = save(env, args.out, name, params.apix, {**dataclasses.asdict(params), **rough, "contact": args.contact,
                                                "swap": args.swap, "axis": args.axis, "shift_xyz": args.shift,
                                                "owner_fine": args.owner_fine, "owner_sigma": args.owner_sigma})
    i = env.info
    print(f"wrote {d}/{name}_*  (ChimeraX: cd {d}; open {name}.cxc)")
    print(f"{i['symmetry']}: {i['volume_A3']:.0f} A^3 (target {i['target_volume_A3']:.0f}); subunits "
          f"{', '.join(f'{v:.0f}' for v in i['subunit_volumes_A3'])} A^3, pieces {i['subunit_pieces']}; "
          f"interface {i['interface_A2_per_subunit']:.0f} A^2/subunit (voxel faces); "
          f"symmetry mismatch {100 * i['symmetry_mismatch_fraction_max']:.2f} %")


def _flat(prefix: str, d: dict) -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(key + ".", v))
        elif isinstance(v, (list, tuple)):
            out[key] = ";".join(str(x) for x in v)
        else:
            out[key] = v
    return out


def cmd_batch(args) -> None:
    from .pipeline import decoy_id, generate, save

    base = params_from_args(args)
    device = args.device or default_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(args.start_seed, args.start_seed + args.n):
        params = dataclasses.replace(base, seed=seed)
        decoy = generate(params, device=device)
        d = save(decoy, out, plots=args.plots, extra_maps=args.extra_maps)
        name = decoy_id(seed)
        row = {"id": name, "path": str(d.relative_to(out)), "map": f"{name}.mrc",
               "half_1": f"{name}_half_1.mrc", "half_2": f"{name}_half_2.mrc",
               "clean": f"{name}_clean.mrc", "mask": f"{name}_mask.mrc"}
        row.update(_flat("param.", params.to_dict()))
        row.update(_flat("diag.", decoy.diagnostics))
        row["alpha_used"] = decoy.info["alpha"]
        row["roughness_amplitude_used"] = decoy.info["roughness_amplitude"]
        rows.append(row)
        print(f"{name}: {decoy.info['timings_s']}", flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} decoys and manifest.csv to {out}")


def cmd_calibrate(args) -> None:
    from .calibration import calibrate
    from .io import write_json
    from .roughness import CALIBRATION_FILE

    base = params_from_args(args)
    alphas = np.round(np.arange(args.alpha_min, args.alpha_max + 1e-9, args.alpha_step), 3)
    seeds = list(range(args.seeds))
    table = calibrate(base, alphas, args.amplitudes, seeds, device=args.device or default_device())
    out = Path(args.out) if args.out else CALIBRATION_FILE
    write_json(out, table)
    print(f"wrote {out}")


# ---------------------------------------------------------------- protein-likeness ladder

def _log(r: dict) -> None:
    print("  step {step}: ".format(**r) + " ".join(f"{k} {v:.4g}" for k, v in r.items()
                                                    if k not in ("step", "loss")), flush=True)


def _save(m, args, info=None) -> None:
    from .model import default_path, write

    path = Path(args.out) if args.out else default_path(Path(args.input), m.rung, m.mirrored)
    write(path, m, overwrite=args.overwrite)
    if info is not None:
        from .io import write_json

        write_json(path.with_name(path.stem + "_info.json"), info)
        print(json.dumps({k: v for k, v in info.items() if not isinstance(v, (dict, list))}, indent=1))
    print(f"wrote {path}: {m.describe().splitlines()[0]}")


def _read(args):
    from .model import read

    return read(args.input)


def cmd_gas(args) -> None:
    from .ladder import gas
    from .model import write

    params = params_from_args(args)
    m = gas(params, frame_box=args.frame_box, device=args.device or default_device(),
            **_symmetry_kw(args, params.seed))
    write(args.out, m, overwrite=args.overwrite)
    print(f"wrote {args.out}: {m.describe().splitlines()[0]}")


def cmd_hardcore(args) -> None:
    from .ladder import hardcore

    m, info = hardcore(_read(args), steps=args.steps, seed=args.seed, device=args.device or default_device(),
                       force=args.force, log=_log)
    _save(m, args, info)


def cmd_polymer(args) -> None:
    from .ladder import build_polymer

    m, info = build_polymer(_read(args), steps=args.steps, seed=args.seed, device=args.device or default_device(),
                            force=args.force, log=_log)
    _save(m, args, info)


def cmd_thread(args) -> None:
    from .ladder import thread_chain

    m, info = thread_chain(_read(args), split=args.split or None, prune=args.prune, relax=args.relax,
                           conserve=not args.no_conserve, seed=args.seed, device=args.device or default_device(), force=args.force, log=_log)
    _save(m, args, info)


def cmd_peptide(args) -> None:
    from .ladder import make_peptide

    m, info = make_peptide(_read(args), steps=args.steps, keep_atoms=not args.add_atoms, seed=args.seed,
                           device=args.device or default_device(), force=args.force, log=_log)
    _save(m, args, info)


def cmd_sidechains(args) -> None:
    from .ladder import make_sidechains

    m, info = make_sidechains(_read(args), fraction=args.fraction, steps=args.steps, stages=args.stages,
                              seed=args.seed, device=args.device or default_device(), force=args.force, log=_log)
    _save(m, args, info)


def cmd_mirror(args) -> None:
    from .ladder import mirror

    _save(mirror(_read(args)), args)


def cmd_render(args) -> None:
    from .ladder import render

    meta = render(_read(args), Path(args.out), apix=args.apix, resolution=args.resolution,
                  snr_scale=args.snr_scale, seed=args.seed, device=args.device or default_device())
    print(f"wrote {args.out}: truth, half1, half2, full, support ({meta['box_voxels']}^3 at {args.apix} A/px)")


def cmd_expand(args) -> None:
    from .model import expand, write

    m = _read(args)
    if not m.symmetry:
        sys.exit(f"{args.input} has no symmetry to expand")
    full = expand(m)
    write(args.out, full, overwrite=args.overwrite)
    print(f"wrote {args.out}: {m.symmetry['group']}, {len(m.symmetry['ops'])} copies, {full.n} atoms")


def cmd_info(args) -> None:
    print(f"{args.input}: {_read(args).describe()}")


def cmd_ladder(args) -> None:
    from . import ladder
    from .model import write

    device = args.device or default_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = params_from_args(args)
    kw = {"seed": params.seed, "device": device, "log": _log}

    def save(m, info=None):
        from .model import default_path

        path = default_path(out / "x", m.rung, m.mirrored)
        write(path, m, overwrite=args.overwrite)
        if info is not None:
            from .io import write_json

            write_json(path.with_name(path.stem + "_info.json"), info)
        print(f"wrote {path}", flush=True)
        return m

    m1 = save(ladder.gas(params, frame_box=args.frame_box, device=device, **_symmetry_kw(args, params.seed)))
    save(*ladder.hardcore(m1, steps=args.steps, **kw))
    m = m1
    steps = [(2, lambda m: ladder.build_polymer(m, steps=args.steps, **kw)),
             (3, lambda m: ladder.thread_chain(m, relax=args.steps, **kw)),
             (4, lambda m: ladder.make_peptide(m, steps=args.steps, **kw)),
             (5, lambda m: ladder.make_sidechains(m, steps=args.steps, **kw))]
    for rung, step in steps:
        if rung > args.to:
            break
        m = save(*step(m))


def _symmetry_args(p) -> None:
    g = p.add_argument_group("symmetry (docs/SYMMETRY.md)")
    g.add_argument("--symmetry", default="C1", help="point group: Cn for now (C1 = none); the model is one "
                                                      "asymmetric unit and --mass-kda the whole assembly's")
    g.add_argument("--contact", type=float, default=0.6, help="overlap of neighbouring subunit bowls (radii)")
    g.add_argument("--swap", type=float, default=0.0, help="domain swap strength 0-1")
    g.add_argument("--axis", default="z", help="symmetry axis: z, random, or x,y,z")
    g.add_argument("--shift", type=float, nargs=3, metavar=("X", "Y", "Z"), help="axis offset from the centre (A)")
    g.add_argument("--owner-fine", type=float, default=0.3,
                   help="weight of the fine (rough) field in subunit ownership; higher = rougher interfaces")
    g.add_argument("--owner-sigma", type=float, default=1.5, help="ownership smoothing (voxels)")


def _symmetry_kw(args, seed: int) -> dict:
    import numpy as np

    if args.axis == "z":
        axis = np.array([1.0, 0, 0])
    elif args.axis == "random":
        v = np.random.default_rng(seed).normal(size=3)
        axis = v / np.linalg.norm(v)
    else:
        axis = np.array([float(x) for x in args.axis.split(",")])[::-1]          # given as x,y,z
    shift = np.array(args.shift[::-1], dtype=float) if args.shift else None    # given as x y z (A)
    return {"symmetry": args.symmetry, "contact": args.contact, "swap": args.swap, "axis": axis, "shift": shift,
            "owner_fine": args.owner_fine, "owner_sigma": args.owner_sigma}


def _model_args(p, steps: bool = True) -> None:
    p.add_argument("input", help="input model (PDB with a decoygen header)")
    p.add_argument("-o", "--out", help="output PDB (default: rung<N>_<name>.pdb next to the input)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--force", action="store_true", help="skip the rung check (recorded in the history)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device")
    if steps:
        p.add_argument("--steps", type=int, default=3000, help="relaxation steps")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="decoygen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate", help="generate one decoy")
    p.add_argument("--out", required=True)
    p.add_argument("--name")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--extra-maps", action="store_true", help="also write the Turing field")
    p.add_argument("--device")
    add_param_args(p)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("batch", help="generate decoys for consecutive seeds + manifest.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--extra-maps", action="store_true")
    p.add_argument("--device")
    add_param_args(p)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("envelope", help="envelope only, optionally symmetric and divided into subunits",
                       description="Envelope (mask) only. With --symmetry, copies of one subunit field compete "
                                   "for space: the assembly is their union, each voxel belongs to the strongest "
                                   "copy, and interfaces form where copies meet (docs/SYMMETRY.md). "
                                   "--mass-kda is the whole assembly's mass.")
    p.add_argument("--out", required=True)
    p.add_argument("--name")
    _symmetry_args(p)
    p.add_argument("--device")
    add_param_args(p)
    p.set_defaults(func=cmd_envelope)

    p = sub.add_parser("calibrate", help="build the (alpha, amplitude) -> D lookup table")
    p.add_argument("--alpha-min", type=float, default=3.8)
    p.add_argument("--alpha-max", type=float, default=5.0)
    p.add_argument("--alpha-step", type=float, default=0.2)
    p.add_argument("--amplitudes", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.7])
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--out", help="default: packaged decoygen/data/roughness_calibration.json")
    p.add_argument("--device")
    add_param_args(p)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("gas", help="rung 1: the v0 decoy's atom gas, as a model")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--frame-box", type=float, help="centre in a frame of this edge (A), e.g. 144")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device")
    _symmetry_args(p)
    add_param_args(p)
    p.set_defaults(func=cmd_gas)

    p = sub.add_parser("hardcore", help="rung 1 -> rung 0: push the gas atoms apart, no bonds")
    _model_args(p)
    p.set_defaults(func=cmd_hardcore)

    p = sub.add_parser("polymer", help="rung 1 -> rung 2: bonded tree with protein geometry")
    _model_args(p)
    p.set_defaults(func=cmd_polymer)

    p = sub.add_parser("thread", help="rung 2 -> rung 3: one linear chain")
    _model_args(p, steps=False)
    p.add_argument("--split", type=int, default=5, help="cut the tree into longest paths of >= SPLIT atoms (0: off)")
    p.add_argument("--prune", type=int, default=4, help="max bonds from the backbone for side atoms")
    p.add_argument("--relax", type=int, default=3000, help="relaxation steps")
    p.add_argument("--no-conserve", action="store_true",
                   help="keep all atoms: bridges then add mass (~15 %%) to the envelope")
    p.set_defaults(func=cmd_thread)

    p = sub.add_parser("peptide", help="rung 3 -> rung 4: N-CA-C(=O) backbone, trans peptides, L-amino acids")
    _model_args(p)
    p.add_argument("--add-atoms", action="store_true",
                   help="add the carbonyl O atoms on top (default: trim side atoms to keep the atom count)")
    p.set_defaults(func=cmd_peptide)

    p = sub.add_parser("sidechains", help="rung 4 -> rung 5: thin to protein residues per mass, real side chains")
    _model_args(p)
    p.add_argument("--fraction", type=float, default=0.2, help="max fraction of residues excised per pass")
    p.add_argument("--stages", type=int, default=5, help="relaxations over which new bonds are switched on")
    p.set_defaults(func=cmd_sidechains)

    p = sub.add_parser("mirror", help="any rung -> its mirror image (last step only)")
    _model_args(p, steps=False)
    p.set_defaults(func=cmd_mirror)

    p = sub.add_parser("render", help="model -> truth map, noisy half-maps, their mean, support")
    p.add_argument("input")
    p.add_argument("-o", "--out", required=True, help="output directory")
    p.add_argument("--apix", type=float, default=1.0)
    p.add_argument("--resolution", type=float, default=3.5, help="FSC = 0.143 here")
    p.add_argument("--snr-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("expand", help="symmetric model -> the whole assembly (one chain per copy)")
    p.add_argument("input")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_expand)

    p = sub.add_parser("info", help="rung and history of a model")
    p.add_argument("input")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("ladder", help="build rungs 0..TO in one directory (gas, hardcore, polymer, ...)")
    p.add_argument("--to", type=int, default=5, choices=range(1, 6))
    p.add_argument("--out", required=True)
    p.add_argument("--frame-box", type=float)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device")
    _symmetry_args(p)
    add_param_args(p)
    p.set_defaults(func=cmd_ladder)

    args = ap.parse_args(argv)
    from .model import RungError

    try:
        args.func(args)
    except (RungError, FileExistsError) as e:
        ap.exit(2, f"decoygen {args.cmd}: {e}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
