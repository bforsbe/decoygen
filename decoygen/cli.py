"""Command line: decoygen generate | batch | calibrate."""

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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="decoygen", description=__doc__)
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

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
