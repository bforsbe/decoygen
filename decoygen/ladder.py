"""The model steps of the protein-likeness ladder: each makes one rung from the one below.

    gas       DecoyParams           -> rung 1  atom gas (the v0 decoy's atoms)
    hardcore  rung 1                -> rung 0  the same atoms, pushed apart, no bonds
    polymer   rung 1                -> rung 2  bonded tree with protein bond/angle/contact distances
    thread    rung 2                -> rung 3  one linear chain
    peptide   rung 3                -> rung 4  N-CA-C(=O) repeat, trans planar peptides, L at CA
    sidechains rung 4               -> rung 5  thinned to protein's residues per mass, real side chains
    mirror    any rung              -> same rung, mirrored (always the last step)
    render    any model             -> noise-free map, two noisy half-maps, their mean, support

Every step checks its input's rung (model.require) and records itself in the output's history.
Steps after gas relax inside the input model's own envelope (model.envelope_mask).

Symmetric models (gas with symmetry="C3" etc.) hold one asymmetric unit: every step works on that unit
and relaxes it against its symmetry images (polymer.Envelope), so interfaces are built from both sides
at every rung; render expands the assembly.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from . import atoms, io, noise, peptide, polymer, sidechains, thread
from .constants import MASS_PER_HEAVY_ATOM
from .envelope import make_envelope
from .fourier import Grid
from .model import Model, envelope_mask, expand, require
from .params import DecoyParams
from .roughness import roughness_parameters
from .turing import turing_field

RENDER_BFACTOR = 40.0
SUPPORT_SIGMA, SUPPORT_LEVEL = 2.0, 0.2


def _envelope(m: Model, device) -> polymer.Envelope:
    """The input's own envelope; for a symmetric model, the whole assembly's, with the operators."""
    if not m.symmetry:
        return polymer.Envelope(envelope_mask(m.pos, m.box), 1.0, device)
    return polymer.Envelope(envelope_mask(expand(m).pos, m.box), 1.0, device, ops=m.symmetry["ops"],
                            centre=m.symmetry["centre"])


def _tensor(m: Model, device) -> torch.Tensor:
    return torch.as_tensor(m.pos, dtype=torch.float32, device=device)


def gas(params: DecoyParams, *, frame_box: float | None = None, device: str = "cpu", symmetry: str | None = None,
        contact: float = 0.6, swap: float = 0.0, axis: np.ndarray | None = None,
        shift: np.ndarray | None = None) -> Model:
    """The default pipeline's envelope and atom gas (pipeline.generate up to sample_gas), centred
    in a frame_box A frame (default: the decoy's own box).

    symmetry ("C3", ...): the envelope is a symmetric assembly (symmetry.make_symmetric_envelope;
    params.mass_kda is the whole assembly's) and the model is one asymmetric unit: the subunit's share
    of the atoms, drawn in subunit 1's region, with the operators."""
    params.validate()
    g = Grid(params.box, params.apix, device)
    rough = roughness_parameters(params.roughness_D, params.alpha, params.roughness_amplitude)
    sym = None
    if symmetry and symmetry.upper() != "C1":
        from .symmetry import make_symmetric_envelope

        env = make_symmetric_envelope(
            g, symmetry=symmetry, volume=params.volume, alpha=rough["alpha"],
            roughness_amplitude=rough["roughness_amplitude"], compactness=params.compactness,
            bias=params.envelope_bias, clearance=params.edge_clearance, seed=params.seed, contact=contact,
            swap=swap, axis=axis, shift=shift)
        mask, region, sym = env.mask, env.labels == 1, env
    else:
        mask, _ = make_envelope(
            g, volume=params.volume, alpha=rough["alpha"], roughness_amplitude=rough["roughness_amplitude"],
            compactness=params.compactness, n_domains=params.n_domains, elongation=params.elongation,
            bias=params.envelope_bias, clearance=params.edge_clearance, seed=params.seed)
        region = mask
    turing, _ = turing_field(
        g, mask, wavelength=params.meso_wavelength, quotient=params.turing_quotient,
        fill_fraction=params.fill_fraction, iterations=params.turing_iterations, dt=params.turing_dt,
        seed=params.seed)
    m = torch.as_tensor(region, device=device)
    intensity = ((1 - params.meso_strength) + params.meso_strength * turing) * m
    n_copies = 1 if sym is None else sym.info["n_subunits"]
    n_atoms = round(params.mass_kda * 1e3 / MASS_PER_HEAVY_ATOM / n_copies)
    pos, el = atoms.sample_gas(g, intensity, n_atoms, params.seed, "atoms")
    own = params.box * params.apix
    frame = float(frame_box or own)
    if frame < own:
        raise ValueError(f"frame_box {frame} A is smaller than the decoy box {own} A")
    defaults = DecoyParams()
    changed = {f.name: getattr(params, f.name) for f in dataclasses.fields(params)
               if f.name != "seed" and getattr(params, f.name) != getattr(defaults, f.name)}
    if sym is not None:
        changed.update(symmetry=sym.info["symmetry"], contact=contact, swap=swap,
                       axis=",".join(f"{v:.4f}" for v in sym.info["axis_zyx"][::-1]))
    off = (frame - own) / 2
    out = Model(pos=pos.cpu().numpy().astype(np.float64) + off, els=el.cpu().numpy(),
                edges=np.zeros((0, 2), dtype=np.int64), box=frame,
                symmetry=None if sym is None else {"group": sym.info["symmetry"], "centre": sym.centre + off,
                                                   "ops": sym.ops})
    return out.then(1, "gas", seed=params.seed, frame_box=frame,
                    **{k: str(v).replace(" ", "") for k, v in changed.items()})


def hardcore(m: Model, *, steps: int = 3000, seed: int = 0, device: str = "cpu", force=False, log=None):
    require(m, 1, "hardcore", force)
    x, info = polymer.relax(_tensor(m, device), m.els, np.zeros((0, 2), dtype=np.int64), polymer.load_reference(),
                            envelope=_envelope(m, device), steps=steps, seed=seed, log=log)
    info.pop("history")
    out = m.then(0, "hardcore", steps=steps, seed=seed, **({"forced": True} if force else {}))
    out.pos = x.cpu().numpy().astype(np.float64)
    return out, info


def build_polymer(m: Model, *, steps: int = 3000, seed: int = 0, device: str = "cpu", force=False, log=None):
    require(m, 1, "polymer", force)
    pos, els, edges, info = polymer.build_polymer(_tensor(m, device), polymer.load_reference(),
                                                  envelope=_envelope(m, device), seed=seed, steps=steps, log=log)
    info.pop("history")
    out = m.then(2, "polymer", steps=steps, seed=seed, **({"forced": True} if force else {}))
    out.pos, out.els, out.edges = pos.cpu().numpy().astype(np.float64), els.cpu().numpy(), edges
    out.resseq = out.names = out.resnames = out.bfactor = None
    return out, info


def thread_chain(m: Model, *, split: int | None = 5, prune: int | None = 4, relax: int = 3000,
                 conserve: bool = True, seed: int = 0,
                 device: str = "cpu", force=False, log=None):
    require(m, 2, "thread", force)
    pos, els, edges, resseq, backbone, _, stats = thread.thread(m.pos, m.els, (
        thread.decompose(m.n, m.edges, split) if split else m.edges), prune=prune, fill=True, conserve=conserve, seed=seed)
    info = dict(stats)
    if relax:
        x, rinfo = polymer.relax(torch.as_tensor(pos, dtype=torch.float32, device=device), els, edges,
                                 polymer.load_reference(), envelope=_envelope(m, device), steps=relax, seed=seed,
                                 log=log)
        rinfo.pop("history")
        pos, info = x.cpu().numpy().astype(np.float64), {**info, **rinfo}
    # backbone atoms are named BB<element>; side atoms <element><n> (unique within a residue)
    names, count = [], {}
    for i in range(len(pos)):
        el = atoms.ELEMENTS[els[i]]
        if backbone[i]:
            names.append(f"BB{el}")
        else:
            count[resseq[i]] = count.get(resseq[i], 0) + 1
            names.append(f"{el}{count[resseq[i]]}"[:4])
    out = m.then(3, "thread", split=split, prune=prune, relax=relax, conserve=conserve, seed=seed, **({"forced": True} if force else {}))
    out.pos, out.els, out.edges, out.resseq, out.names = pos, els, edges, resseq, names
    out.resnames = out.bfactor = None
    return out, info


def backbone_order(m: Model) -> np.ndarray:
    """Backbone atom indices of a rung-3 chain, in chain order."""
    bb = np.flatnonzero([nm.startswith("BB") for nm in (m.names or [])])
    if len(bb) == 0:
        raise ValueError("no backbone atoms (named BB*) in this model: was it made by `decoygen thread`?")
    return bb[np.argsort(m.resseq[bb], kind="stable")]


def make_peptide(m: Model, *, steps: int = 3000, keep_atoms: bool = True, seed: int = 0, device: str = "cpu",
                 force=False, log=None):
    require(m, 3, "peptide", force)
    pep = peptide.build(m.pos, m.els, m.edges, backbone_order(m), keep_atoms=keep_atoms, seed=seed)
    info = {**pep["stats"], "before_relax": peptide.report(pep, pep["pos"])}
    r, extra = peptide.restraints(pep, polymer.load_reference(), device, seed=seed)
    x, rinfo = polymer.relax(torch.as_tensor(pep["pos"], dtype=torch.float32, device=device), pep["els"],
                             pep["edges"], polymer.load_reference(), envelope=_envelope(m, device), steps=steps,
                             seed=seed, log=log, restraints=r, extra=extra)
    rinfo.pop("history")
    pos = x.cpu().numpy().astype(np.float64)
    info.update(rinfo)
    info.update(peptide.report(pep, pos))
    out = m.then(4, "peptide", steps=steps, keep_atoms=keep_atoms, seed=seed, **({"forced": True} if force else {}))
    out.pos, out.els, out.edges = pos, pep["els"], pep["edges"]
    out.resseq, out.names, out.resnames, out.bfactor = pep["resseq"], pep["names"], pep["resnames"], None
    return out, info


def make_sidechains(m: Model, *, fraction: float = 0.2, steps: int = 3000, stages: int = 5, seed: int = 0,
                    device: str = "cpu", force=False, log=None):
    """Thin the backbone to a protein's residues per atom in passes, then real side chains."""
    require(m, 4, "sidechains", force)
    ref, env = polymer.load_reference(), _envelope(m, device)
    ch = sidechains.Chain(m.pos, m.els, m.edges, m.resseq, m.names)
    n_res_in = ch.n_res
    n_target = round(m.n / (len(sidechains.BACKBONE) + sidechains.MEAN_SIZE))
    passes = sidechains.thin(ch, n_target, fraction=fraction, steps=steps, stages=stages, envelope=env, ref=ref,
                             device=device, seed=seed, log=log)
    rng = np.random.default_rng(seed)
    sizes = np.array([len(g) for g in ch.group])
    seq = sidechains.fit_sequence(sizes, m.n - len(sidechains.BACKBONE) * ch.n_res, rng)
    pep = sidechains.place_sidechains(ch, sidechains.load_library(), seq)
    r, extra = sidechains.final_restraints(pep, ref, device, seed=seed)
    x, rinfo = polymer.relax(torch.as_tensor(pep["pos"], dtype=torch.float32, device=device), pep["els"],
                             pep["edges"], ref, envelope=env, steps=steps, seed=seed, restraints=r, extra=extra,
                             log=log)
    rinfo.pop("history")
    pos = x.cpu().numpy().astype(np.float64)
    composition = {aa: seq.count(aa) for aa in sorted(set(seq))}
    info = {"residues_in": n_res_in, "residues_out": len(seq), "atoms_in": m.n, "atoms_out": len(pos),
            "atoms_per_residue": len(pos) / len(seq), "thinning_passes": passes, "composition": composition,
            "sequence": "".join(_ONE.get(a, "X") for a in seq), **rinfo, **peptide.report(pep, pos)}
    out = m.then(5, "sidechains", fraction=fraction, steps=steps, stages=stages, seed=seed,
                 **({"forced": True} if force else {}))
    out.pos, out.els, out.edges = pos, pep["els"], pep["edges"]
    out.resseq, out.names, out.resnames, out.bfactor = pep["resseq"], pep["names"], pep["resnames"], None
    return out, info


_ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
        "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
        "TYR": "Y", "VAL": "V"}


def mirror(m: Model) -> Model:
    """Reflect through the frame's x = box/2 plane; mirroring twice restores the original handedness."""
    out = m.then(m.rung, "mirror") if m.rung is not None else m.then(None, "mirror")
    out.pos = m.pos.copy()
    out.pos[:, 2] = m.box - m.pos[:, 2]                                # [z, y, x]
    out.mirrored = not m.mirrored
    if m.symmetry:                                                     # reflect the operators too: M g M
        mm = np.diag([1.0, 1.0, -1.0])
        c = np.array(m.symmetry["centre"], dtype=np.float64).copy()
        c[2] = m.box - c[2]
        out.symmetry = {**m.symmetry, "centre": c, "ops": np.array([mm @ g @ mm for g in m.symmetry["ops"]])}
    return out


def render(m: Model, out: Path, *, apix: float = 1.0, resolution: float = 3.5, snr_scale: float = 1.0,
           bfactor: float = RENDER_BFACTOR, seed: int = 0, device: str = "cpu") -> dict:
    """truth (noise-free), half1, half2 (independent white noise, FSC = 0.143 at `resolution`),
    full (their mean), support; the model's rung and history go into the MRC labels and maps.json."""
    out.mkdir(parents=True, exist_ok=True)
    m = expand(m)                                                      # symmetric: the whole assembly
    n = round(m.box / apix)
    n += n % 2
    grid = Grid(n, apix, device)
    with torch.no_grad():
        signal = atoms.render(grid, _tensor(m, device), torch.as_tensor(m.els, device=device), bfactor)
        h1, h2, ninfo = noise.half_maps(grid, signal, resolution, snr_scale, "white", seed)
        support = grid.gaussian_filter(signal, SUPPORT_SIGMA)
        support = support > SUPPORT_LEVEL * support.max()
    label = (f"decoygen rung {m.rung}{' mirrored' if m.mirrored else ''}" if m.rung is not None
             else "decoygen render of a non-decoygen model")
    for name, x in (("half1", h1), ("half2", h2), ("full", (h1 + h2) / 2), ("truth", signal),
                    ("support", support.float())):
        io.write_mrc(out / f"{name}.mrc", x.cpu().numpy(), apix, labels=[label, *m.history[-8:]])
    meta = {"rung": m.rung, "mirrored": m.mirrored, "history": m.history, "atoms": m.n, "apix": apix,
            "box_voxels": n, "resolution": resolution, "snr_scale": snr_scale, "bfactor": bfactor, "seed": seed,
            "noise": {k: v for k, v in ninfo.items() if isinstance(v, (int, float, str))}}
    (out / "maps.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta
