"""Pilot: optimise decoy pseudo-atom coordinates against ModelAngelo's C-alpha network.

Direct coordinate optimisation ("hallucination"). The atoms of one default decoy are
moved by Adam to maximise the summed C-alpha probability that ModelAngelo's first-stage
CNN assigns to the rendered, noised map. Forward model, per step:

    atoms -> decoygen.atoms.render (B = 40 A^2, 1.5 A/px, 96^3) -> + white noise
    (fresh each step, level fixed from the initial map as in decoygen.noise)
    -> ModelAngelo preprocessing (z-score, local-std channel)
    -> 27 sliding 64^3 windows, stride 16, crop 6, sigmoid, averaged, as in
       model_angelo/c_alpha/inference.py

Hard constraints after every step: atoms stay inside the decoy envelope (projection to the
nearest inside voxel) and no pair is closer than --dmin (position-based clash relaxation).

Modes:
  controls   real protein (6CVM chain A, 108 kDa) and its mirror image, same forward model
  relax      the decoy under the constraints only (no network): separates de-clashing
             from the effect of the objective
  optimise   constraints + ModelAngelo objective
  atoms      evaluate saved atom models (--pdb NAME=path ...) in the same 144 A frame,
             e.g. experiments/polymer/compare.py's NAME_atoms.pdb
  polymer    optimise the raw polymer (--polymer, PDB with CONECT) against the detector with
             hard covalent geometry; --rewire-every R rebuilds the bond tree every R steps
  soft       the same with soft bonds (pairs classed by distance, valence counts), hardened
             into a fixed bond list for the last steps (--harden-at)

Each mode writes, per state: NAME_full.mrc (noisy map, fixed noise seed; this is what
model_angelo build_no_seq is run on), NAME_clean.mrc, NAME_ca_prob.mrc, NAME_atoms.cif
and NAME_metrics.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from itertools import product
from pathlib import Path

import gemmi
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from decoygen import atoms, io, noise, roughness
from decoygen.constants import MASS_PER_HEAVY_ATOM
from decoygen.diagnostics import chirality
from decoygen.envelope import make_envelope
from decoygen.fourier import Grid
from decoygen.params import DecoyParams
from decoygen.turing import turing_field

USER = Path("/proj/liu-folab-cryoem-dev/users/x_bjfor")
MA_ROOT = USER / "tools/model-angelo"
CKPT_DIR = USER / "torch_home/hub/checkpoints/model_angelo_v1.0/nucleotides_no_seq/c_alpha"
PROTEIN = Path(__file__).parent / "data/6CVM.cif"

APIX = 1.5                  # ModelAngelo resamples every map to 1.5 A/px
BOX = 96                    # 144 A
WIN, STRIDE, CROP = 64, 16, 6
RESOLUTION = 3.5
BFACTOR = 40.0
EVAL_NOISE_SEED = 12345
CHANNELS = ("ca", "p", "backbone")


# ---------------------------------------------------------------- ModelAngelo C-alpha CNN

def local_std(x: torch.Tensor, k: int = 10) -> torch.Tensor:
    """model_angelo.utils.grid.get_local_std without einops (separable, zero-padded)."""
    ker = torch.exp(-torch.linspace(-1.5, 1.5, 2 * k + 1, device=x.device).square())
    ker = ker / ker.sum()

    def blur(v):
        for dim in range(3):
            shape = [1, 1, 1, 1, 1]
            shape[2 + dim] = -1
            v = F.conv3d(v, ker.view(shape), padding="same")
        return v

    mean, sq = blur(x), blur(x * x)
    return (sq - mean**2).clamp(min=1e-12).sqrt()


def preprocess(m: torch.Tensor) -> torch.Tensor:
    x = (m - m.mean()) / (m.std() + 1e-6)
    s = local_std(x[None, None])[0, 0]
    return torch.stack([x, s / (s.max() + 1e-6)])


class CalphaNet:
    def __init__(self, device):
        sys.path.insert(0, str(MA_ROOT))
        spec = importlib.util.spec_from_file_location("ma_c_alpha_model", CKPT_DIR / "model.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.Model().eval()
        ck = torch.load(CKPT_DIR / "chkpt.torch", map_location="cpu", weights_only=False)
        model.load_state_dict(ck.get("model", ck))
        self.model = model.to(device).requires_grad_(False)
        starts = [i * STRIDE for i in range((BOX - WIN) // STRIDE)] + [BOX - 1 - WIN]
        self.corners = list(product(starts, repeat=3))
        count = torch.zeros((BOX,) * 3, device=device)
        for c in self.corners:
            count[self._inner(c)] += 1
        self.inv_count = 1 / (count + 1e-6)

    @staticmethod
    def _window(c):
        return np.s_[:, c[0]:c[0] + WIN, c[1]:c[1] + WIN, c[2]:c[2] + WIN]

    @staticmethod
    def _inner(c):
        return np.s_[c[0] + CROP:c[0] + WIN - CROP, c[1] + CROP:c[1] + WIN - CROP,
                     c[2] + CROP:c[2] + WIN - CROP]

    def _logits(self, inp, c):
        return self.model(inp[self._window(c)][None])[0, :, CROP:WIN - CROP, CROP:WIN - CROP, CROP:WIN - CROP]

    @torch.no_grad()
    def probabilities(self, m: torch.Tensor) -> torch.Tensor:
        """Averaged sigmoid outputs (3, BOX, BOX, BOX), as ModelAngelo assembles them."""
        inp = preprocess(m)
        out = torch.zeros((3,) + (BOX,) * 3, device=m.device)
        for c in self.corners:
            out[(slice(None),) + self._inner(c)] += torch.sigmoid(self._logits(inp, c))
        return out * self.inv_count

    def backward_score(self, m: torch.Tensor, n_windows: int, gen: torch.Generator, power: float = 1.0) -> float:
        """Accumulate d(-S)/dm into m's graph; S = sum_v p(v)^power of the C-alpha probability.

        For power 1, S separates over windows (sum_v p_w(v) / count(v)), so windows are
        back-propagated one at a time into a detached copy of the preprocessed input; a
        random subset of n_windows, rescaled, gives an unbiased stochastic estimate. For
        power > 1 the window average of p^power is used instead of (average p)^power (an
        upper bound by Jensen), which keeps the separation.
        """
        inp = preprocess(m)
        leaf = inp.detach().requires_grad_(True)
        idx = torch.randperm(len(self.corners), generator=gen)[:n_windows].tolist()
        scale = len(self.corners) / len(idx)
        total = 0.0
        for i in idx:
            c = self.corners[i]
            s = (torch.sigmoid(self._logits(leaf, c)[0]) ** power * self.inv_count[self._inner(c)]).sum() * scale
            (-s).backward()
            total += float(s)
        inp.backward(leaf.grad)
        return total


# ---------------------------------------------------------------- atoms and constraints

def decoy_atoms(params: DecoyParams, device):
    """The default pipeline's envelope and atom gas (pipeline.generate up to sample_gas)."""
    g = Grid(params.box, params.apix, device)
    rough = roughness.roughness_parameters(params.roughness_D, params.alpha, params.roughness_amplitude)
    mask, _ = make_envelope(
        g, volume=params.volume, alpha=rough["alpha"], roughness_amplitude=rough["roughness_amplitude"],
        compactness=params.compactness, n_domains=params.n_domains, elongation=params.elongation,
        bias=params.envelope_bias, clearance=params.edge_clearance, seed=params.seed)
    turing, _ = turing_field(
        g, mask, wavelength=params.meso_wavelength, quotient=params.turing_quotient,
        fill_fraction=params.fill_fraction, iterations=params.turing_iterations, dt=params.turing_dt,
        seed=params.seed)
    m = torch.as_tensor(mask, device=device)
    intensity = ((1 - params.meso_strength) + params.meso_strength * turing) * m
    n_atoms = round(params.mass_kda * 1e3 / MASS_PER_HEAVY_ATOM)
    pos, el = atoms.sample_gas(g, intensity, n_atoms, params.seed, "atoms")
    return pos, el, mask


def protein_atoms(path: Path, mirror: bool, device):
    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    chain = st[0]["A"]
    xyz = np.array([a.pos.tolist() for r in chain for a in r])
    if mirror:
        xyz[:, 0] *= -1
    xyz -= (xyz.max(0) + xyz.min(0)) / 2
    xyz += BOX * APIX / 2
    lookup = {e: i for i, e in enumerate(atoms.ELEMENTS)}
    el = [lookup[a.element.name] for r in chain for a in r]
    zyx = torch.tensor(xyz[:, ::-1].copy(), dtype=torch.float32, device=device)
    return zyx, torch.tensor(el, device=device)


class Envelope:
    """Projection onto the decoy envelope (1 A grid, offset into the 1.5 A box)."""

    def __init__(self, mask: np.ndarray, apix: float, offset: float, device):
        self.apix, self.offset = apix, offset
        self.inside = torch.as_tensor(mask, device=device)
        idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
        self.nearest = torch.as_tensor(np.stack(idx, -1), device=device)          # (n, n, n, 3)

    @torch.no_grad()
    def project(self, pos: torch.Tensor) -> int:
        v = torch.round((pos - self.offset) / self.apix).long().clamp(0, self.inside.shape[0] - 1)
        out = ~self.inside[v[:, 0], v[:, 1], v[:, 2]]
        if out.any():
            near = self.nearest[v[out, 0], v[out, 1], v[out, 2]].float()
            pos[out] = near * self.apix + self.offset
        return int(out.sum())


@torch.no_grad()
def relax_clashes(pos: torch.Tensor, dmin: float, iters: int = 10) -> int:
    """Position-based dynamics: push pairs closer than dmin apart symmetrically."""
    for _ in range(iters):
        d = torch.cdist(pos, pos)
        d.fill_diagonal_(math.inf)
        i, j = torch.nonzero(torch.triu(d < dmin, diagonal=1), as_tuple=True)
        if len(i) == 0:
            return 0
        v = pos[i] - pos[j]
        dist = v.norm(dim=1, keepdim=True).clamp(min=1e-3)
        corr = 0.5 * (dmin - dist) * v / dist
        pos += torch.zeros_like(pos).index_add_(0, i, corr).index_add_(0, j, -corr)
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(math.inf)
    return int(torch.triu(d < dmin - 0.05, diagonal=1).sum())


# ---------------------------------------------------------------- analysis and output

def pair_histogram(pos: torch.Tensor, rmax: float = 8.0, dr: float = 0.1) -> dict:
    d = torch.cdist(pos, pos)
    d = d[torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)]
    d = d[d < rmax]
    edges = np.arange(0, rmax + dr / 2, dr)
    h = np.histogram(d.cpu().numpy(), bins=edges)[0]
    r = (edges[1:] + edges[:-1]) / 2
    # neighbours per atom per A of shell, divided by the shell area: ~ g(r) * number density
    per_atom = 2 * h / len(pos) / (4 * np.pi * r**2 * dr)
    return {"r": r.tolist(), "density": per_atom.tolist()}


def atom_stats(pos: torch.Tensor) -> dict:
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(math.inf)
    nn = d.min(1).values
    return {"n_atoms": len(pos),
            "nn_distance_mean_A": float(nn.mean()), "nn_distance_p05_A": float(nn.quantile(0.05)),
            "neighbours_within_1.8A_mean": float((d < 1.8).sum(1).float().mean()),
            "neighbours_within_1.8A_max": int((d < 1.8).sum(1).max()),
            "neighbours_within_4A_mean": float((d < 4.0).sum(1).float().mean())}


def write_atoms(path: Path, pos: torch.Tensor, el: torch.Tensor) -> None:
    st = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    xyz = pos.detach().cpu().numpy()[:, ::-1]
    for k, (p, e) in enumerate(zip(xyz, el.tolist())):
        res = gemmi.Residue()
        res.name, res.het_flag = "UNK", "H"
        res.seqid = gemmi.SeqId(k + 1, " ")
        a = gemmi.Atom()
        a.name = atoms.ELEMENTS[e]
        a.element = gemmi.Element(atoms.ELEMENTS[e])
        a.pos = gemmi.Position(*map(float, p))
        a.b_iso = BFACTOR
        res.add_atom(a)
        chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    st.make_mmcif_document().write_file(str(path))


class ForwardModel:
    def __init__(self, device):
        self.grid = Grid(BOX, APIX, device)

    def signal(self, pos, el):
        return atoms.render(self.grid, pos, el, BFACTOR)

    def noise_sd(self, signal) -> float:
        """Real-space sd of the full map's noise: the mean of two decoygen half-maps."""
        n0 = noise.white_noise_level(self.grid, signal.detach(), RESOLUTION, 1.0)
        return math.sqrt(n0 / self.grid.n**3) / math.sqrt(2)

    def noise(self, sd, seed, device):
        gen = torch.Generator().manual_seed(seed)
        return (sd * torch.randn(self.grid.shape, generator=gen)).to(device)


def evaluate_state(name, pos, el, fm: ForwardModel, net: CalphaNet, sd: float, out: Path, extra=None) -> dict:
    g = fm.grid
    with torch.no_grad():
        signal = fm.signal(pos, el)
        full = signal + fm.noise(sd, EVAL_NOISE_SEED, signal.device)
        clean = g.cosine_lowpass(signal, 1 / RESOLUTION, 0.1 / RESOLUTION)
        probs = net.probabilities(full)
        support = g.gaussian_filter(signal, 2.0)
        support = support > 0.2 * support.max()
    io.write_mrc(out / f"{name}_full.mrc", full.cpu().numpy(), APIX)
    io.write_mrc(out / f"{name}_clean.mrc", clean.cpu().numpy(), APIX)
    io.write_mrc(out / f"{name}_ca_prob.mrc", probs[0].cpu().numpy(), APIX)
    write_atoms(out / f"{name}_atoms.cif", pos, el)
    metrics = {
        "name": name,
        **{f"{c}_prob_sum": float(probs[i].sum()) for i, c in enumerate(CHANNELS)},
        "ca_voxels_above_0.05": int((probs[0] > 0.05).sum()),
        "ca_prob_max": float(probs[0].max()),
        "chirality_clean": chirality(g, clean, support),
        "signal_max_over_mean_in_support": float(signal.max() / signal[support].mean()),
        "noise_sd": sd,
        **atom_stats(pos),
        "pair_histogram": pair_histogram(pos),
        **(extra or {}),
    }
    io.write_json(out / f"{name}_metrics.json", metrics)
    print(f"{name}: " + ", ".join(f"{k}={v:.4g}" for k, v in metrics.items()
                                  if isinstance(v, float)), flush=True)
    return metrics


# ---------------------------------------------------------------- modes

def load_atoms(path: Path, device):
    """Heavy atoms of a PDB/mmCIF file as ([z, y, x] positions in A, element indices)."""
    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    lookup = {e: i for i, e in enumerate(atoms.ELEMENTS)}
    found = [(a.pos.tolist(), lookup[a.element.name]) for ch in st[0] for r in ch for a in r
             if a.element.name in lookup]
    xyz = np.array([f[0] for f in found])
    return (torch.tensor(xyz[:, ::-1].copy(), dtype=torch.float32, device=device),
            torch.tensor([f[1] for f in found], device=device))


def run_atoms(args, fm, net, device):
    for spec in args.pdb:
        name, path = spec.split("=", 1)
        pos, el = load_atoms(Path(path), device)
        sd = fm.noise_sd(fm.signal(pos, el))
        evaluate_state(name, pos, el, fm, net, sd, args.out, extra={"source": path})


def run_controls(args, fm, net, device):
    for name, mirror in (("protein", False), ("protein_mirror", True)):
        pos, el = protein_atoms(PROTEIN, mirror, device)
        sd = fm.noise_sd(fm.signal(pos, el))
        evaluate_state(name, pos, el, fm, net, sd, args.out)


def run_optimise(args, fm, net, device, use_net: bool):
    params = DecoyParams(seed=args.seed)
    pos1, el, mask = decoy_atoms(params, device)
    offset = (BOX * APIX - params.box * params.apix) / 2
    pos = (pos1 + offset).clone()
    env = Envelope(mask, params.apix, offset, device)
    sd = fm.noise_sd(fm.signal(pos, el))
    tag = f"dmin{args.dmin:g}" + (f"_p{args.power:g}" if args.power != 1 else "")
    if not (args.out / "decoy_initial_metrics.json").exists():
        evaluate_state("decoy_initial", pos, el, fm, net, sd, args.out)

    pos.requires_grad_(True)
    opt = torch.optim.Adam([pos], lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    history = []
    steps = args.steps if use_net else args.relax_steps
    t0 = time.perf_counter()
    for step in range(steps + 1):
        with torch.no_grad():
            n_out = env.project(pos.data)
            n_clash = relax_clashes(pos.data, args.dmin)
        if step == steps:
            break
        opt.zero_grad()
        if use_net:
            m = fm.signal(pos, el) + fm.noise(sd, 1000 + step, device)
            score = net.backward_score(m, args.windows, gen, args.power)
            opt.step()
        else:
            score = float("nan")
        history.append({"step": step, "score": score, "clashes_after_relax": n_clash,
                        "projected_outside": n_out, "time_s": time.perf_counter() - t0})
        if step % 10 == 0:
            print(f"step {step}: S={score:.2f} clashes={n_clash} projected={n_out} "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    name = f"decoy_{'hallucinated' if use_net else 'relaxed'}_{tag}"
    pos = pos.detach()
    evaluate_state(name, pos, el, fm, net, sd, args.out,
                   extra={"dmin": args.dmin, "lr": args.lr, "power": args.power, "steps": steps,
                          "windows_per_step": args.windows,
                          "final_clashes": n_clash, "displacement_rms_A":
                              float(((pos - (pos1 + offset)) ** 2).sum(1).mean().sqrt())})
    io.write_json(args.out / f"{name}_history.json", {"history": history})


def run_polymer(args, fm, net, device):
    """Raw polymer vs the C-alpha CNN, with hard covalent geometry.

    Each step: one Adam step on the detector objective, then --geo-steps Adam steps on the
    polymer restraints alone (bonds, angles, torsion floor, sterics, envelope), so geometry
    stays at reference values whatever the detector wants. With --rewire-every R > 0 the bond
    tree is rebuilt from the current positions at step 0 and every R steps after (spanning
    forest with per-element bond caps, elements fixed, fixed seed so changes come from atom
    movement only); surviving bonds keep their targets and new ones are relaxed for
    --rewire-relax steps.
    """
    from decoygen import polymer

    ref = polymer.load_reference()
    pos_np, el_np, edges = io.read_pdb(args.polymer, atoms.ELEMENTS)
    pos = torch.tensor(pos_np, dtype=torch.float32, device=device)
    el = torch.tensor(el_np, device=device)
    n = len(pos)
    params = DecoyParams(seed=args.seed)
    _, _, mask = decoy_atoms(params, device)
    offset = (BOX * APIX - params.box * params.apix) / 2
    env = polymer.Envelope(mask, params.apix, device)
    caps = polymer.element_caps(el_np)
    restr = polymer.Restraints(el_np, edges, ref, device, seed=args.seed)
    sd = fm.noise_sd(fm.signal(pos, el))
    tag = (f"rewire{args.rewire_every}" if args.rewire_every else "fixed") + (
        f"_p{args.power:g}" if args.power != 1 else "")
    name = f"decoy_polymer_{tag}"
    pos0 = pos.clone()
    pos.requires_grad_(True)
    opt = torch.optim.Adam([pos], lr=args.lr)
    geo_opt = torch.optim.Adam([pos], lr=args.geo_lr)
    gen = torch.Generator().manual_seed(args.seed)

    def geometry(k):
        restr.update_neighbours(pos.detach())
        for _ in range(k):
            e = restr.terms(pos)
            loss = (sum(e.values()) + (env.outside(pos - offset) ** 2).sum()) / n
            geo_opt.zero_grad()
            loss.backward()
            geo_opt.step()

    def rewire(step):
        nonlocal restr
        new = polymer.spanning_forest(pos.detach().cpu().numpy().astype(np.float64),
                                      branch_weight=args.branch_weight, max_degree=caps, seed=args.seed)
        old_set = {tuple(e) for e in np.sort(restr.edges, axis=1)}
        new_set = {tuple(e) for e in np.sort(new, axis=1)}
        restr = polymer.Restraints(el_np, new, ref, device, seed=args.seed + step, previous=restr)
        geometry(args.rewire_relax)
        return len(new_set - old_set)

    history = []
    t0 = time.perf_counter()
    for step in range(args.steps):
        changed = rewire(step) if args.rewire_every and step % args.rewire_every == 0 else 0
        opt.zero_grad()
        m = fm.signal(pos, el) + fm.noise(sd, 1000 + step, device)
        score = net.backward_score(m, args.windows, gen, args.power)
        opt.step()
        geometry(args.geo_steps)
        rec = {"step": step, "score": score, "bonds_changed": changed, "time_s": time.perf_counter() - t0}
        if step % 10 == 0 or step == args.steps - 1:
            rec.update(restr.report(pos.detach()))
            print(f"step {step}: S={score:.1f} changed={changed} bond_rmsd={rec['bond_rmsd_A']:.3f} "
                  f"angle_rmsd={rec['angle_rmsd_A']:.3f} steric>0.2={rec['steric_violations_gt_0.2A']} "
                  f"{rec['time_s']:.0f}s", flush=True)
        history.append(rec)
    pos = pos.detach()
    final = restr.report(pos)
    evaluate_state(name, pos, el, fm, net, sd, args.out,
                   extra={"power": args.power, "lr": args.lr, "steps": args.steps, "geo_steps": args.geo_steps,
                          "rewire_every": args.rewire_every, "geometry": final,
                          "bonds_changed_total": int(sum(h["bonds_changed"] for h in history)),
                          "displacement_rms_A": float(((pos - pos0) ** 2).sum(1).mean().sqrt())})
    io.write_pdb(args.out / f"{name}_atoms.pdb", pos.cpu().numpy(), el_np.tolist(), atoms.ELEMENTS,
                 restr.edges, BFACTOR)
    io.write_json(args.out / f"{name}_history.json", {"history": history})


def run_soft(args, fm, net, device):
    """Raw polymer vs the C-alpha CNN with soft bonds (polymer.SoftChemistry), then hardened.

    Steps [0, harden): the bond list is not fixed. Pairs are classed bonded / 1-3 / free by
    distance with sigmoid boundaries whose width anneals geometrically from --soft-width to
    --soft-width-end, and per-atom valence counts (target = the atom's degree in the input
    polymer) replace the bond graph. Each step: one detector Adam step, --geo-steps steps on
    the soft energy plus envelope. At step harden = --harden-at * steps the state is evaluated
    as NAME_soft, a hard bond list is read off (bond cutoff, element caps) and relaxed with the
    usual Restraints; the remaining steps run with that fixed topology and are evaluated as NAME.

    With --trajectory-every K, every K-th step is written to NAME_traj/frame_SSSS.pdb with its
    current bonds as CONECT and a per-atom B-factor marking changes against the input polymer
    (2: in a bond the input did not have, 1: lost a bond and gained none, 0: unchanged), plus
    NAME_traj/movie.cxc to play them in ChimeraX.
    """
    from decoygen import polymer

    ref = polymer.load_reference()
    pos_np, el_np, edges0 = io.read_pdb(args.polymer, atoms.ELEMENTS)
    pos = torch.tensor(pos_np, dtype=torch.float32, device=device)
    el = torch.tensor(el_np, device=device)
    n = len(pos)
    degree = np.minimum(np.bincount(edges0.ravel(), minlength=n), polymer.element_caps(el_np))
    params = DecoyParams(seed=args.seed)
    _, _, mask = decoy_atoms(params, device)
    offset = (BOX * APIX - params.box * params.apix) / 2
    env = polymer.Envelope(mask, params.apix, device)
    soft = polymer.SoftChemistry(el_np, degree, ref, device, count_weight=args.count_weight)
    restr = None
    sd = fm.noise_sd(fm.signal(pos, el))
    harden = round(args.harden_at * args.steps)
    name = f"decoy_polymer_soft_cw{args.count_weight:g}_lr{args.lr:g}_p{args.power:g}"
    pos0 = pos.clone()
    pos.requires_grad_(True)
    opt = torch.optim.Adam([pos], lr=args.lr)
    geo_opt = torch.optim.Adam([pos], lr=args.geo_lr)
    gen = torch.Generator().manual_seed(args.seed)

    def width(step):
        f = min(step / max(harden - 1, 1), 1.0)
        return args.soft_width * (args.soft_width_end / args.soft_width) ** f

    def geometry(k, w=None):
        if restr is None:
            soft.update_neighbours(pos.detach())
        else:
            restr.update_neighbours(pos.detach())
        for _ in range(k):
            e = soft.terms(pos, w) if restr is None else restr.terms(pos)
            loss = (sum(e.values()) + (env.outside(pos - offset) ** 2).sum()) / n
            geo_opt.zero_grad()
            loss.backward()
            geo_opt.step()

    def extra(phase, geo, topo):
        return {"phase": phase, "power": args.power, "lr": args.lr, "steps": args.steps, "harden_step": harden,
                "geo_steps": args.geo_steps, "soft_width": [args.soft_width, args.soft_width_end],
                "count_weight": args.count_weight, "geometry": geo, "topology": topo,
                "displacement_rms_A": float(((pos.detach() - pos0) ** 2).sum(1).mean().sqrt())}

    traj = args.out / f"{name}_traj"
    old_bonds = {tuple(e) for e in np.sort(edges0, axis=1)}

    def frame(step):
        x = pos.detach()
        edges = soft.bonds(x) if restr is None else restr.edges
        cur = {tuple(e) for e in np.sort(edges, axis=1)}
        b = np.zeros(n)
        b[np.array(list(old_bonds - cur), dtype=np.int64).ravel()] = 1
        b[np.array(list(cur - old_bonds), dtype=np.int64).ravel()] = 2
        io.write_pdb(traj / f"frame_{step:04d}.pdb", x.cpu().numpy(), el_np.tolist(), atoms.ELEMENTS, edges, b)

    if args.trajectory_every:
        traj.mkdir(parents=True, exist_ok=True)
    history = []
    t0 = time.perf_counter()
    for step in range(args.steps):
        if step == harden:
            x = pos.detach()
            edges = soft.bonds(x)
            topo = polymer.topology_stats(n, edges, edges0)
            r = polymer.Restraints(el_np, edges, ref, device, seed=args.seed)
            evaluate_state(f"{name}_soft", x, el, fm, net, sd, args.out, extra=extra("soft", r.report(x), topo))
            io.write_pdb(args.out / f"{name}_soft_atoms.pdb", x.cpu().numpy(), el_np.tolist(), atoms.ELEMENTS,
                         edges, BFACTOR)
            restr = r
            geometry(args.harden_relax)
            print(f"hardened at step {step}: {topo} {restr.report(pos.detach())}", flush=True)
        opt.zero_grad()
        m = fm.signal(pos, el) + fm.noise(sd, 1000 + step, device)
        score = net.backward_score(m, args.windows, gen, args.power)
        opt.step()
        w = width(step)
        geometry(args.geo_steps, w)
        rec = {"step": step, "score": score, "width_A": w if restr is None else 0.0,
               "time_s": time.perf_counter() - t0}
        if step % 10 == 0 or step == args.steps - 1:
            x = pos.detach()
            if restr is None:
                with torch.no_grad():
                    rec["energy"] = {k: float(v) / n for k, v in soft.terms(x, w).items()}
                rec["topology"] = polymer.topology_stats(n, soft.bonds(x), edges0)
                t = rec["topology"]
                print(f"step {step}: S={score:.1f} width={w:.3f} new={t['bonds_new']} lost={t['bonds_lost']} "
                      f"bonds={t['n_bonds']} cycles={t['cycles']} "
                      f"deg={[round(f, 3) for f in t['degree_fractions'][1:4]]} "
                      + " ".join(f"{k}={v:.3g}" for k, v in rec["energy"].items())
                      + f" {rec['time_s']:.0f}s", flush=True)
            else:
                rec.update(restr.report(x))
                print(f"step {step}: S={score:.1f} hard bond_rmsd={rec['bond_rmsd_A']:.3f} "
                      f"angle_rmsd={rec['angle_rmsd_A']:.3f} steric>0.2={rec['steric_violations_gt_0.2A']} "
                      f"{rec['time_s']:.0f}s", flush=True)
        history.append(rec)
        if args.trajectory_every and (step % args.trajectory_every == 0 or step == args.steps - 1):
            frame(step)
    if args.trajectory_every:
        write_movie_script(traj)
    pos = pos.detach()
    if restr is not None:
        evaluate_state(name, pos, el, fm, net, sd, args.out,
                       extra=extra("hard", restr.report(pos), polymer.topology_stats(n, restr.edges, edges0)))
        io.write_pdb(args.out / f"{name}_atoms.pdb", pos.cpu().numpy(), el_np.tolist(), atoms.ELEMENTS,
                     restr.edges, BFACTOR)
    io.write_json(args.out / f"{name}_history.json", {"history": history})


def write_movie_script(traj: Path) -> None:
    """ChimeraX script that plays the frames of a soft-bond trajectory as a movie."""
    frames = sorted(p.name for p in traj.glob("frame_*.pdb"))
    k = len(frames)
    (traj / "movie.cxc").write_text(f"""# Soft-bond trajectory, {k} frames (one model per frame). In ChimeraX, from this folder:
#   open movie.cxc
# then drag or play the model-series slider. Atoms are coloured by B-factor: grey unchanged,
# orange lost a bond, blue in a new bond. To record instead, uncomment the last three lines.
close session
set bgColor white
graphics silhouettes true
lighting soft
open frame_*.pdb
style stick
color bfactor #1-{k} palette 0,#c8c6c0:1,#eb6834:2,#2a78d6
view #1
clip near -8 far 8 position #1
mseries slider #1-{k}
# movie record supersample 2
# mseries #1-{k} pause 2 loop false
# wait {2 * k}
# movie encode movie.mp4 framerate 25
""")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["controls", "relax", "optimise", "atoms", "polymer", "soft"])
    ap.add_argument("--polymer", type=Path, default=Path("runs/bookmarks/raw-polymer/polymer_atoms.pdb"))
    ap.add_argument("--rewire-every", type=int, default=0, help="rebuild the bond tree every R steps (0: fixed)")
    ap.add_argument("--rewire-relax", type=int, default=50, help="geometry steps after each re-wiring")
    ap.add_argument("--geo-steps", type=int, default=20, help="geometry-only steps after each detector step")
    ap.add_argument("--geo-lr", type=float, default=0.01)
    ap.add_argument("--branch-weight", type=float, default=0.91, help="as fitted when the polymer was built")
    ap.add_argument("--soft-width", type=float, default=0.15, help="soft-bond class boundary width at the start (A)")
    ap.add_argument("--soft-width-end", type=float, default=0.03, help="... annealed geometrically to this")
    ap.add_argument("--count-weight", type=float, default=1.0, help="weight of the soft valence counts")
    ap.add_argument("--harden-at", type=float, default=0.8, help="fraction of steps before bonds are made hard")
    ap.add_argument("--harden-relax", type=int, default=300, help="geometry steps right after hardening")
    ap.add_argument("--trajectory-every", type=int, default=0, help="soft mode: write a movie frame every K steps")
    ap.add_argument("--pdb", nargs="+", default=[], metavar="NAME=PATH", help="atom models for mode 'atoms'")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dmin", type=float, default=2.2, help="minimum interatomic distance (A)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--relax-steps", type=int, default=50)
    ap.add_argument("--windows", type=int, default=27, help="sliding windows per step (27 = all)")
    ap.add_argument("--power", type=float, default=1.0,
                    help="objective sum p^power; >1 rewards confident C-alpha peaks over diffuse probability")
    ap.add_argument("--lr", type=float, default=0.05, help="Adam step size (A)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    fm = ForwardModel(args.device)
    net = CalphaNet(args.device)
    if args.mode == "controls":
        run_controls(args, fm, net, args.device)
    elif args.mode == "atoms":
        run_atoms(args, fm, net, args.device)
    elif args.mode == "polymer":
        run_polymer(args, fm, net, args.device)
    elif args.mode == "soft":
        run_soft(args, fm, net, args.device)
    else:
        run_optimise(args, fm, net, args.device, use_net=args.mode == "optimise")


if __name__ == "__main__":
    main()
