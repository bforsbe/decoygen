# Brief: Generator for non-protein decoy cryo-EM densities

**Audience:** Claude Code (or another engineer) starting from scratch.
**Status:** Design brief. The old notebook code in the appendix is for reference only; do not port it. Rewrite cleanly.

---

## 1. Purpose

Build a generator for **synthetic 3D density maps (decoys)** that match proteins as closely as possible in low-order statistics but are **not recognizably protein-like**. Downstream uses:

- Hard negatives for training and evaluating map-interpretation networks (ModelAngelo-like builders, DeepEMhancer-like post-processors).
- Probing hallucination: does a tool invent protein features in density that has protein-like statistics but no protein structure?

**Hard constraint: the generator must never be trained or fitted on real maps or real atomic models.** Allowed sources of information, in increasing order of data exposure:

| Tier | Source | Example | Allowed |
|---|---|---|---|
| 0 | Physics and textbook constants | Atom form factors, protein density 1.35 g/cm³, ~1.21 Å³/Da, bond lengths, hard-core radii | Yes, the default |
| 1 | Published literature scalars | Surface fractal dimension ~2.1–2.4 (Lewis & Rees 1985); SASA ≈ 6.3·M^0.73 Å² (Miller et al. 1987) | Yes |
| 2 | Frozen pretrained networks as critics | DeepEMhancer, ModelAngelo, EMReady used as black-box or white-box scorers | Yes, but never fine-tune them |
| 3 | 1D summary statistics derived from real data | BEST curve (Popov & Bourenkov 2003) average protein intensity profile | Optional, behind a flag, off by default |
| — | Real maps, real models, patches of either | — | **Never** |

---

## 2. Core idea

Separate the statistics into two groups:

- **Match (low-order, generic):** 1-point density histogram, radial power spectrum, solvent fraction, volume per molecular mass, envelope roughness and compactness.
- **Break (high-order, protein-specific):** a single unbranched connected chain; Cα spacing ~3.8 Å; α-helices (5.4 Å pitch, ~10 Å rods, right-handed); β-sheets (4.8 Å strand spacing, planar); side-chain bumps; chirality.

Two maps with identical power spectra differ only in their Fourier phases, so the target is **protein-like amplitudes with non-protein phases**. The radial power spectrum is averaged over orientation: a ~10 Å or ~4.7 Å peak can come from *isotropic* pair correlations (a "liquid" with preferred spacings) instead of rods and sheets. **Spectral peaks must therefore be produced by isotropic correlations, never by building rods, sheets or chains.**

---

## 3. User-facing input spec

The generator takes (with sensible defaults and ranges):

| Parameter | Meaning | Implementation hint |
|---|---|---|
| `box` (N) | Cubic box edge in voxels; even | — |
| `apix` | Voxel size in Å | Render at this sampling |
| `resolution` | Target resolution d (Å) | Final low-pass/MTF; also the inner cutoff for roughness |
| `mass_kda` | Molecular mass | Volume V ≈ 1.21·M Å³ (M in Da) sets the mask volume |
| `roughness_D` | Surface fractal dimension, ~2.0–2.5 (protein ~2.1–2.4) | Spectral exponent of the envelope field; see §4.1 |
| `compactness` | Global roundness (target sphericity, or mean-curvature-flow time) | MBO iterations; see §4.1 |
| `n_domains`, `elongation` | Optional global shape | Union of sub-envelopes; anisotropic scaling |
| `meso_wavelength` | Turing wavelength in Å (default near 10) | Sets the mesoscale density modulation |
| `meso_strength` | 0–1, how strongly the Turing field modulates atom density | Low = statistical density wave; high = visible worms (more protein-like, avoid) |
| `fill_fraction` | Turing level-set fraction | ~0.5 gives a bicontinuous network (preferred); low gives tubes (helix-like, avoid) |
| `target_spectrum` | `physics` (default) / `physics+peaks` / `best` (tier 3) | §4.3 |
| `snr`, `noise_model` | Half-map noise | §4.4 |
| `seed` | Integer | Full determinism |

"Fractal dimension" is meaningful only between the resolution d and the molecule radius R ≈ (3V/4π)^(1/3), often less than one decade, so treat it as a **roughness exponent** (Hurst H, D = 3 − H). Roughness (D) and compactness ("surface energy") are separate knobs: surface tension smooths the smallest scales first and controls global roundness, while D controls how much area appears at finer scales.

---

## 4. Pipeline

### 4.1 Envelope (mask)

1. Gaussian random field with power ∝ k^−α. For 3 < α < 5 the level-set surface dimension is approximately **D ≈ (9 − α)/2**, i.e. **α ≈ 9 − 2D**. Threshold at the quantile giving volume V.
2. Keep the largest connected component and centre it with periodic wrapping (see appendix A.3). Require edge clearance.
3. Compactness: apply a mean-curvature-flow analogue (MBO scheme: Gaussian blur then re-threshold at constant volume, repeated n times). More iterations give rounder shapes.
4. Optional multiple domains: union of sub-envelopes, as in the old "dual region blob".
5. **Calibrate α → D empirically.** The steps above shift the theoretical mapping. Build a lookup table by generating masks over an α grid and measuring D with a Richardson plot (log isosurface area vs. log blur scale, fitted between d and R).
6. Diagnostic: surface area at the given resolution vs. mass; compare the scaling with the tier-1 literature SASA relation, noting that the probe scale differs.

### 4.2 Mesoscale field (Turing)

- Reaction–diffusion in the mask: `x ← tanh(x + dt · F⁻¹[g(k) · F[x]])`, updates applied only inside the mask.
- Generalize the old difference-of-Gaussians (DoG) kernel to a **designed radial growth function g(k)**. For DoG with σs, σl the peak is at k*² = ln(σl²/σs²) / (2π²(σl² − σs²)), giving λ = 1/k*. The old "vermiculation" preset (σs = 1, σl = 4 px) gives λ ≈ 10.3 px, i.e. the protein ~10 Å helix-packing scale at 1 Å/px.
- Pattern formation is winner-take-all in k, so g(k) controls morphology, not the final spectrum. The spectrum is set in §4.3.
- Morphology control: fill fraction (spots → tubes → bicontinuous → lamellae), junction density (Y-junctions are a non-protein marker; real chains don't branch), and statistical achirality (Turing patterns are mirror-symmetric in distribution; verify).
- Output: a smooth intensity field λ(x) in [0, 1], mixed as `λ = (1 − meso_strength) + meso_strength · turing`.

### 4.3 Atomistic layer (pseudo-atoms)

- **Inhomogeneous hard-core point process** (3D Poisson-disk sampling) with intensity λ(x) inside the mask, at protein heavy-atom number density (~1 heavy atom per ~17–18 Å³; verify from tier-0 constants).
- **Two hard-core scales:** ~1.5 Å (bonded-like) and ~3.5 Å (non-bonded); a mix of C/N/O(/S) pseudo-atoms in protein-like proportions, but **no bonds, residues, chain or chirality**.
- **Fit to a target structure factor S(s) with reverse Monte Carlo (RMC)**: Metropolis moves of individual atoms with energy = χ² between the computed and target S(s), plus hard-core constraints and confinement to the mask. Alternative: iterative Boltzmann inversion of a pair potential. Compute S(s) with the Debye formula on a neighbour list, or by FFT of gridded points.
- Target spectrum options:
  - `physics`: whatever S(s) the hard-core process gives (liquid-like packing). Pure tier 0.
  - `physics+peaks`: add parametric isotropic bumps near ~10 Å and ~4.5–5 Å; their heights and positions are free parameters for the optimizer (§6). A g(r) peak at r maps to an S(s) maximum nearer s ≈ 1.2/r, not 1/r, so calibrate numerically.
  - `best`: match the BEST curve (tier 3, off by default).
- The difference between `physics` and BEST is roughly the secondary-structure signature, which is the quantity under control.

**Simpler alternative (implement as a fallback or baseline):** skip the atoms and impose the target radial amplitude on the Turing/mask field by Gerchberg–Saxton projection: keep phases, set the radial amplitude, re-apply mask, positivity and histogram constraints, and repeat a few iterations (generalizes appendix A.2). Faster, but not atomistic.

### 4.4 Rendering to maps

- Atoms → density with electron scattering factors (e.g. Peng 5-Gaussian parameterization; `gemmi` can do this), per-atom B-factors, then the resolution low-pass/MTF.
- **Two half-maps** with independent noise realizations, plus the **noise-free map**. Noise: coloured Gaussian noise whose spectrum gives a sensible half-map FSC for the target resolution; solvent region gets the same noise model (no hard mask edge in the final map).
- Use the same pipeline for any `apix`/`box`, so decoys can match the conventions of any downstream tool.

---

## 5. Compatibility with ModelAngelo / DeepEMhancer-style training and evaluation

**Verify each tool's exact expectations by reading its repository/docs before implementing; don't trust the notes below blindly.**

- **Map files:** MRC/CCP4, mode 2 (float32), correct voxel size in the header, origin 0, standard axis order (mapc/mapr/maps = 1/2/3), cubic even box. Use `mrcfile` and validate headers on write.
- **Per-decoy outputs** (EMDB-like layout, but never real EMDB IDs):
  ```
  decoy_000123/
    decoy_000123.mrc            # full map (noisy, averaged halves)
    decoy_000123_half_1.mrc
    decoy_000123_half_2.mrc
    decoy_000123_clean.mrc      # noise-free target (for DeepEMhancer-style supervised pairs)
    decoy_000123_mask.mrc       # envelope
    decoy_000123_pseudoatoms.cif  # HETATM records, residue name UNX (unknown atom/ion), no polymer entity
    decoy_000123.json           # all parameters, seed, versions, measured diagnostics
  ```
- `manifest.csv` across the dataset: id, paths, all parameters, diagnostics, critic scores.
- **Labels for builder-style training:** decoys are negatives. Provide an all-zero Cα/backbone label volume on request, and the pseudo-atom positions (explicitly non-protein).
- **Supervised denoising pairs:** (half maps or noisy full map) → clean map, at the same voxel size.
- **Wrapper scripts** that run the critics on a dataset folder:
  - DeepEMhancer (TensorFlow/Keras; accepts a full map or half maps; check which precomputed model to use and any internal resampling).
  - ModelAngelo `build_no_seq` mode (PyTorch); record residues built and mean confidence; if possible, also extract the first-stage Cα probability map.
- **Implementation stack:** PyTorch for the generator (GPU FFTs; differentiable if needed later), NumPy/SciPy fallback acceptable for v0. Deterministic seeding throughout. Generation of a 64³–128³ decoy should take seconds, not minutes.

---

## 6. Scoring and optimization (no real data)

### 6.1 Critics

1. **DeepEMhancer phase residual (tier 2).** For a denoiser trained to minimize mean squared error, Tweedie's formula gives D(x) − x ≈ σ²∇log p(x), so the residual is the gradient of the learned protein prior (as in regularization-by-denoising, RED). DeepEMhancer also sharpens, so separate amplitude from phase: match the radial amplitudes of x and D(x), then compute **1 − FSC(x, D(x)) over the 3–8 Å shells**. Repeat at several added-noise levels to get a hallucination-propensity curve. A good decoy maps to itself, not toward protein.
2. **ModelAngelo buildability (tier 2).** Residues built and confidence in `build_no_seq`; sum of the Cα probability map if accessible. This is the main "is it recognizably protein" score.
3. **Latent statistics, data-free (tier 2), if a critic has BatchNorm.** BatchNorm running means and variances store the first and second moments of real-map activations per layer (DeepInversion, Yin et al. CVPR 2020). Match them in **early** layers (texture-level similarity) and push **deep** layers or the output head away from protein. First check whether DeepEMhancer, ModelAngelo or EMReady actually contain BatchNorm. Without it, use feature visualization to find maximally protein-activating directions and penalize projection onto them.
4. **Physics diagnostics (tier 0):** radial power spectrum vs. target, histogram, Richardson D, compactness, junction density, local anisotropy (structure-tensor eigenvalue distribution), a chirality statistic, half-map FSC.

### 6.2 Optimizer

- Parameter vector of about 10–15 values (α/D, compactness, meso wavelength, meso strength, fill fraction, spectral bump heights and positions, B-factor, noise level).
- **CMA-ES or Bayesian optimization**, black-box. Objective: maximize similarity (physics diagnostics + early-layer statistics) **subject to proteinness < τ**, so decoys sit just on the non-protein side of the boundary (hard negatives).
- **No per-voxel gradient optimization against critics** by default; it yields adversarial noise. The restricted parameter space is the regularizer.
- **Held-out critic:** optimize against critic A, report against critic B (and visual inspection). If decoys will later evaluate DeepEMhancer's hallucination, do not use DeepEMhancer as the optimization critic.

---

## 7. Milestones

1. **v0 generator:** envelope (with α → D calibration) + Turing intensity field + Gerchberg–Saxton amplitude imposition + half-map rendering + MRC/JSON output + diagnostics plots. Unit tests for spectra, volume, determinism and header validity.
2. **Atomistic layer:** inhomogeneous hard-core sampler + RMC to target S(s) + `gemmi`-based rendering + pseudo-atom mmCIF.
3. **Critic wrappers:** DeepEMhancer phase-residual score, ModelAngelo buildability; batch runner over a dataset folder.
4. **Optimization loop:** CMA-ES over parameters, constraint handling, held-out evaluation, report.
5. **Dataset export:** manifest, train/val split by seed, documentation of the parameter distribution.

---

## 8. Known pitfalls

- Turing tubes ~5 Å wide at a ~10 Å period look like backbone or helices at 4–6 Å resolution. Prefer fill fractions near 0.5 (bicontinuous) or a low `meso_strength`.
- The α → D relation is only approximate after largest-component selection, edge clearing and binary projection; calibrate it.
- DeepEMhancer's amplitude changes dominate any naive "how much did it change" metric; use the phase-only score.
- Critic overfitting: always report against a held-out critic.
- Never write outputs with real EMDB/PDB identifiers or protein residue names.

---

## Appendix A — Reference code from the earlier notebooks (do not port; for intent only)

### A.1 3D Turing pattern in a mask (DoG reaction–diffusion)

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def turing_pattern_3d(size, scale=4, quotient=3, sigma_short=None, sigma_long=None,
                      gain=1, dt=.2, iterations=300, fraction=0.5, mask=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    size_3d = (size,)*3 if isinstance(size, int) else tuple(size)
    x = np.random.randn(*size_3d)
    if sigma_short is None and sigma_long is None:
        sigma_short = 4/scale                 # activation scale
        sigma_long  = sigma_short*quotient    # inhibition scale
    for i in range(iterations):
        dog = gaussian_filter(x, sigma_short) - gaussian_filter(x, sigma_long)
        dx = dt * gain * dog
        if mask is not None:
            dx = mask * dx                    # evolve only inside the envelope
        x = np.tanh(x + dx)                   # soft bounding
    if mask is not None:
        threshold = np.quantile(x[mask == 1].ravel(), fraction)
        return (x > threshold).astype(np.uint8)
    return (x > np.quantile(x, 1 - fraction)).astype(np.uint8)

# 2D notes from the notebook:  sigma_short/sigma_long  0.7/2.0 stippling,
# 1.0/4.0 vermiculation, 1.5/6.0 thick worms.  "scale": ~1 spots, 2-10 mazes, 10-30 blobs.
# 3D settings used: scale 4.5, quotient 4, gain 1, dt 0.3, 100 iterations, fill fraction 0.74.
```

### A.2 Coloured noise by iterative spectral projection (Gerchberg–Saxton-style)

```python
from numpy.fft import fftn, ifftn, fftshift, ifftshift

def make_frequency_grid(shape):
    freqs = [np.fft.fftfreq(s) for s in shape]
    grids = np.meshgrid(*freqs, indexing='ij')
    r = np.sqrt(sum(g**2 for g in grids))
    return (*grids, fftshift(r))

def target_magnitude(shape, alpha, f0=1e-6):
    r = make_frequency_grid(shape)[-1]
    mag = 1.0 / np.power(r + f0, alpha / 2.0)   # power ~ 1/f^alpha
    return mag / np.nanmax(mag)

def enforce_real_constraints(img, mode, clip_range=(0, 1), threshold=0.5):
    if mode == 'float':
        lo, hi = img.min(), img.max()
        return (img - lo) / (hi - lo) * (clip_range[1] - clip_range[0]) + clip_range[0]
    if mode == 'binary':
        t = np.sort(img.ravel())[int(img.size * threshold)]
        return (img >= t).astype(float)
    return img

def generate_colored_noise(shape, alpha=0.0, rng=None, iterations=0,
                           real_constraint='float', clip_range=(0, 1), binary_threshold=0.5):
    rng = rng or np.random.default_rng()
    img = rng.standard_normal(shape)
    mag = target_magnitude(shape, alpha)
    for _ in range(max(1, iterations)):
        phases = np.angle(fftshift(fftn(img)))           # keep phases
        img = np.real(ifftn(ifftshift(mag * np.exp(1j * phases))))  # impose amplitude
        img = enforce_real_constraints(img, real_constraint, clip_range, binary_threshold)
    return img
```

### A.3 Random envelope: threshold coloured noise, keep largest object, centre periodically

```python
from scipy.ndimage import label, center_of_mass

def random_mask_3d(shape, alpha=4.2, coverage=0.95, rng_seed=None):
    base = generate_colored_noise(shape, alpha, np.random.default_rng(rng_seed),
                                  iterations=10, real_constraint='binary',
                                  binary_threshold=coverage)
    return 1 - base

def largest_object_nd(b):
    lab, n = label(b, structure=np.ones((3,)*b.ndim, bool))
    if n == 0:
        return np.zeros_like(b, bool)
    return lab == (np.bincount(lab.ravel())[1:].argmax() + 1)

def periodic_select_and_center_nd(b):
    tiled = largest_object_nd(np.tile(b, (3,)*b.ndim))   # handle wrap-around objects
    if not tiled.any():
        return np.zeros_like(b)
    c = center_of_mass(tiled)
    idx = [(np.arange(s) + int(ci - s//2)) % ts for s, ci, ts in zip(b.shape, c, tiled.shape)]
    return tiled[np.ix_(*idx)].astype(b.dtype)

# 3D settings used: box 180, alpha_mask 4.2-4.5, coverage ~0.05-0.08 (occupied fraction),
# up to 10 tries; a valid mask also needed n_edge (~10 px) clearance to the box edge.
# 2D notes: alpha 1-3 roundish, 3-4 sharp-edged, 4-5 dominated by fine detail
# (empirical, after binary projection).
```

### A.4 Earlier map assembly (blur + background noise + soft mask)

```python
# pattern_3d: binary Turing pattern inside blob_mask
r = make_frequency_grid(pattern_3d.shape)[-1]
mtf = np.sinc(r / 0.2)                                   # crude sinc MTF, cutoff 0.2 cycles/px
blur = lambda v: np.real(ifftn(ifftshift(fftshift(fftn(v)) * mtf)))

protein_like = blur(pattern_3d.astype(np.float32) - 0.4)          # 0.4 = "mass adjustment" offset
noise = 1.5 * generate_colored_noise(pattern_3d.shape, alpha=-0.5, iterations=10,
                                      clip_range=(-1, 1))           # slightly blue noise
background = blur(noise * (1 - blob_mask))
soft = gaussian_filter(blob_mask.astype(float), sigma=5)
final = protein_like * soft + background * (1 - soft)
# A later step applied a Gaussian low-pass whose high-frequency tail was multiplied by
# noise to retain high-frequency texture. Saved as MRC with voxel_size = 1.0.
```

### A.5 Earlier evaluation idea

Generated maps were passed through DeepEMhancer, and the magnitude of change was used as a score of protein-likeness. §6.1 replaces this with a phase-only, multi-noise-level version.

### A.6 Related earlier work (2D, point clouds)

A 2D notebook tested point-cloud representations: Mitchell best-candidate blue-noise sampling inside random masks, Delaunay triangulation restricted to the mask, and a Metropolis Monte Carlo "spring graph" relaxation (`mc_step(sigma, temperature, mode='metropolis')`). The Metropolis machinery is conceptually the starting point for the RMC in §4.3 (energy becomes χ² against target S(s)).
