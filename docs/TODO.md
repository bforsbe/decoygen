# To do

## Packing and envelope (from the packing analysis, 2026-11-06)

- [ ] **Protein baseline vs size.** Run `experiments/packing/packing.py` over a size-stratified PDB set (~200
      structures, 5–500 kDa). Measure volume per Da (1.4 A probe), interior atom density, depth profile,
      pocket fraction, cavities. These become the calibration target for decoygen, and the baseline for the
      packing/density project.
- [ ] **Fit the envelope to real structures, as functions of mass.** Fit `roughness_D` / amplitude,
      `compactness`, `n_domains`, `elongation` to the baseline above, not only to a fractal-dimension target.
      Decoy surface pockets are 0.12 vs 0.13–0.18 for proteins (rung 1 gas: 0.07).
- [ ] **Target the molecular volume of the finished atoms, not the mask volume.**
  - Rung 5 atoms fill 115 000 of the 121 000 A³ envelope; relaxation leaves an empty skin.
  - 1.21 A³/Da is the partial specific volume, but proteins measured geometrically (1.4 A probe) give about
    1.26 A³/Da.
  - Together these leave rung 5 about 9 % denser than protein.
- [ ] Commit the atom bookkeeping (`thread` conserve, `peptide` bypass), the new test and `experiments/packing/`
      once the rebuilt rungs 3–5 are evaluated (ModelAngelo jobs 18147615, 18147616).

## Ladder

- [ ] Rungs 6–7: secondary structure, fold.
- [ ] Symmetry and multi-domain assemblies. Started with Cn: 2-D toy in `experiments/symmetry/toy2d.py`
      (competing copies of one subunit field; owner = argmax; group averaging after non-symmetric steps).
  - Design and toy figure: `docs/SYMMETRY.md`, `docs/figures/symmetry_cn_toy2d.png`.
  - [ ] Atoms built symmetrically along the way: ASU + symmetry operations in the file, images in every
        relaxation's sterics, ASU confined to its own subunit region (SYMMETRY.md, "Atoms").
  - [x] 3-D Cn envelope + subunit label map (`decoygen envelope --symmetry Cn`), general rotation-list design.
  - [ ] Core exclusion for hollow assemblies (SYMMETRY.md, "Hollow assemblies"): a Bernoulli draw p(group, n),
        pore radius as the controlled quantity, a group-invariant term in the field; fill only voids cut off
        from solvent and outside the core region; any pore topology allowed (channel, bowl, closed).
  - [ ] Swap: C3 still mostly opens a cleft (6 % of a subunit across vs 3 % without); tune.
  - [ ] Interface area from a smooth surface (not voxel faces) before calibrating.
  - [ ] Calibrate interface area per subunit, pore presence p(group, n), pore radius and swap frequency against
        real oligomers in the PDB
        (the bowl overlap `contact` is only the knob; interface area is the target).
- [ ] Junction density (skeleton Y-junctions per volume) as a non-protein marker (METHODS §9).
