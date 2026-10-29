"""User-facing parameter set for one decoy."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .constants import PROTEIN_VOLUME_PER_DA

TARGET_SPECTRA = ("physics", "physics+peaks", "best")
NOISE_MODELS = ("white", "none")


@dataclass
class DecoyParams:
    # Sampling
    box: int = 128                      # cubic box edge (voxels), even
    apix: float = 1.0                   # voxel size (A)
    resolution: float = 3.5             # target resolution d (A)

    # Envelope
    mass_kda: float = 100.0             # sets mask volume V = 1.21 A^3/Da * M
    roughness_D: float = 2.25           # surface fractal dimension (roughness exponent), via calibration
    alpha: float | None = None          # GRF spectral exponent; default 9 - 2*roughness_D
    roughness_amplitude: float | None = None  # fine/coarse GRF amplitude; default from the calibration table
    compactness: float = 8.0            # mean-curvature-flow time (A^2), MBO scheme
    n_domains: int = 1
    elongation: float = 1.0             # >= 1, volume-preserving stretch along a random axis
    envelope_bias: float = 1.5          # weight of the quadratic confinement term vs. the unit-variance GRF
    edge_clearance: float = 10.0        # A between envelope and box edge

    # Mesoscale (Turing) field
    meso_wavelength: float = 10.0       # A
    meso_strength: float = 0.3          # 0..1
    fill_fraction: float = 0.5          # Turing level-set fraction; ~0.5 bicontinuous
    turing_quotient: float = 4.0        # sigma_long / sigma_short of the DoG growth function
    turing_iterations: int = 200
    turing_dt: float = 0.3

    # Target spectrum
    target_spectrum: str = "physics"
    peak_d: tuple[float, ...] = (10.0, 4.7)       # A, positions of isotropic S(s) bumps at s = 1/d
    peak_heights: tuple[float, ...] = (0.5, 0.3)
    peak_rel_width: float = 0.15                  # Gaussian sigma as a fraction of peak s
    spectrum_file: str | None = None              # 'best': two columns s (1/A), S(s) multiplier
    allow_tier3: bool = False
    gs_iterations: int = 20

    # Rendering and noise
    bfactor: float = 40.0               # A^2
    noise_model: str = "white"
    snr_scale: float = 1.0              # multiplies half-map SNR; 1.0 puts FSC=0.143 at `resolution`

    seed: int = 0

    def validate(self) -> None:
        if self.box <= 0 or self.box % 2:
            raise ValueError(f"box must be a positive even integer, got {self.box}")
        if self.apix <= 0:
            raise ValueError("apix must be positive")
        if self.resolution < 2 * self.apix:
            raise ValueError(f"resolution {self.resolution} A is beyond Nyquist (2*apix = {2 * self.apix} A)")
        if not 2.0 <= self.roughness_D <= 2.6:
            raise ValueError("roughness_D outside the supported range 2.0-2.6")
        if self.alpha is not None and not 2.0 < self.alpha < 6.0:
            raise ValueError("alpha outside the supported range (2, 6)")
        if self.roughness_amplitude is not None and self.roughness_amplitude < 0:
            raise ValueError("roughness_amplitude must be >= 0")
        if self.compactness < 0:
            raise ValueError("compactness must be >= 0")
        if self.n_domains < 1:
            raise ValueError("n_domains must be >= 1")
        if self.elongation < 1.0:
            raise ValueError("elongation must be >= 1")
        if not 0.0 <= self.meso_strength <= 1.0:
            raise ValueError("meso_strength must be in [0, 1]")
        if not 0.05 <= self.fill_fraction <= 0.95:
            raise ValueError("fill_fraction must be in [0.05, 0.95]")
        if self.meso_wavelength < 2 * self.apix:
            raise ValueError("meso_wavelength below Nyquist")
        if self.target_spectrum not in TARGET_SPECTRA:
            raise ValueError(f"target_spectrum must be one of {TARGET_SPECTRA}")
        if self.target_spectrum == "best" and not (self.allow_tier3 and self.spectrum_file):
            raise ValueError("target_spectrum='best' is tier 3: needs allow_tier3=True and spectrum_file")
        if len(self.peak_d) != len(self.peak_heights):
            raise ValueError("peak_d and peak_heights must have equal length")
        if self.noise_model not in NOISE_MODELS:
            raise ValueError(f"noise_model must be one of {NOISE_MODELS}")
        if self.bfactor < 0:
            raise ValueError("bfactor must be >= 0")
        # Irregular envelopes reach ~1.5-1.9 R from their centre; reject hopeless boxes up front
        # (make_envelope retries the rest and raises EnvelopeError if none fits).
        extent = 2 * 1.6 * self.equivalent_radius * self.elongation * self.n_domains ** (1 / 3)
        if extent + 2 * self.edge_clearance > self.box * self.apix:
            raise ValueError(
                f"box ({self.box * self.apix:.0f} A) too small for mass {self.mass_kda} kDa "
                f"(rough extent {extent:.0f} A + clearance)"
            )

    @property
    def volume(self) -> float:
        """Envelope volume in A^3."""
        return PROTEIN_VOLUME_PER_DA * self.mass_kda * 1e3

    @property
    def equivalent_radius(self) -> float:
        return (3 * self.volume / (4 * 3.141592653589793)) ** (1 / 3)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["peak_d"] = list(self.peak_d)
        d["peak_heights"] = list(self.peak_heights)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> DecoyParams:
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        d = dict(d)
        for key in ("peak_d", "peak_heights"):
            if key in d:
                d[key] = tuple(d[key])
        return cls(**d)
