"""Reciprocal-space grid, shell averaging and FFT-based filters.

Conventions: arrays are cubic (n, n, n) float32 on `device`; voxel i sits at
i * apix (origin 0); FFTs are periodic real FFTs (rfftn). Spatial frequency s
is in 1/A. Shell averages use a sort + cumsum in float64 so they are
deterministic on GPU (no atomics).
"""

from __future__ import annotations

import math

import torch


class Grid:
    def __init__(self, n: int, apix: float, device="cpu"):
        self.n = n
        self.apix = apix
        self.device = torch.device(device)
        f = torch.fft.fftfreq(n, d=apix, device=self.device)
        fr = torch.fft.rfftfreq(n, d=apix, device=self.device)
        kz, ky, kx = torch.meshgrid(f, f, fr, indexing="ij")
        self.k = (kz, ky, kx)
        self.s2 = kz**2 + ky**2 + kx**2
        self.s = torch.sqrt(self.s2)
        # Shell index in units of 1/(n*apix); corners beyond Nyquist get their own shells.
        self.shell = torch.round(self.s * n * apix).long()
        self.nshell = int(self.shell.max().item()) + 1
        flat = self.shell.reshape(-1)
        self._order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat.cpu(), minlength=self.nshell).to(self.device)
        self.shell_counts = counts
        self._ends = torch.cumsum(counts, 0) - 1
        self.shell_s = torch.arange(self.nshell, device=self.device, dtype=torch.float64) / (n * apix)
        # Derivative multipliers; the Nyquist plane is zeroed so odd derivatives stay real.
        nyq = n // 2

        def deriv(k, axis):
            d = 2j * math.pi * k
            idx = [slice(None)] * 3
            idx[axis] = nyq
            d = d.clone()
            d[tuple(idx)] = 0
            return d

        self._deriv = tuple(deriv(k, a) for a, k in enumerate(self.k))

    @property
    def shape(self):
        return (self.n,) * 3

    @property
    def nyquist_shell(self) -> int:
        return self.n // 2

    def rfft(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfftn(x)

    def irfft(self, X: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfftn(X, s=self.shape)

    def shell_mean(self, values: torch.Tensor) -> torch.Tensor:
        """Mean of a half-space (rfft-shaped) real array over each shell, float64."""
        v = values.reshape(-1)[self._order].double()
        cs = torch.cumsum(v, 0)
        sums = cs[self._ends.clamp(min=0)]
        sums = torch.cat([sums[:1], sums[1:] - sums[:-1]])
        return sums / self.shell_counts.clamp(min=1)

    def expand(self, profile: torch.Tensor) -> torch.Tensor:
        """Broadcast a per-shell profile to the rfft grid."""
        return profile[self.shell]

    def power_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        X = self.rfft(x)
        return self.shell_mean(X.real**2 + X.imag**2)

    def gaussian_filter(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Periodic Gaussian blur, sigma in A."""
        if sigma <= 0:
            return x
        return self.irfft(self.rfft(x) * torch.exp(-2 * math.pi**2 * sigma**2 * self.s2))

    def gradient(self, x: torch.Tensor):
        X = self.rfft(x)
        return tuple(self.irfft(X * d) for d in self._deriv)

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        return self.irfft(self.rfft(x) * (-4 * math.pi**2 * self.s2))

    def cosine_lowpass(self, x: torch.Tensor, cutoff: float, width: float) -> torch.Tensor:
        """Raised-cosine low-pass: 1 below `cutoff`, 0 above `cutoff + width` (1/A)."""
        t = ((self.s - cutoff) / width).clamp(0, 1)
        return self.irfft(self.rfft(x) * (0.5 * (1 + torch.cos(math.pi * t))))

    def fsc(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        A, B = self.rfft(a), self.rfft(b)
        num = self.shell_mean((A * B.conj()).real)
        den = torch.sqrt(self.shell_mean(A.abs() ** 2) * self.shell_mean(B.abs() ** 2))
        return num / den.clamp(min=1e-30)
