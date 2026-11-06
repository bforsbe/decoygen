"""2-D toy of the symmetric envelope and its subdivision into subunits (Cn).

Stages (the 3-D version will follow the same steps):
  1 subunit field   f1 = coarse GRF + a * fine GRF - b * q, in the subunit's own frame; q is a bowl
                    centred at distance r from the axis (plus, for a domain swap, a notch where the arm
                    would sit and an arm reaching into the neighbour's notch)
  2 copies          f_k(x) = f1(g_k^-1 x) for every symmetry operation g_k (rotation by 2 pi k / n)
  3 competition     F = max_k f_k (assembly field), label = argmax_k f_k (who owns each pixel)
  4 threshold       F > t with t set so the area is n * A_subunit (largest component)
  5 smoothing       MBO step, then group averaging m <- mean_k m(g_k^-1 x), re-threshold at 0.5: any
                    step that ignores symmetry is made symmetric this way
  6 placement       rigid transform of the finished assembly (axis need not be at the box centre)

usage: python toy2d.py OUT.png
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import ndimage

N = 256                                   # grid (px)
AREA = 2200.0                             # subunit area (px^2)
R = np.sqrt(AREA / np.pi)                 # subunit radius
ALPHA, AMP, BIAS = 3.6, 0.55, 1.6         # 2-D: alpha = 2H + 2 -> H = 0.8
# Okabe-Ito, fixed order around the ring (no two neighbours alike), solvent off-white
SUB = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]
SOLVENT = "#f6f6f4"
yy, xx = np.mgrid[0:N, 0:N].astype(float)
C0 = np.array([N / 2, N / 2])


def grf(rng):
    """Split-field GRF on the grid: (coarse, fine), each unit variance, periodic."""
    k = np.sqrt(np.fft.fftfreq(N)[:, None] ** 2 + np.fft.fftfreq(N)[None, :] ** 2)
    k[0, 0] = 1.0
    w = np.fft.fft2(rng.normal(size=(N, N)))
    amp = k ** (-ALPHA / 2)
    amp[0, 0] = 0
    low = 1 / (1 + (k * R) ** 4)
    out = []
    for f in (low, 1 - low):
        g = np.fft.ifft2(w * amp * f).real
        out.append((g - g.mean()) / g.std())
    return out


def axis_distance(n, contact):
    """Distance r of each subunit centre from the axis such that neighbouring bowls (radius R)
    overlap by `contact` x R: neighbour spacing 2 r sin(pi / n) = (2 - contact) R."""
    return (2 - contact) * R / (2 * np.sin(np.pi / n))


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def local(points, theta, centre=C0):
    """Coordinates of box points in the frame of a copy rotated by theta about `centre`."""
    return (points - centre) @ rot(theta) + centre                 # g^-1 x  (row vectors)


def bowl(p, c, radius):
    return ((p - c) ** 2).sum(-1) / radius ** 2


def subunit_q(p, n, r, swap):
    """Confinement of subunit 0 in its own frame (axis at C0, body centred at angle 0, distance r).
    swap > 0 (domain swap): in a monomer the arm would dock in a slot on the body's edge facing the
    previous copy. Here the body gives that slot up (notch), and the arm docks in the same slot of the
    NEXT copy, g.S, which sits on that copy's edge facing us, so the arm crosses the interface."""
    body = C0 + np.array([r, 0.0])
    q = bowl(p, body, R)
    if swap > 0:
        g = rot(2 * np.pi / n)
        prev = C0 + g.T @ (body - C0)
        slot = body + 0.55 * R * (prev - body) / np.linalg.norm(prev - body)
        q = q + 3.0 * swap * np.exp(-bowl(p, slot, 0.42 * R))       # body gives up its slot
        target = C0 + g @ (slot - C0)                               # the next copy's slot
        d = target - body
        t = np.clip(((p - body) @ d) / (d @ d), 0, 1)
        arm = ((p - (body + t[..., None] * d)) ** 2).sum(-1) / (0.38 * R) ** 2
        q = np.minimum(q, arm + 0.2 * (1 - swap))
    return q


def sample(img, p, order=1):
    return ndimage.map_coordinates(img, [p[..., 1], p[..., 0]], order=order, mode="grid-wrap")


def copies(coarse, fine, n, r, swap):
    """f_k on the box for k = 0..n-1 (stage 2). The GRF of subunit 0 lives in its own frame."""
    pts = np.stack([xx, yy], -1)
    out = []
    for k in range(n):
        p = local(pts, 2 * np.pi * k / n)
        out.append(sample(coarse, p, 3) + AMP * sample(fine, p, 3) - BIAS * subunit_q(p, n, r, swap))
    return np.array(out)


def largest(mask):
    lab, m = ndimage.label(mask)
    if m == 0:
        return mask
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)


def threshold_to_area(field, area):
    lo, hi = np.percentile(field, 1), field.max()
    for _ in range(40):
        t = (lo + hi) / 2
        a = largest(field > t).sum()
        lo, hi = (t, hi) if a > area else (lo, t)
    return largest(field > t)


def symmetrise(mask, n, area):
    """Group average of a binary mask, re-thresholded to the area (stage 5); pinholes filled
    (a ring's pore is far larger and stays)."""
    pts = np.stack([xx, yy], -1)
    avg = np.mean([sample(mask.astype(float), local(pts, 2 * np.pi * k / n)) for k in range(n)], 0)
    m = threshold_to_area(avg, area)
    holes, _ = ndimage.label(ndimage.binary_fill_holes(m) & ~m)
    small = np.flatnonzero(np.bincount(holes.ravel())[1:] < 0.02 * area) + 1
    return m | np.isin(holes, small)


def mbo(mask, sigma, area):
    return threshold_to_area(ndimage.gaussian_filter(mask.astype(float), sigma), area)


def tidy_owners(sub, mask):
    """Keep each subunit's largest piece; hand every stray island to the nearest kept subunit.
    Applied identically to every copy, so it keeps the symmetry (up to interpolation)."""
    out = sub.copy()
    for k in np.unique(sub[sub > 0]):
        lab, m = ndimage.label(sub == k)
        if m > 1:
            big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
            out[(lab > 0) & (lab != big)] = 0
    hole = mask & (out == 0)
    if hole.any():
        _, (iy, ix) = ndimage.distance_transform_edt(out == 0, return_indices=True)
        out[hole] = out[iy[hole], ix[hole]]
    return out


def build(n, contact, swap=0.0, seed=0):
    r = axis_distance(n, contact)
    coarse, fine = grf(np.random.default_rng(seed))
    f = copies(coarse, fine, n, r, swap)
    F = f.max(0)
    label = np.array([ndimage.gaussian_filter(x, 1.5) for x in f]).argmax(0)   # who owns each pixel
    raw = threshold_to_area(F, n * AREA)
    smooth = symmetrise(mbo(raw, 2.0, n * AREA), n, n * AREA)
    sub = tidy_owners(np.where(smooth, label + 1, 0), smooth)
    if len(np.unique(sub[sub > 0])) < n:
        raise ValueError(f"C{n} contact {contact}: copies do not touch, assembly falls apart")
    return {"coarse": coarse, "fine": fine, "f": f, "F": F, "label": label, "raw": raw, "final": smooth,
            "sub": sub, "r": r}


def place(sub, theta, shift):
    """Rigid transform of the label map: rotate by theta about the box centre, then shift (px)."""
    pts = np.stack([xx, yy], -1)
    p = (pts - C0 - shift) @ rot(theta) + C0
    return ndimage.map_coordinates(sub, [p[..., 1], p[..., 0]], order=0, mode="constant")


def check(sub, n):
    areas = [int((sub == k + 1).sum()) for k in range(n)]
    pieces = [ndimage.label(sub == k + 1)[1] for k in range(n)]
    contact = []
    for k in range(n):
        a = sub == k + 1
        b = sub == (k + 1) % n + 1
        contact.append(int((a & ndimage.binary_dilation(b)).sum()))
    return areas, pieces, contact


W = 78                                    # half-width of the zoomed panels (px)


def zoom(ax, centre=C0, w=W):
    ax.set_xlim(centre[0] - w, centre[0] + w)
    ax.set_ylim(centre[1] - w, centre[1] + w)


def show_field(ax, f, level, title):
    """Field clipped to a band around the threshold, so the roughness is visible; outline at the level."""
    ax.imshow(f, cmap="Greys_r", origin="lower", vmin=level - 2.5, vmax=level + 1.5)
    ax.contour(f, levels=[level], colors="#1f2328", linewidths=1.0, origin="lower")
    ax.plot(*C0, "+", color="#d62728", ms=10, mew=2)
    ax.set_title(title, fontsize=9, loc="left", color="#1f2328")
    zoom(ax)


def show_labels(ax, sub, n, axis=C0, title="", centre=C0, w=W):
    cmap = ListedColormap([SOLVENT] + SUB[:n])
    ax.imshow(sub, cmap=cmap, vmin=-0.5, vmax=n + 0.5, origin="lower", interpolation="nearest")
    for k in range(n):                                           # each subunit's own outline
        ax.contour(sub == k + 1, levels=[0.5], colors="#1f2328", linewidths=0.8, origin="lower")
    ax.plot(*axis, "+", color="#d62728", ms=10, mew=2)
    ax.set_title(title, fontsize=9, loc="left", color="#1f2328")
    zoom(ax, centre, w)


def level_for(F, area):
    return np.sort(F.ravel())[::-1][int(area)]


def tidy(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d0d0d0")


def copies_panel(ax, f, F, n, title):
    lev = level_for(F, n * AREA)
    for k in range(n):
        ax.contourf(f[k] > lev, levels=[0.5, 1.5], colors=[SUB[k]], alpha=0.35, origin="lower")
        ax.contour(f[k], levels=[lev], colors=[SUB[k]], linewidths=1.4, origin="lower")
    ax.set_aspect("equal")
    ax.plot(*C0, "+", color="#d62728", ms=10, mew=2)
    ax.set_title(title, fontsize=9, loc="left", color="#1f2328")
    zoom(ax)


def main(out):
    n, contact = 3, 0.6
    s = build(n, contact, seed=4)
    r = s["r"]
    fig, axs = plt.subplots(3, 6, figsize=(17, 9.6), layout="constrained")
    for ax in axs.ravel():
        tidy(ax)
    pts = np.stack([xx, yy], -1)

    # row 1: the stages, C3
    a = axs[0]
    f0 = s["f"][0]
    show_field(a[0], f0, level_for(f0, AREA), "1  subunit field f1\n(red +: symmetry axis)")
    a[0].plot(*(C0 + [r, 0]), "o", color=SUB[0], ms=5)
    copies_panel(a[1], s["f"], s["F"], n, "2  copies f1(g_k^-1 x), k = 0..2\n(overlap = future interface)")
    show_field(a[2], s["F"], level_for(s["F"], n * AREA), "3a assembly field F = max_k f_k")
    a[3].imshow(s["label"], cmap=ListedColormap(SUB[:n]), origin="lower", alpha=0.8)
    a[3].contour(s["raw"], levels=[0.5], colors="#1f2328", linewidths=1.2, origin="lower")
    a[3].plot(*C0, "+", color="#d62728", ms=10, mew=2)
    a[3].set_title("3b owner = argmax_k f_k\n4  threshold to 3 x area (black)", fontsize=9, loc="left")
    zoom(a[3])
    raw_sub = tidy_owners(np.where(s["raw"], s["label"] + 1, 0), s["raw"])
    show_labels(a[4], raw_sub, n, title="4  subunits before smoothing")
    show_labels(a[5], s["sub"], n, title="5  MBO + group average:\nfinal envelope and interfaces")

    # row 2: placement and variants (contact = overlap of neighbouring bowls, in R)
    b = axs[1]
    shift = np.array([-40.0, 30.0])
    show_labels(b[0], place(s["sub"], 0.7, shift), n, axis=C0 + shift, centre=C0, w=N / 2,
                title="6  placed in the box:\naxis off centre, rotated")
    variants = [(2, 0.6, 1, "C2"), (5, 0.6, 2, "C5"), (7, 0.5, 3, "C7: ring with a pore"),
                (4, 1.1, 5, "C4, deep contact"), (3, 0.25, 6, "C3, shallow contact")]
    for ax, (nn, cc, sd, t) in zip(b[1:], variants):
        v = build(nn, cc, seed=sd)
        _, _, cont = check(v["sub"], nn)
        w = max(W, v["r"] + 2.2 * R)
        show_labels(ax, v["sub"], nn, w=w, title=f"{t}, contact {cc} R\ninterface {np.mean(cont):.0f} px/subunit")

    # row 3: domain swap
    c = axs[2]
    sw = build(3, 0.6, swap=1.0, seed=4)
    q = subunit_q(pts, 3, sw["r"], 1.0)
    c[0].imshow(-q, cmap="Greys_r", origin="lower", vmin=-2.5, vmax=0)
    c[0].plot(*C0, "+", color="#d62728", ms=10, mew=2)
    c[0].set_title("swap 1  bowl with a notch +\narm towards the next copy", fontsize=9, loc="left")
    zoom(c[0])
    f0 = sw["f"][0]
    show_field(c[1], f0, level_for(f0, AREA), "swap 2  subunit field f1")
    copies_panel(c[2], sw["f"], sw["F"], 3, "swap 3  copies: each arm lies\nin the next copy's notch")
    show_labels(c[3], sw["sub"], 3, title="swap 4-5  subunits")
    one = np.where(sw["sub"] == 1, 2, np.where(sw["sub"] > 0, 1, 0))
    c[4].imshow(one, cmap=ListedColormap([SOLVENT, "#dcdcdc", SUB[0]]), origin="lower", interpolation="nearest")
    c[4].contour(sw["sub"] > 0, levels=[0.5], colors="#1f2328", linewidths=1.0, origin="lower")
    c[4].plot(*C0, "+", color="#d62728", ms=10, mew=2)
    c[4].set_title("swap  subunit 1 alone: its arm\nreaches across into subunit 2", fontsize=9, loc="left")
    zoom(c[4])
    a0, p0, c0 = check(s["sub"], 3)
    a1, p1, c1 = check(sw["sub"], 3)
    c[5].axis("off")
    c[5].text(0, 1, "checks       C3 (row 1)   swap (row 3)\n"
              f"area px      {a0}\n             {'':12s}{a1}\n"
              f"pieces       {p0}   {p1}\n"
              f"interface px {c0}   {c1}\n\n"
              "Equal areas, one piece each:\n"
              "symmetric up to interpolation.\n"
              "In 3-D the atoms are built in one\n"
              "subunit and replicated: exact.\n\n"
              "2-D only: a ring's pore is a hole;\n"
              "in 3-D it is a tunnel.",
              va="top", fontsize=8.5, family="monospace", color="#1f2328")
    fig.savefig(out, dpi=130)
    print("C3", a0, p0, c0, "| swap", a1, p1, c1)


if __name__ == "__main__":
    main(sys.argv[1])
