"""Fit a multi-scale dot field to the hero original by analysis-by-synthesis.

Dots are Gaussians, peak-normalised, summed additively in display space, which
is exactly what the WebGL page draws. Greedy matching pursuit places each dot
where the unexplained energy is largest, at whichever scale explains it best,
then amplitudes are jointly refined. Large scales become the neon bloom, small
scales the beads and stipple, and clipping makes the white-hot cores.
"""
import base64, json, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

SRC = 'assets/logo-v3-original.png'
N, GRID = int(sys.argv[1]), int(sys.argv[2])
OUT = sys.argv[3] if len(sys.argv) > 3 else 'tools/dots.json'
BLOOM = float(os.environ.get('BLOOM', 14.))
GAP = float(os.environ.get('GAP', 1.8)) * BLOOM  # overlapping Gaussians sum smooth; scattered ones read as lumps
SCALES = np.array([.8, 1.15, 1.7, 2.6, BLOOM])
LATTICE = len(SCALES) - 1
OVERLAP = 2 * np.pi * (BLOOM / GAP) ** 2
AMAX = 2.5
CAP = np.where(SCALES > 4, float(os.environ.get('BIG', .3)), AMAX * 2)  # big dots are bloom: never let one read as a blotch
CORE = float(sys.argv[4]) if len(sys.argv) > 4 else 1.

im = np.asarray(Image.open(SRC).convert('RGB')).astype(np.float32) / 255
bg = np.median(im.reshape(-1, 3), 0)
d = np.clip(im - bg, 0, 1)
V = d.max(2)
H, W = V.shape

def gauss(a, s):
    return ndi.gaussian_filter(a, float(s), truncate=3.0, mode='constant')

# hue field: tent-weighted fit of a GRID x GRID map, ignoring blown-out cores, which carry
# no hue. The fit and the page share this exact map, so what is solved is what is drawn.
HALO = float(os.environ.get('HALO', .35))
w = ((V > .05) & (V < HALO)) * V  # the hue of the glow, not of the bead: gold belongs to the core
cell_x, cell_y = (W - 1) / (GRID - 1), (H - 1) / (GRID - 1)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
fx, fy = xx / cell_x, yy / cell_y
ix, iy = np.minimum(fx.astype(int), GRID - 2), np.minimum(fy.astype(int), GRID - 2)
ax, ay = fx - ix, fy - iy
num = np.zeros((GRID, GRID, 3)); den = np.zeros((GRID, GRID))
for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
    t = (ax if dx else 1 - ax) * (ay if dy else 1 - ay) * w
    for c in range(3):
        np.add.at(num[..., c], (iy + dy, ix + dx), (t * d[..., c]))
    np.add.at(den, (iy + dy, ix + dx), t)
ok = den > 1e-3
grid = np.zeros((GRID, GRID, 3)); grid[ok] = num[ok] / den[ok][:, None]
near = ndi.distance_transform_edt(~ok, return_distances=False, return_indices=True)
grid = grid[near[0], near[1]]
grid /= np.maximum(grid.max(2, keepdims=True), 1e-3)
grid = np.round(grid * 255) / 255
hue = (grid[iy, ix] * ((1 - ax) * (1 - ay))[..., None] + grid[iy, ix + 1] * (ax * (1 - ay))[..., None]
       + grid[iy + 1, ix] * ((1 - ax) * ay)[..., None] + grid[iy + 1, ix + 1] * (ax * ay)[..., None]).astype(np.float32)

# placement target: where red clips into a solid line, separate beads survive only in green.
# Dots are placed on this, then their brightness is solved against the true image below.
GAIN = float(os.environ.get('GAIN', 3.))
hot = np.clip((V - .45) / .4, 0, 1)
T = np.minimum(np.maximum(V, hot * d[..., 1] * GAIN), AMAX * 2).astype(np.float32)

class Field:
    def __init__(s): s.x = np.zeros(0, int); s.y = np.zeros(0, int); s.k = np.zeros(0, int); s.a = np.zeros(0)
    def add(s, x, y, k, a):
        s.x = np.r_[s.x, x]; s.y = np.r_[s.y, y]; s.k = np.r_[s.k, k]; s.a = np.r_[s.a, a]
    def keep(s, m):
        s.x, s.y, s.k, s.a = s.x[m], s.y[m], s.k[m], s.a[m]

def render(f, chan=None):
    M = np.zeros((H, W), np.float32)
    for i, sig in enumerate(SCALES):
        m = f.k == i
        if not m.any(): continue
        D = np.zeros((H, W), np.float32)
        w = f.a[m] if chan is None else f.a[m] * chan[m]
        np.add.at(D, (f.y[m], f.x[m]), w)
        M += gauss(D, sig) * (2 * np.pi * sig * sig)
    return M

sat = T >= AMAX * 2 - .01
def residual(f):
    M = render(f)
    R = T - M
    R[sat & (M > T)] = 0      # white-hot pixels are a floor, not a target
    return R

def refine(f, n=3, step=.55):
    for _ in range(n):
        R = residual(f)
        for i, sig in enumerate(SCALES):
            m = f.k == i
            if m.any():
                # overlapping lattice dots share the same residual, so each takes its share
                share = OVERLAP if i == LATTICE else 1.
                f.a[m] += step / share * 2 * gauss(R, sig)[f.y[m], f.x[m]]
        f.a = np.clip(f.a, 0, CAP[f.k])

# bloom: a regular lattice wherever the original glows, solved below like any other dot
f = Field()
soft = gauss(V, BLOOM)
ly, lx = np.mgrid[0:H:GAP, 0:W:GAP].astype(int)
ly, lx = ly.ravel(), lx.ravel()
keep = soft[ly, lx] > float(os.environ.get('FLOOR', .03))
f.add(lx[keep], ly[keep], np.full(keep.sum(), LATTICE), .4 * soft[ly, lx][keep] / OVERLAP)
print('bloom lattice', keep.sum(), end='  ')
K = max(N // 45, 40)
while len(f.a) < N * 1.08:
    R = residual(f)
    F = np.stack([gauss(R, s) for s in SCALES[:LATTICE]])
    S = F * SCALES[:LATTICE, None, None] ** .8  # below 1 favours small dots: detail over blur
    best = S.argmax(0); score = S.max(0)
    pk = peak_local_max(score, min_distance=2, threshold_abs=1e-4, num_peaks=K, exclude_border=False)
    if not len(pk): break
    y, x = pk[:, 0], pk[:, 1]
    k = best[y, x]
    f.add(x, y, k, np.clip(1.6 * F[k, y, x], 0, CAP[k]))
    refine(f, 2)
refine(f, 14)
o = np.argsort(-f.a * np.where(f.k == LATTICE, 99., SCALES[f.k]))[:N]  # weakest contribution goes first, never the lattice
f.keep(np.sort(o))

# positions are fixed: now solve brightness against the picture itself
T = V.astype(np.float32); sat = T >= .97
CAP = np.where(SCALES > 4, CAP, AMAX)
f.a = np.minimum(f.a, CAP[f.k])
refine(f, 30)


# quantise exactly as the page will decode
lo, hi = max(f.a[f.a > 0].min(), .012), AMAX
q = np.clip(np.round(np.log(np.clip(f.a, lo, hi) / lo) / np.log(hi / lo) * 31), 0, 31).astype(int)
f.a = lo * (hi / lo) ** (q / 31)

def bilerp(g, x, y):
    fx = x / (W - 1) * (GRID - 1); fy = y / (H - 1) * (GRID - 1)
    ix = np.minimum(fx.astype(int), GRID - 2); iy = np.minimum(fy.astype(int), GRID - 2)
    ax = (fx - ix)[:, None]; ay = (fy - iy)[:, None]
    return (g[iy, ix] * (1 - ax) + g[iy, ix + 1] * ax) * (1 - ay) + (g[iy + 1, ix] * (1 - ax) + g[iy + 1, ix + 1] * ax) * ay

cols = bilerp(grid, f.x.astype(float), f.y.astype(float))
glow = np.stack([render(f, cols[:, c]) for c in range(3)], 2)

# heat: each dot also carries a gold-white core, g cubed, so narrower than its glow by
# root 3. Solved against what the glow leaves unexplained in green, the channel that
# separates a hot bead from a red one.
WHITE = np.array([0., 1., .28], np.float32)
ROOT = np.sqrt(float(os.environ.get('POWER', 3.)))
def core(h, chan=1.):
    C = np.zeros((H, W), np.float32)
    for i, sig in enumerate(SCALES):
        m = (f.k == i) & (h > 0)
        if not m.any(): continue
        D = np.zeros((H, W), np.float32)
        np.add.at(D, (f.y[m], f.x[m]), h[m])
        sc = sig / ROOT
        C += gauss(D, sc) * (2 * np.pi * sc * sc)
    return C * chan
h = np.zeros(len(f.a))
tiny = SCALES[f.k] <= 4.3
for _ in range(24):
    Rg = d[..., 1] - np.clip(glow[..., 1] + core(h), 0, 1)
    Rg[(d[..., 1] >= .97) & (glow[..., 1] + core(h) >= d[..., 1])] = 0
    for i, sig in enumerate(SCALES):
        m = (f.k == i) & tiny
        if m.any():
            h[m] += .5 * 2 * gauss(Rg, sig / ROOT)[f.y[m], f.x[m]]
    h = np.clip(h, 0, 3.)
hq = np.clip(np.round(np.sqrt(h / 3.) * 7), 0, 7).astype(int)
h = (hq / 7) ** 2 * 3.
sim = np.clip(glow + core(h)[..., None] * WHITE, 0, 1)
np.save(OUT.replace('.json', '.npy'), sim.astype(np.float32))

# 10-bit grid over the 1254px source; build.py orders and packs these
px = np.round(f.x / (W - 1) * 1023).astype(int); py = np.round(f.y / (H - 1) * 1023).astype(int)
energy = (f.a + h) * np.minimum(SCALES[f.k], 2.6) ** 2  # beads before bloom in the first flight
json.dump({'n': len(px), 'grid': GRID, 'lo': float(lo), 'hi': float(hi), 'scales': SCALES.tolist(),
           'pal': base64.b64encode(np.round(grid * 255).astype(np.uint8).tobytes()).decode(),
           'x': px.tolist(), 'y': py.tolist(), 'k': f.k.tolist(), 'q': q.tolist(), 'h': hq.tolist(),
           'e': np.round(energy, 3).tolist()}, open(OUT, 'w'))

from skimage.metrics import structural_similarity as ssim
crop = (150, 1100, 150, 1030)
a = np.clip(d, 0, 1)[crop[0]:crop[1], crop[2]:crop[3]]; b = sim[crop[0]:crop[1], crop[2]:crop[3]]
rm = [np.sqrt(((a[..., c]-b[..., c])**2).mean())*255 for c in range(3)]
ca = np.clip(d, 0, 1)[250:700, 760:960]; cb = sim[250:700, 760:960]
lap = lambda x: float(ndi.laplace(x.mean(2)).var())
mask = (gauss(V, 3) < .1) & (gauss(V, 25) > .02)
hp = lambda x: gauss(np.minimum(x, .3), 5.) - gauss(np.minimum(x, .3), 24.)
lump = float(hp(sim.max(2))[mask].std() / hp(d.max(2))[mask].std())
print(f'lumps={lump:.2f}x', end='  ')
print(f'crown ssim={ssim(ca, cb, channel_axis=2, data_range=1):.4f} sharp={lap(cb)/lap(ca):.3f} g={ca[...,1].mean():.4f}/{cb[...,1].mean():.4f}', end='  ')
print(f'rgb={rm[0]:.1f}/{rm[1]:.1f}/{rm[2]:.1f} N={len(px)} grid={GRID} rmse={np.sqrt(((a-b)**2).mean())*255:.2f}/255 ssim={ssim(a,b,channel_axis=2,data_range=1):.4f} per-scale={np.bincount(f.k, minlength=len(SCALES)).tolist()} hot={(hq>0).sum()} lo={lo:.3f}')
