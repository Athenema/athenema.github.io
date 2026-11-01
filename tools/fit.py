"""Fit a multi-scale dot field to the hero original by analysis-by-synthesis.

Dots are Gaussians, peak-normalised, summed additively in display space, which
is exactly what the WebGL page draws. Greedy matching pursuit places each dot
where the unexplained energy is largest, at whichever scale explains it best,
then amplitudes are jointly refined. Large scales become the neon bloom, small
scales the beads and stipple, and clipping makes the white-hot cores.
"""
import base64, json, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

SRC = 'assets/logo-v3-original.png'
N, GRID = int(sys.argv[1]), int(sys.argv[2])
OUT = sys.argv[3] if len(sys.argv) > 3 else 'tools/dots.json'
SCALES = np.array([.8, 1.15, 1.7, 2.6, 4.2, 7.5, 13.])
AMAX = 2.5
CAP = np.where(SCALES > 4, .3, AMAX)  # big dots are bloom: never let one read as a blotch
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
w = (V < .8) * V
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

T = V.astype(np.float32)

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

sat = T >= .97
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
                f.a[m] += step * 2 * gauss(R, sig)[f.y[m], f.x[m]]
        f.a = np.clip(f.a, 0, CAP[f.k])

f = Field()
K = max(N // 45, 40)
while len(f.a) < N * 1.08:
    R = residual(f)
    F = np.stack([gauss(R, s) for s in SCALES])
    S = F * SCALES[:, None, None] ** .8  # below 1 favours small dots: detail over blur
    best = S.argmax(0); score = S.max(0)
    pk = peak_local_max(score, min_distance=2, threshold_abs=1e-4, num_peaks=K, exclude_border=False)
    if not len(pk): break
    y, x = pk[:, 0], pk[:, 1]
    k = best[y, x]
    f.add(x, y, k, np.clip(1.6 * F[k, y, x], 0, CAP[k]))
    refine(f, 2)
refine(f, 14)
o = np.argsort(-f.a * SCALES[f.k])[:N]          # weakest contribution goes first
f.keep(np.sort(o))
refine(f, 10)


# quantise exactly as the page will decode
lo, hi = max(f.a[f.a > 0].min(), .012), AMAX
q = np.clip(np.round(np.log(np.clip(f.a, lo, hi) / lo) / np.log(hi / lo) * 63), 0, 63).astype(int)
f.a = lo * (hi / lo) ** (q / 63)

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
ROOT = np.sqrt(3.)
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
hq = np.clip(np.round(np.sqrt(h / 3.) * 15), 0, 15).astype(int)
h = (hq / 15) ** 2 * 3.
sim = np.clip(glow + core(h)[..., None] * WHITE, 0, 1)
np.save('tools/sim.npy', sim.astype(np.float32))

o = np.lexsort((f.x, f.y))
x, y, k, q, hq = f.x[o], f.y[o], f.k[o], q[o], hq[o]
# 10-bit grid over the 1254px source
px = np.round(x / (W - 1) * 1023).astype(int); py = np.round(y / (H - 1) * 1023).astype(int)
o2 = np.lexsort((px, py)); px, py, k, q, hq = px[o2], py[o2], k[o2], q[o2], hq[o2]
dy = np.diff(py, prepend=0)
planes = np.concatenate([dy.astype(np.uint8), (px & 255).astype(np.uint8), ((px >> 8) | (k << 2)).astype(np.uint8), q.astype(np.uint8), hq.astype(np.uint8)])
json.dump({'n': len(px), 'grid': GRID, 'lo': float(lo), 'hi': float(hi), 'scales': SCALES.tolist(),
           'pal': base64.b64encode(np.round(grid * 255).astype(np.uint8).tobytes()).decode(),
           'data': base64.b64encode(planes.tobytes()).decode()}, open(OUT, 'w'))

from skimage.metrics import structural_similarity as ssim
crop = (150, 1100, 150, 1030)
a = np.clip(d, 0, 1)[crop[0]:crop[1], crop[2]:crop[3]]; b = sim[crop[0]:crop[1], crop[2]:crop[3]]
rm = [np.sqrt(((a[..., c]-b[..., c])**2).mean())*255 for c in range(3)]
print(f'rgb={rm[0]:.1f}/{rm[1]:.1f}/{rm[2]:.1f} N={len(px)} grid={GRID} rmse={np.sqrt(((a-b)**2).mean())*255:.2f}/255 ssim={ssim(a,b,channel_axis=2,data_range=1):.4f} per-scale={np.bincount(k, minlength=len(SCALES)).tolist()} hot={(hq>0).sum()} lo={lo:.3f}')
