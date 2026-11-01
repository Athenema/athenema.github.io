"""Extract the live dot field from the hero original into compact inline data.

Beads: band-pass peaks, so bloom is ignored and each bead is one dot.
Rims: saturated lines have flat tops with no peak, so sample them greedily.
Weight: geometric mean of band-pass and absolute brightness.
Colour: a coarse grid over the figure, so dots spend no bits on it.
"""
import base64, json, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

SRC = '/Users/samdoidge/code/athenema.github.io/assets/logo-v3-original.png'
THR, MIN_DIST, RIM, RIM_GAP, GRID = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])

im = np.asarray(Image.open(SRC).convert('RGB')).astype(float)
bg = np.median(im.reshape(-1, 3), 0)
d = np.clip(im - bg, 0, None)
v = d.max(2)
s1 = ndi.gaussian_filter(v, .9)
dog = s1 - ndi.gaussian_filter(v, 4.5)
pk = peak_local_max(dog, min_distance=MIN_DIST, threshold_abs=THR, exclude_border=4)
ys, xs = list(pk[:, 0]), list(pk[:, 1])
beads = len(ys)

# second pass: greedy Poisson along saturated ridges
taken = np.zeros(v.shape, bool)
taken[pk[:, 0], pk[:, 1]] = True
near = ndi.distance_transform_edt(~taken)
cy, cx = np.nonzero((s1 > RIM) & (near >= RIM_GAP))
order = np.argsort(-s1[cy, cx])
g = int(np.ceil(RIM_GAP))
for y, x in zip(cy[order], cx[order]):
    if not taken[max(y-g, 0):y+g+1, max(x-g, 0):x+g+1].any():
        taken[y, x] = True
        ys.append(y); xs.append(x)
ys, xs = np.array(ys), np.array(xs)

amp = np.sqrt(np.maximum(dog[ys, xs], 6) * s1[ys, xs])
amp[beads:] = s1[ys[beads:], xs[beads:]] * .62  # flat-topped rims score low on band-pass

# isolated sparkles: real, but small; keep them from reading as bubbles
tree = ndi.uniform_filter(taken.astype(float), 31) * 31 * 31
amp = np.where(tree[ys, xs] < 8, np.minimum(amp, np.percentile(amp, 62)), amp)

x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
span = max(x1 - x0, y1 - y0)
ox, oy = (x0 + x1) / 2 - span / 2, (y0 + y1) / 2 - span / 2
nx = np.round((xs - ox) / span * 1023).astype(int)
ny = np.round((ys - oy) / span * 1023).astype(int)

# colour field: mean hue per cell, weighted by energy, gaps filled from nearest cell
cb = np.stack([ndi.gaussian_filter(d[..., c], 2.5) for c in range(3)], 2)
col = cb[ys, xs]; col = col / np.maximum(col.max(1, keepdims=True), 1)
gx = np.round(nx / 1023 * (GRID - 1)).astype(int)
gy = np.round(ny / 1023 * (GRID - 1)).astype(int)
field = np.zeros((GRID, GRID, 3)); wt = np.zeros((GRID, GRID))
np.add.at(field, (gy, gx), col * amp[:, None]); np.add.at(wt, (gy, gx), amp)
filled = wt > 0
field[filled] /= wt[filled][:, None]
ind = ndi.distance_transform_edt(~filled, return_distances=False, return_indices=True)
field = field[ind[0], ind[1]]
field = ndi.gaussian_filter(field, (.5, .5, 0))
field = np.clip(field - .5 * field.min(2, keepdims=True), 0, None)  # bead cores are blown out, restore saturation
field = field / np.maximum(field.max(2, keepdims=True), 1e-3)

lo, hi = np.log(amp.min()), np.log(np.percentile(amp, 99.5))
w = np.clip(np.round((np.log(amp) - lo) / (hi - lo) * 63), 0, 63).astype(int)

o = np.lexsort((nx, ny))
nx, ny, w = nx[o], ny[o], w[o]
dy = np.diff(ny, prepend=0)
planes = np.concatenate([
    dy.astype(np.uint8),
    (nx & 255).astype(np.uint8),
    ((nx >> 8) | (w << 2)).astype(np.uint8),
])
out = {
    'n': int(len(nx)), 'grid': GRID,
    'pal': base64.b64encode(np.round(field * 255).astype(np.uint8).tobytes()).decode(),
    'data': base64.b64encode(planes.tobytes()).decode(),
}
json.dump(out, open('tools/dots.json', 'w'))
print('beads', beads, 'rim', len(ys) - beads, 'total', len(ys), 'box', [int(x0), int(x1), int(y0), int(y1)])
