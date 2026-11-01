"""Score a browser capture of the resting figure against the original logo.

Both are reduced to emitted light on black and aligned on the page's own mapping,
so the number measures the live render, not the fit's simulation.
usage: fidelity.py capture.png left top size [side-by-side.png]
"""
import sys
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

SRC = 'assets/logo-v3-original.png'
CX, CY, SPAN = 578.5, 627.5, 877
path, left, top, size = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])

cap = np.asarray(Image.open(path).convert('RGB')).astype(np.float32) / 255
o = np.asarray(Image.open(SRC).convert('RGB')).astype(np.float32) / 255
o = np.clip(o - np.median(o.reshape(-1, 3), 0), 0, 1)

H, W = cap.shape[:2]
# source pixels covered by the capture
scale = SPAN / size
x0, y0 = CX + (0 - left - size / 2) * scale, CY + (0 - top - size / 2) * scale
x1, y1 = x0 + W * scale, y0 + H * scale
src = Image.fromarray((o * 255).astype(np.uint8)).transform((W, H), Image.EXTENT, (x0, y0, x1, y1), Image.BICUBIC)
src = np.asarray(src).astype(np.float32) / 255

rm = [float(np.sqrt(((src[..., c] - cap[..., c]) ** 2).mean()) * 255) for c in range(3)]
s = ssim(src, cap, channel_axis=2, data_range=1)
print(f'rgb rmse {rm[0]:.2f}/{rm[1]:.2f}/{rm[2]:.2f}  rmse {np.sqrt(((src - cap) ** 2).mean()) * 255:.2f}  ssim {s:.4f}')
if len(sys.argv) > 5:
    Image.fromarray((np.concatenate([src, cap], 1) * 255).astype(np.uint8)).save(sys.argv[5])
