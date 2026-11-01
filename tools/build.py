"""Pack the solved dots into the page.

Dots ship in two inline scripts. The first carries the renderer and the dots that matter
most and is sized to land in the first TCP flight, about 14KB. The second streams in
behind it, so the figure starts one round trip sooner and still ends up complete.

Run from the repo root: edit tools/template.html, then python3 tools/build.py to write index.html.
To re-solve the dots against the logo first: python3 tools/fit.py 6000 16 tools/dots.json
"""
import base64, gzip, json, os, sys
import numpy as np

src = 'tools/dots.json'
out = '.'
FIRST_FLIGHT = 13600
os.makedirs(out, exist_ok=True)
d = json.load(open(src))
x, y, k, q, h, e = (np.array(d[n]) for n in 'x y k q h e'.split())

def pack(i):
    o = i[np.lexsort((x[i], y[i]))]
    dy = np.diff(y[o], prepend=0)
    planes = np.concatenate([dy.astype(np.uint8), (x[o] & 255).astype(np.uint8),
                             ((x[o] >> 8) | (k[o] << 2)).astype(np.uint8), (q[o] | (h[o] << 5)).astype(np.uint8)])
    return base64.b64encode(planes.tobytes()).decode()

t = open('tools/template.html').read()
for a, b in (('__TOTAL__', str(d['n'])), ('__GRID__', str(d['grid'])), ('__LO__', repr(round(d['lo'], 5))),
             ('__HI__', repr(round(d['hi'], 5))), ('__SIGMA__', json.dumps(d['scales'])), ('__PAL__', d['pal'])):
    t = t.replace(a, b)

rank = np.argsort(-e)
def page(n1):
    a, b = rank[:n1], rank[n1:]
    return t.replace('__N1__', str(n1)).replace('__D1__', pack(a)).replace('__N2__', str(len(b))).replace('__D2__', pack(b))

def first_size(n1):
    h = page(n1)
    return len(gzip.compress(h[:h.index('<script>dots(')].encode(), 6))

lo, hi = 200, d['n']
while hi - lo > 20:
    mid = (lo + hi) // 2
    lo, hi = (mid, hi) if first_size(mid) <= FIRST_FLIGHT else (lo, mid)
html = page(lo)
open(out + '/index.html', 'w').write(html)
print('first script dots', lo, 'gz up to it', first_size(lo), 'whole page raw', len(html), 'gz', len(gzip.compress(html.encode(), 6)))
