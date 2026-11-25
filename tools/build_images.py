"""Build web-ready photos for newarkcoworks.com from the Dropbox originals.

Usage: python3 -I build_images.py <source_dir> <out_dir>
Crops are fractions of the original frame (x0, y0, x1, y1).
An optional fifth value in a crop rotates the frame first (degrees, clockwise) to level it.
A soft blur covers a colleague's face in the hero shot.
The cafe-table photo has two waste bins standing against the left chair; remove_bins() paints
them out (wall, baseboard and carpet are rebuilt from the same photo) so both chairs fit in frame.
Needs Pillow, NumPy and SciPy.
"""
import sys, os
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageOps
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import binary_dilation, gaussian_filter, label

SRC, OUT = sys.argv[1], sys.argv[2]

def soft_blur(im, box, radius, shape="rect"):
    """Blur a region given in image fractions, feathered so no edge shows."""
    W, H = im.size
    x0, y0, x1, y1 = [int(v) for v in (box[0]*W, box[1]*H, box[2]*W, box[3]*H)]
    blurred = im.filter(ImageFilter.GaussianBlur(radius))
    mask = Image.new("L", im.size, 0)
    d = ImageDraw.Draw(mask)
    (d.ellipse if shape == "ellipse" else d.rectangle)((x0, y0, x1, y1), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(6, radius // 2)))
    return Image.composite(blurred, im, mask)

# ---- Cafe-table photo (IMG_5578): paint out the bins. Coordinates are pixels in the upright 4284x5712 frame.

def _carpet_line(x):
    """Row where the carpet starts, measured along the wall."""
    return 3486 + 0.055 * (x - 1195)


def _leg_left_edge(y):
    """Left edge of the left chair's rear leg, measured against the bin liner and the wall."""
    pts = [(3036, 1142), (3100, 1134), (3120, 1129), (3140, 1132), (3200, 1144), (3260, 1155),
           (3300, 1165), (3340, 1177), (3372, 1186), (3400, 1196), (3450, 1213), (3486, 1224), (3540, 1242)]
    ys, xs = zip(*pts)
    return np.interp(y, ys, xs)


# Baseboard shading relative to the wall above it, by rows above the carpet line (measured at x=1440-1480).
_BASEBOARD = [(-125, (0, 0, 0)), (-111, (6, 4, 0)), (-101, (13, 11, 1)), (-91, (39, 32, 17)), (-81, (31, 24, 12)),
              (-71, (21, 15, 4)), (-61, (13, 7, -2)), (-51, (2, -5, -15)), (-11, (2, -4, -13)), (0, (0, -8, -16))]


def remove_bins(im):
    a = np.asarray(im).astype(np.float64)
    H, W, _ = a.shape
    lum = a.mean(2)
    yy, xx = np.mgrid[0:H, 0:W]
    cline = _carpet_line(xx)
    L = _leg_left_edge(yy)

    # 1. where the bins and their liners are
    right = np.where(yy < 3372, 1176, L - 1)
    P1 = (xx >= 940) & (xx < right) & (yy >= 3036) & (yy < cline)          # left of the chair leg
    seat = np.interp(yy, [3190, 3200, 3210, 3220, 3240, 3260, 3290], [1300, 1262, 1250, 1239, 1226, 1217, 1219])
    P2 = (xx >= 1150) & (xx < seat + 2) & (yy >= 3075) & (yy < 3290)        # liner showing right of the leg
    carpet_fix = (yy >= cline) & (xx >= 940) & (((xx < 1178) & (yy < 3612)) | ((xx < 1140) & (yy < 3800)))

    # chair ironwork to keep: dark and neutral (the liner's pink edge is dark but red)
    dark = (lum < 78) & (a[..., 0] - a[..., 1] < 12)
    iron = (P1 & dark & (xx >= L - 1) & (yy < 3372)) | (P2 & dark & (xx >= L - 1))
    lab, n = label(iron)
    if n:
        sizes = np.bincount(lab.ravel())
        iron &= sizes[lab] >= 40
    replace_wall = (P1 | P2) & ~iron

    # 2. wall: smooth (harmonic) fill from the real wall around the patch
    x0, x1, y0, y1 = 940, 1340, 2990, 3545
    win = np.zeros((H, W), bool)
    win[y0:y1, x0:x1] = True
    wall_zone = win & (yy < cline)
    unknown = wall_zone & (P1 | P2 | binary_dilation((lum < 78) & wall_zone, iterations=2))
    known = wall_zone & ~unknown
    uy, ux = np.nonzero(unknown)
    idx = -np.ones((H, W), int)
    idx[uy, ux] = np.arange(len(uy))
    n = len(uy)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], []
    rhs, deg = np.zeros((n, 3)), np.zeros(n)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        ny, nx = uy + dy, ux + dx
        isu, isk = unknown[ny, nx], known[ny, nx]
        deg += isu + isk
        rows.append(np.nonzero(isu)[0]); cols.append(idx[ny[isu], nx[isu]]); vals.append(-np.ones(isu.sum()))
        rhs[isk] += a[ny[isk], nx[isk]]
    deg[deg == 0] = 1
    A = csr_matrix((np.concatenate([deg] + vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    wall = a.copy()
    wall[uy, ux] = np.stack([spsolve(A, rhs[:, c]) for c in range(3)], 1)
    # baseboard: add its shading back along the bottom of the patch
    r = (yy - cline)[y0:y1, x0:x1]
    rs = [p[0] for p in _BASEBOARD]
    scale = (wall[y0:y1, x0:x1].mean(2) / 128.0)[..., None]
    wall[y0:y1, x0:x1] += np.stack([np.interp(r, rs, [p[1][c] for p in _BASEBOARD]) for c in range(3)], -1) * scale
    # grain so the patch does not look airbrushed
    rng = np.random.default_rng(5578)
    wall[y0:y1, x0:x1] += gaussian_filter(rng.normal(0, 2.6, (y1 - y0, x1 - x0)), 0.8)[..., None]

    out = a.copy()
    m = np.maximum(gaussian_filter(replace_wall.astype(float), 1.2), replace_wall)
    m[iron | ~wall_zone] = 0
    out = out * (1 - m[..., None]) + wall * m[..., None]

    # 3. carpet: clean carpet from further down the same columns, matched to the local light
    cy0, cy1, cx0, cx1, shift = 3440, 3830, 900, 1500, 330
    sl = (slice(cy0, cy1), slice(cx0, cx1))
    fix = carpet_fix[sl]
    valid = ((yy >= cline + 3) & ~carpet_fix & (xx >= 940))[sl].astype(float)
    lo = lum[sl]

    def local_light(sigma):
        wgt = gaussian_filter(valid, sigma)
        return gaussian_filter(valid * lo, sigma) / np.maximum(wgt, 1e-6), wgt
    near, wn = local_light(28)
    far, _ = local_light(110)
    target = np.where(wn > 0.08, near, far)
    src = a[cy0 + shift:cy1 + shift, cx0:cx1]
    gain = np.clip(target / gaussian_filter(src.mean(2), 28), 0.6, 1.25)[..., None]
    cm = gaussian_filter(binary_dilation(fix, iterations=4).astype(float), 5)
    cm[(yy < cline)[sl]] = 0
    out[sl] = out[sl] * (1 - cm[..., None]) + src * gain * cm[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# name: (source file, crop, [blurs], [output widths])
SPEC = {
    "hero":       ("IMG_5573.jpeg", (0.00, 0.00, 1.00, 0.80),
                   [((0.556, 0.328, 0.583, 0.376), 16, "ellipse")],  # colleague's face
                   [1800, 900]),
    "window-desk": ("IMG_5568.jpeg", (0.30, 0.12, 0.97, 0.80, 2.2), [], [1600, 800]),
    "meeting-table": ("IMG_5569.jpeg", (0.155, 0.06, 0.90, 0.92), [], [1600, 800]),
    "conference-room": ("IMG_5576.jpeg", (0.00, 0.04, 0.96, 0.96), [], [1600, 800]),
    "nook":       ("IMG_5578.jpeg", (0.2295, 0.25, 0.8595, 0.74), [], [1200, 600]),   # both chairs whole
    "snacks":     ("IMG_5572.jpeg", (0.00, 0.00, 1.00, 1.00), [], [1400, 700]),       # whole counter
    "reception":  ("IMG_5574.jpeg", (0.20, 0.02, 1.00, 0.92), [], [1600, 800]),
    "mascot":     ("IMG_5575.jpeg", (0.00, 0.00, 0.60, 1.00), [], [1200, 600]),
    "broad-street": ("IMG_5567.jpeg", (0.00, 0.06, 1.00, 0.86), [], [1400, 700]),
}
RETOUCH = {"nook": remove_bins}

os.makedirs(OUT, exist_ok=True)
for name, (fn, crop, blurs, widths) in SPEC.items():
    im = ImageOps.exif_transpose(Image.open(os.path.join(SRC, fn))).convert("RGB")
    if name in RETOUCH:
        im = RETOUCH[name](im)
    for box, radius, shape in blurs:
        im = soft_blur(im, box, radius, shape)
    if len(crop) > 4:
        im = im.rotate(-crop[4], resample=Image.BICUBIC)
    W, H = im.size
    im = im.crop((int(crop[0]*W), int(crop[1]*H), int(crop[2]*W), int(crop[3]*H)))
    for w in widths:
        h = round(im.height * w / im.width)
        out = im.resize((w, h), Image.LANCZOS)
        path = os.path.join(OUT, f"{name}-{w}.jpg")
        out.save(path, "JPEG", quality=80, optimize=True, progressive=True)
        print(f"{name}-{w}.jpg  {w}x{h}  {os.path.getsize(path)//1024} KB")

# Social share card (1200x630) from the hero frame
im = Image.open(os.path.join(OUT, "hero-1800.jpg"))
card = ImageOps.fit(im, (1200, 630), Image.LANCZOS, centering=(0.5, 0.4))
card.save(os.path.join(OUT, "share-card.jpg"), "JPEG", quality=82, optimize=True, progressive=True)
print("share-card.jpg 1200x630", os.path.getsize(os.path.join(OUT, "share-card.jpg"))//1024, "KB")
