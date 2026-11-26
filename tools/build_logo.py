"""Newark Coworks logo: the House of Prayer and the Plume House, drawn from the Broad Street
photo (IMG_5567.jpeg in the Newark Coworks Dropbox folder).

The outline of each building, its colours and its features come from the photo. The drawing
is made at the photo's full size in three parts:

1. The church. Cut out along its real outline; shadows lifted; every wall laid out stone by
   stone in the wall's own perspective (courses run to the vanishing points measured from the
   photo), each stone taking its tone from the photo; windows, louvres, mouldings and roofs
   kept from the photo and traced in pencil.
2. The Plume House. Poles, signs, a wire and a tree stand in front of it in the photo, so it is
   rebuilt in the photo's perspective: coursed stonework on both walls (tones sampled from the
   clean part of the real wall), the real windows with their shutters copied from the two clean
   ones, the real cornice repeated along the front, a clean roof, and no cast shadows. The
   entrance porch and the far attic window are reconstructed from the little that shows.
3. The two are set on one ground line, the house up against the church, with the tip of the
   steeple at the centre of the picture.

Usage: python3 -I tools/build_logo.py <IMG_5567.jpeg> <site_dir> [--files <dir>] [--debug <dir>]
Needs Pillow, NumPy and OpenCV with the contrib modules (opencv-contrib-python, for ximgproc).
Coordinates written as "half-res" are pixels of the upright photo at half size (2142 x 2856).
"""
import sys, os
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageFont

SRC, SITE = sys.argv[1], sys.argv[2]
opt = lambda flag: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None
FILES, DEBUG = opt("--files"), opt("--debug")
for d in (FILES, DEBUG):
    if d:
        os.makedirs(d, exist_ok=True)

PHOTO = cv2.cvtColor(np.asarray(ImageOps.exif_transpose(Image.open(SRC)).convert("RGB")), cv2.COLOR_RGB2BGR)
assert PHOTO.shape[:2] == (5712, 4284), PHOTO.shape


# ================================================================ perspective
S = 2.0                                               # full-res pixels per half-res unit
# Vanishing points, surveyed from the photo (course directions of each wall, window jambs, roof lines)
VV = np.array([1196.0, -4145.0]) * S                  # verticals of both buildings (far above)
C_FRONT = np.array([-1947.0, 2061.0]) * S             # church: horizontals of the faces towards Broad Street
C_SIDE = np.array([2853.0, 2158.0]) * S               # church: horizontals of the nave side
C_DIAG = np.array([19219.0, 2488.0]) * S              # church: horizontals of the spire face between the two
P_FRONT = np.array([-9280.0, 2227.0]) * S             # Plume House: horizontals of the front
P_SIDE = np.array([2097.0, 2141.0]) * S               # Plume House: horizontals of the gable end


class Plane:
    def __init__(self, A, V, ref):
        l = np.cross(np.r_[A, 1.0], np.r_[V, 1.0]); self.l = l / l[2]
        M = np.stack([A / np.linalg.norm(A), V / np.linalg.norm(V)], 1)
        self.M, self.Mi = M, np.linalg.inv(M)
        self.k = np.ones(2); self.o = np.zeros(2)
        ref = np.asarray(ref, float)
        ua = (A - ref) / np.linalg.norm(A - ref); uv = (V - ref) / np.linalg.norm(V - ref)
        p0 = self.to_plane(ref[None])[0]
        ds = self.to_plane((ref + ua)[None])[0] - p0
        dt = self.to_plane((ref + uv)[None])[0] - p0
        # one plane unit = one image pixel at the reference point; s grows to the right, t grows downwards
        self.k = np.array([np.sign(ds[0]) * np.sign(ua[0]) / abs(ds[0]), np.sign(dt[1]) * np.sign(uv[1]) / abs(dt[1])])
        self.o = self.to_plane(ref[None])[0]             # the reference point is the origin

    def to_plane(self, xy):
        xy = np.asarray(xy, np.float64)
        w = self.l[0] * xy[..., 0] + self.l[1] * xy[..., 1] + 1.0
        q = np.stack([xy[..., 0] / w, xy[..., 1] / w], -1) @ self.Mi.T
        return q * self.k - self.o

    def to_image(self, st):
        st = (np.asarray(st, np.float64) + self.o) / self.k
        q = st @ self.M.T
        w = 1.0 / (1.0 - self.l[0] * q[..., 0] - self.l[1] * q[..., 1])
        return q * w[..., None]

    def grid(self, x0, y0, x1, y1):
        """Plane coordinates of every pixel of an image window (float32 s, t)."""
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        st = self.to_plane(np.stack([xs, ys], -1))
        return st[..., 0].astype(np.float32), st[..., 1].astype(np.float32)

    def rectify(self, img, s0, t0, s1, t1, scale=1.0, interp=cv2.INTER_CUBIC):
        """The wall seen square-on: plane window (s0..s1, t0..t1) sampled from the image."""
        ss, tt = np.meshgrid(np.arange(s0, s1, 1 / scale), np.arange(t0, t1, 1 / scale))
        xy = self.to_image(np.stack([ss, tt], -1)).astype(np.float32)
        return cv2.remap(img, xy[..., 0], xy[..., 1], interp, borderMode=cv2.BORDER_REFLECT)


# ================================================================ the church
X0, Y0, X1, Y1 = 1760, 740, 3990, 4210        # the church's window in the photo (full-res pixels)
GROUND = 4200                                 # the ground line both buildings stand on
h, w = Y1 - Y0, X1 - X0


def P(pts):
    """Half-res photo coordinates -> pixel coordinates inside the window."""
    return np.array(pts, np.float64) * S - [X0, Y0]


def poly(pts):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(P(pts)).astype(np.int32)], 1)
    return m > 0


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


STORE = {}
def save(name, arr): STORE[name] = arr
def load(name): return STORE[name]


# ------------------------------------------------------------------ outline and clean plate
def plate():
    img = PHOTO[Y0:Y1, X0:X1].copy()
    b = img.astype(np.int32)
    sky = ((b[..., 0] - b[..., 2]) > 45) & (b[..., 0] > 95) & ((b[..., 0] - b[..., 1]) > 15)
    sky = cv2.morphologyEx(sky.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab = cv2.connectedComponents(sky)
    sky = np.isin(lab, list(set(np.unique(lab[0, :])) | set(np.unique(lab[:, -1])) - {0}))
    # the sky gives the top and right of the church; the tower's left edge (against a tree) and the
    # far end of the nave are drawn by hand; the bottom is the ground line
    outline = [(950, 380), (1400, 380), (1968, 1250), (1968, 2100), (915, 2100), (916, 2076), (921, 1970), (939, 1925),
               (944, 1757), (955, 1730), (960, 1620), (967, 1596), (974, 1440), (976, 1405), (986, 1356), (950, 1342)]
    mask = poly(outline) & ~sky
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    mask = lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    yy, xx = np.mgrid[0:h, 0:w]
    hy, hx = (yy + Y0) / S, (xx + X0) / S                     # half-res photo coordinates of each pixel

    def reflect(v, lo, hi):
        span = hi - lo
        t = np.mod(v - lo, 2 * span)
        return lo + np.where(t > span, 2 * span - t, t)

    def blend(region, layer, r=3.0):
        nonlocal img
        k = cv2.GaussianBlur(region.astype(np.float32), (0, 0), r)[..., None]
        img = (img * (1 - k) + layer * k).astype(np.uint8)

    # parish notice board at the foot of the tower: the buttress shadow carries on behind it
    src_y = np.clip(np.round(reflect(hy, 1966, 2004) * S - Y0).astype(int), 0, h - 1)
    blend(poly([(961, 2006), (1010, 2006), (1010, 2104), (961, 2104)]), img[src_y, xx], 4.0)
    # the shadow of a street lamp and its arm falls on the tower: cover it with the wall just below
    def shifted(dx, dy):
        return np.roll(img, (-int(dy * S), -int(dx * S)), (0, 1))
    blend(poly([(1008, 1785), (1030, 1789), (1053, 1804), (1053, 1819), (1034, 1817), (1018, 1806), (1008, 1800)]), shifted(4, 30), 3.0)
    blend(poly([(940, 1770), (982, 1770), (982, 1792), (962, 1802), (954, 1826), (940, 1826)]), shifted(3, 52), 3.0)
    # below the railings' top edge the wall is hidden; continue it down to the ground by mirroring
    cut = 2073.0 + (hx - 916) * 22 / 1052
    src_y = np.clip(np.round((2 * cut - hy) * S - Y0).astype(int), 0, h - 1)
    blend(hy > cut, img[src_y, xx], 2.0)
    save("img", img); save("mask", mask)


# ------------------------------------------------------------------ light
def tone():
    img, mask = load("img"), load("mask")
    inside = mask.astype(np.uint8)
    _, idx = cv2.distanceTransformWithLabels(1 - inside, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(inside)
    lut = np.zeros((idx.max() + 1, 2), np.int32)
    lut[idx[ys, xs]] = np.stack([ys, xs], 1)
    work = np.where(mask[..., None], img, img[lut[idx][..., 0], lut[idx][..., 1]])     # building colours carried outwards

    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] / 255
    Lm = cv2.medianBlur((L * 255).astype(np.uint8), 15).astype(np.float32) / 255
    sh = cv2.GaussianBlur(1 - smoothstep(0.16, 0.26, Lm), (0, 0), 2.4)                 # 1 in shadow, 0 in sunlight
    C, G = 0.20, 2.2
    L2 = L * (1 - sh) + (C + (1 - C) * (1 - (1 - L) ** G)) * sh                        # raise the black point of the shadows
    gain = (L2 + 0.04) / (L + 0.04)
    lin = cv2.GaussianBlur((work.astype(np.float32) / 255) ** 2.2, (0, 0), 7)
    blue = (lin[..., 0] - lin[..., 2]) / (lin.sum(2) + 1e-4)
    blue = cv2.medianBlur(np.clip((blue + 0.6) / 1.4 * 255, 0, 255).astype(np.uint8), 31).astype(np.float32) / 255 * 1.4 - 0.6
    # Slate. In open shade stone photographs blue too, so colour alone cannot tell roof from wall:
    # colour is trusted only on the main roofs; the two lean-to roofs in deep shade are drawn by hand.
    main_roofs = poly([(1262, 1405), (1400, 1325), (1967, 1250), (1967, 1808), (1700, 1695), (1614, 1700), (1625, 1758), (1560, 1738), (1470, 1745), (1470, 1700)])
    weak = ((blue > 0.26) & main_roofs).astype(np.uint8)
    n, lab_r = cv2.connectedComponents(weak)
    keep = np.zeros(n, bool); keep[np.unique(lab_r[(blue > 0.42) & main_roofs])] = True; keep[0] = False
    roof = keep[lab_r]
    roof = cv2.morphologyEx(roof.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
    roof |= poly([(1625, 1758), (1700, 1785), (1800, 1820), (1870, 1848), (1967, 1876), (1967, 1903), (1870, 1868), (1800, 1842), (1700, 1802), (1640, 1774)])   # aisle
    roof |= poly([(1570, 1820), (1600, 1810), (1700, 1856), (1800, 1903), (1852, 1926), (1852, 1952), (1740, 1920), (1728, 1905), (1575, 1834)])                 # porch
    roof &= mask
    slate = cv2.GaussianBlur(roof.astype(np.float32), (0, 0), 1.5)
    save("roof", roof)
    chroma = np.clip(1 + 0.4 * (gain - 1), 1, 2.2 - 0.7 * slate)
    A_, B_ = (lab[..., 1] - 128) * chroma, (lab[..., 2] - 128) * chroma
    stone = sh * (1 - slate)
    A_ += stone * (np.clip(A_, 2.5, 7) - A_)                                           # shaded stone: a muted version of its own colour
    B_ += stone * (np.clip(B_, 6, 13) - B_)
    lifted = cv2.cvtColor(np.clip(np.dstack([L2 * 255, A_ + 128, B_ + 128]), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    opened = cv2.cvtColor(np.clip(np.dstack([(L * (1 - sh) + (0.30 + 0.70 * L ** 0.5) * sh) * 255, A_ + 128, B_ + 128]), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    save("lifted", lifted); save("opened", opened); save("shade", sh.astype(np.float32)); save("slate", slate.astype(np.float32))
    save("Lm", Lm.astype(np.float32))
    fine = cv2.bilateralFilter(lifted, 0, 16, 3.0)                                     # photographic fill: grain off, detail kept
    wall = cv2.ximgproc.rollingGuidanceFilter(lifted, d=-1, sigmaColor=30, sigmaSpace=4.5, numOfIter=4)   # mortar lines taken out
    save("fine", fine); save("wall", wall)


# ------------------------------------------------------------------ stonework
def hash01(a, b):
    """Repeatable pseudo-random number in 0..1 for a pair of integers."""
    v = np.sin(a.astype(np.float64) * 12.9898 + b.astype(np.float64) * 78.233) * 43758.5453
    return (v - np.floor(v)).astype(np.float32)


def courses(t, heights, seed):
    """Course number for every wall coordinate t: courses of a few different heights, laid at random."""
    rs = np.random.default_rng(seed)
    lo, hi = float(t.min()) - 50, float(t.max()) + 50
    hs = rs.choice(heights, int((hi - lo) / min(heights)) + 4)
    edges = lo + np.concatenate([[0], np.cumsum(hs)])
    c = np.searchsorted(edges, t.ravel(), side="right").reshape(t.shape) - 1
    return c.astype(np.int32), edges


def bricks(s, c, edges, ratio, jitter=0.3):
    """Stone number along each course: lengths vary around ratio x the course height, joints staggered."""
    height = (edges[c + 1] - edges[c]).astype(np.float32)
    mean = height * ratio
    off = hash01(c, c * 0 + 7) * mean
    u = (s - off) / mean
    k0 = np.floor(u).astype(np.int32)
    b0 = k0 + (hash01(c, k0) - 0.5) * 2 * jitter                # this stone's left joint
    b1 = k0 + 1 + (hash01(c, k0 + 1) - 0.5) * 2 * jitter        # and the next one
    return np.where(u < b0, k0 - 1, np.where(u >= b1, k0 + 1, k0)).astype(np.int32)


def walls():
    mask, lifted, wall, fine = load("mask"), load("lifted"), load("wall"), load("fine")
    sh, roof, Lm = load("shade"), load("roof"), load("Lm")
    yy, xx = np.mgrid[0:h, 0:w]

    # --- which wall plane each pixel belongs to: 1 front (lit), 2 side (shade), 3 the spire's diagonal face
    spire_zone = poly([(900, 300), (1420, 300), (1420, 1400), (1344, 1360), (1184, 1276), (964, 1336), (900, 1345)])
    tower_side = poly([(1182, 1284), (1328, 1356), (1332, 1440), (1266, 1420), (1182, 1536)])
    right_zone = poly([(1270, 1300), (2000, 1300), (2000, 2110), (1468, 2110), (1468, 1704), (1266, 1420)])
    shade = sh > 0.5
    fam = np.ones((h, w), np.uint8)
    fam[right_zone & shade] = 2
    fam[tower_side] = 2
    Lbig = cv2.medianBlur((Lm * 255).astype(np.uint8), 41).astype(np.float32) / 255
    fam[spire_zone] = np.where(Lbig > 0.50, 1, np.where(Lbig > 0.19, 3, 2))[spire_zone]
    fam = cv2.medianBlur(fam, 31)
    fam[~mask] = 0

    # --- what is plain wall: not roof, not a window, not a moulding or other feature
    small = cv2.resize(wall, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    lab_s = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    med = np.dstack([cv2.medianBlur(lab_s[..., i], 47) for i in range(3)]).astype(np.float32)
    dE = np.sqrt((((lab_s.astype(np.float32) - med) * [100 / 255, 1, 1]) ** 2).sum(2))
    feat = (dE > 26).astype(np.uint8)
    n, lab_f, stats, _ = cv2.connectedComponentsWithStats(feat, connectivity=8)
    big = stats[:, cv2.CC_STAT_AREA] >= 90                                # a pit or a stain in one stone is not a feature
    big[0] = False
    feature = cv2.resize(big[lab_f].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    feature = cv2.dilate(feature, np.ones((5, 5), np.uint8)) > 0
    lancet = lambda x0, x1, y_spring, y_top, y_bot: [(x0, y_bot), (x0, y_spring), ((x0 + x1) / 2, y_top), (x1, y_spring), (x1, y_bot)]
    windows = np.zeros((h, w), bool)
    for pts in (lancet(1036, 1072, 1392, 1362, 1594), lancet(1090, 1130, 1384, 1354, 1586),          # belfry, front
                lancet(1214, 1242, 1385, 1362, 1482),                                               # belfry, side
                lancet(1194, 1218, 1730, 1698, 1917), lancet(1248, 1298, 1632, 1578, 1907), lancet(1316, 1348, 1715, 1683, 1902),   # west window
                lancet(1046, 1078, 1962, 1938, 2042)):                                              # tower, ground floor
        windows |= poly(pts)
    is_wall = mask & (fam > 0) & ~cv2.dilate(roof.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~feature & ~windows
    is_wall = cv2.morphologyEx(is_wall.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0

    # --- lay the stones on each plane
    planes = {1: (Plane(C_FRONT, VV, (2600, 3700)), (19, 22, 25), 1.55, 11),
              2: (Plane(C_SIDE, VV, (3500, 3900)), (19, 22, 25), 1.25, 12),
              3: (Plane(C_DIAG, VV, (2340, 1800)), (19, 22, 25), 1.75, 13)}
    ident = np.zeros((h, w), np.int64)
    for f, (pl, heights, ratio, seed) in planes.items():
        sel = is_wall & (fam == f)
        if not sel.any(): continue
        ys, xs = np.nonzero(sel)
        st = pl.to_plane(np.stack([xs + X0, ys + Y0], -1).astype(np.float64))
        c, edges = courses(st[:, 1], heights, seed)
        k = bricks(st[:, 0].astype(np.float32), c, edges, ratio)
        ident[ys, xs] = (f * 2 + shade[ys, xs]) * 10 ** 12 + (c.astype(np.int64) + 5000) * 10 ** 6 + (k.astype(np.int64) + 500000)

    # --- one tone per stone, from the photograph, with some of its shading kept
    ys, xs = np.nonzero(ident)
    uniq, inv = np.unique(ident[ys, xs], return_inverse=True)
    src = wall[ys, xs].astype(np.float64)
    cnt = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    mean = np.stack([np.bincount(inv, src[:, i], len(uniq)) / cnt for i in range(3)], 1)
    vary = 1 + (hash01(uniq % 9973, uniq // 9973 % 7919) - 0.5) * 0.07          # no two stones quite alike
    stone = mean[inv] * vary[inv, None]
    fill = fine.astype(np.float32)
    fill[ys, xs] = 0.62 * stone + 0.38 * src
    # joints: wherever the stone number changes
    j = np.zeros((h, w), np.float32)
    j[:, :-1] = np.maximum(j[:, :-1], ((ident[:, :-1] != ident[:, 1:]) & (ident[:, :-1] > 0) & (ident[:, 1:] > 0)))
    j[:-1, :] = np.maximum(j[:-1, :], ((ident[:-1, :] != ident[1:, :]) & (ident[:-1, :] > 0) & (ident[1:, :] > 0)))
    save("fill", np.clip(fill, 0, 255).astype(np.uint8)); save("joint", j); save("fam", fam); save("is_wall", is_wall)


# ------------------------------------------------------------------ pencil
def valleys(gray, sigma, lo, hi, k=1.6):
    """Thin dark lines only: a difference of Gaussians that ignores flat dark areas."""
    d = cv2.GaussianBlur(gray, (0, 0), sigma * k) - cv2.GaussianBlur(gray, (0, 0), sigma)
    return smoothstep(lo, hi, d)


def drop_specks(ink, min_len):
    b = (ink > 0.35).astype(np.uint8)
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    ok = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= min_len
    ok[0] = False
    return ink * cv2.dilate(ok[lab_].astype(np.uint8), np.ones((3, 3), np.uint8))


def pencil(ink, rng, weight=1.0):
    """Graphite: a soft-edged line broken up by the tooth of the paper, pressure easing on and off."""
    tooth = cv2.GaussianBlur(rng.normal(0, 1, ink.shape).astype(np.float32), (0, 0), 1.4)
    tooth = 0.5 + 0.5 * np.tanh(tooth * 2.6)
    press = cv2.GaussianBlur(rng.normal(0, 1, ink.shape).astype(np.float32), (0, 0), 30)
    press = 0.82 + 0.18 * np.tanh(press * 30)
    return np.clip(cv2.GaussianBlur(ink, (0, 0), 0.9) * (0.55 + 0.45 * tooth) * press * weight, 0, 1)


def apply_pencil(fill, ink):
    """Pencil over colour: the graphite darkens what is under it rather than covering it."""
    g = np.array([70, 64, 62], np.float32) / 255
    out = fill.astype(np.float32) * (1 - ink[..., None] * 0.92 * (1 - g * 0.55))
    return out * (1 - ink[..., None] * 0.25) + g * 255 * ink[..., None] * 0.25


def draw():
    mask, fill, j, wall, opened, is_wall = load("mask"), load("fill"), load("joint"), load("wall"), load("opened"), load("is_wall")
    rng = np.random.default_rng(7)
    # the joints between the stones
    joints = cv2.dilate(j, np.ones((2, 2), np.uint8))
    # everything else: windows, louvres, mouldings, roof edges, the cross
    calm = cv2.ximgproc.rollingGuidanceFilter(opened, d=-1, sigmaColor=34, sigmaSpace=5.0, numOfIter=4)
    blab = cv2.cvtColor(cv2.GaussianBlur(calm, (0, 0), 1.6), cv2.COLOR_BGR2LAB)
    trace = np.zeros(mask.shape, np.uint8)
    for c, (lo, hi) in enumerate(((30, 75), (12, 28), (12, 28))):
        trace |= cv2.Canny(blab[..., c], lo, hi, L2gradient=True)
    inner = cv2.erode(is_wall.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0       # inside plain wall the joints do the drawing
    trace[inner] = 0
    trace = drop_specks(trace.astype(np.float32) / 255, 22)
    trace = cv2.dilate(trace, np.ones((2, 2), np.uint8))
    gray = cv2.cvtColor(calm, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    seams = drop_specks(valleys(gray, 2.2, 0.012, 0.04), 18)
    seams[inner] = 0
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
    outline = cv2.dilate(edge, np.ones((4, 4), np.uint8))
    ink = np.maximum.reduce([pencil(joints, rng, 0.62), pencil(trace, rng, 0.85), pencil(seams, rng, 0.7), pencil(outline, rng, 0.95)])
    out = apply_pencil(fill, ink)
    alpha = np.clip((cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2) - 0.25) / 0.5, 0, 1)
    rgba = np.dstack([np.clip(out, 0, 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
    return rgba


# ================================================================ the Plume House
def plume_house():
    """The Plume House, rebuilt stone by stone in the photo's own perspective. Returns RGBA and its place in the photo."""
    PX0, PY0, PX1, PY1 = 250, 3290, 1830, 4210                 # the house's window in the photo
    h, w = PY1 - PY0, PX1 - PX0
    SS = 3                                                 # supersampling for hand-drawn parts

    PF = Plane(P_FRONT, VV, (840, 3880))                   # front wall
    PS = Plane(P_SIDE, VV, (1540, 3900))                   # gable end
    V_SLOPE = np.array([1514.0, 628.0]) * S                # where the roof's rakes meet
    PR = Plane(P_FRONT, V_SLOPE, (900, 3500))              # front roof slope

    yy, xx = np.mgrid[0:h, 0:w]
    pix = np.stack([xx + PX0, yy + PY0], -1).astype(np.float64)
    sF, tF = [c.astype(np.float32) for c in np.moveaxis(PF.to_plane(pix), -1, 0)]
    sS, tS = [c.astype(np.float32) for c in np.moveaxis(PS.to_plane(pix), -1, 0)]
    sR, tR = [c.astype(np.float32) for c in np.moveaxis(PR.to_plane(pix), -1, 0)]


    def smoothstep(a_, b_, x):
        t = np.clip((x - a_) / (b_ - a_), 0, 1)
        return t * t * (3 - 2 * t)


    def hash01(p, q):
        v = np.sin(np.asarray(p, np.float64) * 12.9898 + np.asarray(q, np.float64) * 78.233) * 43758.5453
        return (v - np.floor(v)).astype(np.float32)


    def img_poly(pts_full):
        """Anti-aliased mask (0..1) of a polygon given in full-res photo coordinates."""
        m = np.zeros((h * SS, w * SS), np.uint8)
        p = (np.array(pts_full, np.float64) - [PX0, PY0]) * SS
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 255)
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


    def plane_poly(plane, pts_st):
        return img_poly(plane.to_image(np.array(pts_st, np.float64)))


    def H(pts):                                            # half-res photo coordinates -> full-res
        return [(x * S, y * S) for x, y in pts]


    # ------------------------------------------------------------------ colours, sampled from the photo
    rect = PF.rectify(PHOTO, -640, -340, 600, 400)             # the front wall seen square-on; origin (-640, -340)
    RO = np.array([-640, -340])


    def rsub(s0, t0, s1, t1):
        return rect[t0 - RO[1]:t1 - RO[1], s0 - RO[0]:s1 - RO[0]]


    clean = [rsub(236, -195, 268, -70), rsub(90, -46, 470, -8), rsub(424, -195, 486, 95)]
    px = np.concatenate([c.reshape(-1, 3) for c in clean]).astype(np.float32)
    lum = px @ np.array([0.114, 0.587, 0.299], np.float32)
    MORTAR = px[lum > np.percentile(lum, 93)].mean(0)
    palette = []
    for c in clean:                                        # tones of single stones: small tiles that hold no mortar
        for y in range(0, c.shape[0] - 7, 4):
            for x in range(0, c.shape[1] - 9, 5):
                tile = c[y:y + 7, x:x + 9].reshape(-1, 3).astype(np.float32)
                tl = tile @ np.array([0.114, 0.587, 0.299], np.float32)
                if tl.max() - tl.min() < 26:
                    palette.append(tile.mean(0))
    palette = np.array(palette, np.float32)
    pl = palette @ np.array([0.114, 0.587, 0.299], np.float32)
    palette = palette[(pl > np.percentile(pl, 8)) & (pl < np.percentile(pl, 97))]
    ROOF = np.array([80, 88, 88], np.float32)
    RIDGE = np.array([126, 140, 150], np.float32)
    WHITE = np.array([222, 236, 241], np.float32)          # the painted woodwork in sunlight


    # ------------------------------------------------------------------ stonework
    def lay(s, t, heights, ratio, seed, mortar_w=1.5, jitter=0.3):
        """Coursed stonework on a wall: returns stone number, and how far each point is from the nearest joint."""
        rs = np.random.default_rng(seed)
        lo, hi = float(t.min()) - 60, float(t.max()) + 60
        hs = rs.choice(heights, int((hi - lo) / min(heights)) + 4)
        edges = (lo + np.concatenate([[0], np.cumsum(hs)])).astype(np.float32)
        c = (np.searchsorted(edges, t.ravel(), side="right").reshape(t.shape) - 1).astype(np.int32)
        height = edges[c + 1] - edges[c]
        mean = height * ratio * (0.85 + 0.3 * hash01(c, c * 0 + 3))
        off = hash01(c, c * 0 + 7) * mean
        u = (s - off) / mean
        k0 = np.floor(u).astype(np.int32)
        b0 = k0 + (hash01(c, k0) - 0.5) * 2 * jitter
        b1 = k0 + 1 + (hash01(c, k0 + 1) - 0.5) * 2 * jitter
        k = np.where(u < b0, k0 - 1, np.where(u >= b1, k0 + 1, k0)).astype(np.int32)
        left = k + (hash01(c, k) - 0.5) * 2 * jitter
        right = k + 1 + (hash01(c, k + 1) - 0.5) * 2 * jitter
        d = np.minimum.reduce([t - edges[c], edges[c + 1] - t, (u - left) * mean, (right - u) * mean])
        return c, k, d


    def stone_wall(s, t, seed, tone=1.0):
        c, k, d = lay(s, t, (20.0, 22.0, 24.0), 1.55, seed)
        pick = (hash01(c * 7 + 1, k * 13 + 5) * (len(palette) - 1)).astype(np.int32)
        col = palette[pick] * (0.94 + 0.12 * hash01(c * 3 + 2, k * 5 + 1))[..., None]
        rs = np.random.default_rng(seed + 100)
        grain = cv2.GaussianBlur(rs.normal(0, 1, s.shape).astype(np.float32), (0, 0), 1.6)
        col = col * (1 + 0.035 * grain / grain.std())[..., None]
        m = 1 - smoothstep(0.9, 2.3, d)                    # mortar, about three pixels wide
        col = col * (1 - m[..., None]) + MORTAR * m[..., None]
        joint = 1 - smoothstep(0.0, 0.9, d)                # centre line of the joint, for the pencil
        return col * tone, joint


    # ------------------------------------------------------------------ the layers, back to front
    canvas = np.zeros((h, w, 3), np.float32)
    cover = np.zeros((h, w), np.float32)
    ink_joint = np.zeros((h, w), np.float32)
    ink_line = np.zeros((h, w), np.float32)                # edges of the drawn parts


    def put(mask, colour):
        nonlocal canvas, cover
        m = mask[..., None]
        canvas = canvas * (1 - m) + colour * m
        cover = np.maximum(cover, mask)


    def edge_of(mask):
        e = cv2.morphologyEx((mask > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        return e.astype(np.float32)


    # chimney, behind the ridge
    ch_front = img_poly(H([(322, 1674), (352, 1674), (352, 1712), (322, 1712)]))
    ch_side = img_poly(H([(352, 1675), (392, 1681), (392, 1712), (352, 1712)]))
    put(ch_side, np.array([150, 158, 162], np.float32))
    put(ch_front, np.array([200, 221, 229], np.float32))
    ink_line = np.maximum.reduce([ink_line, edge_of(ch_front), edge_of(ch_side)])

    # roof: one clean slope, courses of shingles running to the same vanishing point as the eaves
    roof = img_poly(H([(160, 1823), (287, 1710), (776, 1682), (779, 1689), (693, 1803), (520, 1809)]))
    course = np.floor(tR / 9.0)
    shade = 1 + 0.03 * (hash01(course, course * 0 + 1) - 0.5) + 0.04 * (tR - tR[roof > 0.5].mean()) / np.ptp(tR[roof > 0.5])
    rs = np.random.default_rng(5)
    grain = cv2.GaussianBlur(rs.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.2)
    put(roof, ROOF * (shade * (1 + 0.02 * grain / grain.std()))[..., None])
    fr = tR / 9.0 - course
    ink_joint = np.maximum(ink_joint, (1 - smoothstep(0.0, 0.12, np.minimum(fr, 1 - fr))) * (roof > 0.5) * 0.55)
    ridge = img_poly(H([(285, 1707), (776, 1679), (779, 1689), (287, 1716)]))
    put(ridge, RIDGE)
    ink_line = np.maximum.reduce([ink_line, edge_of(roof), edge_of(ridge)])

    # gable end: stonework up to the rakes, two attic windows
    G = [(-205, 318), (-205, -206), (-170, -264), (-21, -539), (55, -595), (242, -264), (242, 318)]      # runs a little under the front wall
    gable = plane_poly(PS, G)
    col, joint = stone_wall(sS, tS, seed=21, tone=0.84)
    put(gable, col)
    ink_joint = np.maximum(ink_joint, joint * (gable > 0.5))
    ink_line = np.maximum(ink_line, edge_of(gable))
    for s0 in (-77, 80):                                   # attic windows: white frame, two sashes of six panes
        frame = plane_poly(PS, [(s0, -396), (s0 + 60, -396), (s0 + 60, -300), (s0, -300)])
        glass = plane_poly(PS, [(s0 + 8, -388), (s0 + 52, -388), (s0 + 52, -308), (s0 + 8, -308)])
        put(frame, WHITE * 0.9)
        put(glass, np.array([118, 112, 104], np.float32))
        bars = np.zeros((h, w), np.float32)
        for sv in (s0 + 22.7, s0 + 37.3):
            bars = np.maximum(bars, plane_poly(PS, [(sv - 1.3, -388), (sv + 1.3, -388), (sv + 1.3, -308), (sv - 1.3, -308)]))
        for tv, half in ((-368, 1.2), (-348, 2.6), (-328, 1.2)):
            bars = np.maximum(bars, plane_poly(PS, [(s0 + 8, tv - half), (s0 + 52, tv - half), (s0 + 52, tv + half), (s0 + 8, tv + half)]))
        put(bars, WHITE * 0.9)
        sill = plane_poly(PS, [(s0 - 5, -300), (s0 + 65, -300), (s0 + 65, -291), (s0 - 5, -291)])
        put(sill, palette.mean(0) * 0.72)
        ink_line = np.maximum.reduce([ink_line, edge_of(frame), edge_of(glass), edge_of(sill)])
    # rake boards along the gable
    for p0, p1 in (((-176, -252), (-21, -539)), ((-21, -539), (55, -595)), ((55, -595), (248, -252))):
        d = np.array(p1, float) - p0; n = np.array([-d[1], d[0]]) / np.hypot(*d) * 7.0
        if n[1] < 0: n = -n                                # the board hangs below the roof edge
        board = plane_poly(PS, [p0, p1, tuple(np.array(p1) + n), tuple(np.array(p0) + n)])
        put(board, WHITE * 0.93)
        ink_line = np.maximum(ink_line, edge_of(board))

    # front wall: stonework from the cornice to the ground
    WALL = [(-527, -215), (489, -215), (489, 322), (-527, 322)]
    front = plane_poly(PF, WALL)
    col, joint = stone_wall(sF, tF, seed=11)
    put(front, col)
    ink_joint = np.maximum(ink_joint, joint * (front > 0.5))
    ink_line = np.maximum(ink_line, edge_of(front))


    def paste(src, s0, t0, centre, scale, mask_fn):
        """Put a piece of the square-on photo (origin s0, t0) on the wall, centred at `centre`, scaled."""
        sh_, sw_ = src.shape[:2]
        cs, ct = s0 + sw_ / 2, t0 + sh_ / 2
        u = ((sF - centre[0]) / scale + cs - s0).astype(np.float32)
        v = ((tF - centre[1]) / scale + ct - t0).astype(np.float32)
        samp = cv2.remap(src, u, v, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        m = mask_fn((sF - centre[0]) / scale + cs, (tF - centre[1]) / scale + ct)
        put(m, samp)
        return m


    def box(s, t, s0, t0, s1, t1, soft=0.8):
        return smoothstep(-soft, soft, np.minimum.reduce([s - s0, s1 - s, t - t0, t1 - t]))


    win_detail = []
    # upper windows: the clean fourth window, with its shutters and sill, in all five bays
    U = rsub(80, -212, 240, -46).copy()
    def upper_mask(s, t):
        return np.maximum(box(s, t, 86, -207, 233, -61), box(s, t, 108, -62, 217, -50))
    for c_, k_ in ((-396, 1.0), (-222, 1.0), (-20, 1.0), (158.5, 1.0), (342, 1.0)):
        m = paste(U, 80, -212, (c_, -129), k_, upper_mask)
        ink_line = np.maximum(ink_line, edge_of(m)); win_detail.append(m)

    # lower windows: the clean fifth window, with its lintel, in the four bays beside the door
    Lw = rsub(250, -8, 430, 232).copy()
    def lower_mask(s, t):
        lint = smoothstep(-0.8, 0.8, np.minimum.reduce([t + 4, 26 - t, s - (254 + (t + 4) * 0.28), (403 - (t + 4) * 0.25) - s]))
        return np.maximum(box(s, t, 268, 26, 413, 229), lint)
    for c_, k_ in ((-397.5, 1.0), (-222, 1.0), (157.5, 1.0), (340, 1.0)):
        m = paste(Lw, 250, -8, (c_, 112), k_, lower_mask)
        sill = plane_poly(PF, [(c_ - 56 * k_, 112 + 117 * k_), (c_ + 56 * k_, 112 + 117 * k_), (c_ + 56 * k_, 112 + 128 * k_), (c_ - 56 * k_, 112 + 128 * k_)])
        put(sill, palette.mean(0) * 0.8)
        ink_line = np.maximum.reduce([ink_line, edge_of(m), edge_of(sill)]); win_detail.append(m)

    detail = np.maximum.reduce(win_detail)                 # where the photo's own detail gets traced

    # cornice: a clean stretch of the real one (fourteen brackets), repeated along the whole front
    strip = rsub(120, -268, 448, -196).copy()
    u = np.mod(sF - 120, 328.0).astype(np.float32)
    v = (tF + 268).astype(np.float32)
    samp = cv2.remap(np.concatenate([strip, strip[:, :2]], 1), u, v, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    cornice = box(sF, tF, -560, -263, 512, -201, 0.6)
    put(cornice, samp)
    detail = np.maximum(detail, cornice)
    ink_line = np.maximum(ink_line, edge_of(cornice))
    ret = plane_poly(PS, [(-192, -268), (-128, -268), (-128, -251), (-152, -237), (-192, -237)])     # its return on the gable end
    put(ret, WHITE * 0.9)
    ink_line = np.maximum(ink_line, edge_of(ret))


    # ------------------------------------------------------------------ entrance porch (reconstructed)
    # The photo shows only pieces of it between the traffic signal and the signs: the left end of a
    # pediment with its rake, a column with its capital, and the white frame of the doorway behind.
    # It is drawn square-on in wall coordinates and then set into the perspective. The porch stands
    # about a metre and a half in front of the wall, so its front is drawn shifted to the left, as the
    # camera (well to the right of the door) sees it.
    D0 = np.array([-250.0, -80.0]); DW, DH, DS = 380, 410, 3
    dcol = np.zeros((DH * DS, DW * DS, 3), np.float32)
    dalpha = np.zeros((DH * DS, DW * DS), np.uint8)
    dline = np.zeros((DH * DS, DW * DS), np.uint8)


    def dpoly(pts, colour, outline=True):
        p = np.round((np.array(pts, np.float64) - D0) * DS).astype(np.int32)
        cv2.fillPoly(dcol, [p], [float(c) for c in colour], cv2.LINE_AA)
        cv2.fillPoly(dalpha, [p], 255, cv2.LINE_AA)
        if outline:
            cv2.polylines(dline, [p], True, 255, DS, cv2.LINE_AA)


    def drect(s0, t0, s1, t1, colour, outline=True):
        dpoly([(s0, t0), (s1, t0), (s1, t1), (s0, t1)], colour, outline)


    CD, SHIFT = -22.0, -78.0                               # door centre on the wall; sideways shift of the porch front
    STONE = np.array([156, 170, 178], np.float32)
    GLASS = np.array([118, 112, 104], np.float32)
    LEAF = np.array([58, 62, 66], np.float32)
    # the doorway in the wall: pilasters, lintel, sidelights, a six-panel door
    drect(CD - 92, 28, CD + 92, 250, WHITE * 0.96)
    for sgn in (-1, 1):
        x0, x1 = sorted((CD + sgn * 92, CD + sgn * 76))
        drect(x0, 28, x1, 250, WHITE)
        x0, x1 = sorted((CD + sgn * 70, CD + sgn * 44))
        drect(x0, 58, x1, 196, GLASS)
        for tv in (92.5, 127, 161.5):
            drect(x0, tv - 1.2, x1, tv + 1.2, WHITE, False)
        drect(x0 + 3, 204, x1 - 3, 244, WHITE * 0.93)
    drect(CD - 92, 28, CD + 92, 48, WHITE)
    drect(CD - 34, 54, CD + 34, 250, LEAF)
    for px0 in (CD - 28, CD + 4):
        for t0_, t1_ in ((62, 104), (112, 170), (178, 242)):
            drect(px0, t0_, px0 + 24, t1_, LEAF * 1.28)
    # porch floor and steps
    F0, F1 = CD - 92 + SHIFT - 8, CD + 92 + SHIFT + 8      # front edge of the floor
    drect(F1, 250, CD + 100, 322, STONE * 0.86)            # its side, running back to the wall
    drect(F0, 250, F1, 322, STONE)
    for tv in (274, 298):
        cv2.line(dline, tuple(np.round((np.array([F0, tv]) - D0) * DS).astype(int)), tuple(np.round((np.array([F1, tv]) - D0) * DS).astype(int)), 255, DS, cv2.LINE_AA)
    # the porch itself: side beam and roof slope running back to the wall, then columns, beam and pediment in front
    B0, B1 = CD - 86 + SHIFT, CD + 86 + SHIFT              # front beam
    drect(B1, 0, CD + 98, 24, WHITE * 0.88)
    apex_f, tip_f = (CD + SHIFT, -53), (B1 + 14, 0)
    dpoly([apex_f, tip_f, (tip_f[0] - SHIFT, 0), (apex_f[0] - SHIFT, -53)], ROOF * 1.05)
    for cx in (B0 + 16, B1 - 16):
        drect(cx - 12, 24, cx + 12, 250, WHITE)
        drect(cx - 15, 24, cx + 15, 34, WHITE)
        drect(cx - 15, 242, cx + 15, 250, WHITE)
    drect(B0, 0, B1, 24, WHITE)
    dpoly([(B0 - 14, 0), (B1 + 14, 0), apex_f], WHITE)
    dpoly([(B0 + 12, -6), (B1 - 12, -6), (apex_f[0], apex_f[1] + 7)], WHITE * 0.93)

    u = ((sF - D0[0]) * DS).astype(np.float32); v = ((tF - D0[1]) * DS).astype(np.float32)
    k = cv2.GaussianBlur(dalpha, (0, 0), DS * 0.5)
    pa = cv2.remap(k, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32) / 255
    pc = cv2.remap(cv2.GaussianBlur(dcol, (0, 0), DS * 0.5), u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    pl = cv2.remap(cv2.GaussianBlur(dline, (0, 0), DS * 0.35), u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32) / 255
    put(pa, pc)
    ink_joint = ink_joint * (1 - pa)
    ink_line = np.maximum(ink_line * (1 - pa), np.clip(pl * 1.6, 0, 1))

    # ------------------------------------------------------------------ pencil and cut-out
    rng = np.random.default_rng(9)


    def pencil(ink, weight=1.0, width=0.9):
        tooth = cv2.GaussianBlur(rng.normal(0, 1, ink.shape).astype(np.float32), (0, 0), 1.4)
        tooth = 0.5 + 0.5 * np.tanh(tooth * 2.6)
        press = cv2.GaussianBlur(rng.normal(0, 1, ink.shape).astype(np.float32), (0, 0), 30)
        press = 0.82 + 0.18 * np.tanh(press * 30)
        return np.clip(cv2.GaussianBlur(ink, (0, 0), width) * (0.55 + 0.45 * tooth) * press * weight, 0, 1)


    ground = smoothstep(-0.6, 0.6, GROUND - (yy + PY0)).astype(np.float32)
    alpha = np.clip(cover, 0, 1) * ground
    solid = alpha > 0.5
    base = np.clip(canvas, 0, 255).astype(np.uint8)
    # the photo's own detail (window bars, shutter slats, cornice brackets), traced where it was pasted in
    calm = cv2.bilateralFilter(base, 0, 26, 2.2)
    blab = cv2.cvtColor(cv2.GaussianBlur(calm, (0, 0), 1.0), cv2.COLOR_BGR2LAB)
    trace = np.zeros((h, w), np.uint8)
    for c_, (lo, hi) in enumerate(((34, 80), (12, 28), (12, 28))):
        trace |= cv2.Canny(blab[..., c_], lo, hi, L2gradient=True)
    trace = trace.astype(np.float32) / 255 * (detail > 0.5)
    outline = cv2.dilate(cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32), np.ones((4, 4), np.uint8))
    ink = np.maximum.reduce([pencil(ink_joint, 0.5, 0.7), pencil(cv2.dilate(ink_line, np.ones((2, 2), np.uint8)), 0.85), pencil(cv2.dilate(trace, np.ones((2, 2), np.uint8)), 0.6), pencil(outline, 0.95)])
    g = np.array([70, 64, 62], np.float32) / 255
    out = canvas * (1 - ink[..., None] * 0.92 * (1 - g * 0.55))
    out = out * (1 - ink[..., None] * 0.25) + g * 255 * ink[..., None] * 0.25
    rgba = np.dstack([np.clip(out, 0, 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
    return rgba, (PX0, PY0)


# ================================================================ compose
plate(); tone(); walls()
church = draw()
house, (HX, HY) = plume_house()
MEET = 110                                             # the house moves this far right to stand against the tower
ca = church[..., 3] > 128
ys, xs = np.nonzero(ca)
tip_x = int(round(xs[ys < ys.min() + 12].mean())) + X0
top = ys.min() + Y0
left = np.nonzero((house[..., 3] > 128).any(0))[0].min() + HX + MEET
right = xs.max() + X0
half = max(tip_x - left, right - tip_x) + 24           # the steeple tip is the centre of the picture
MX0, MY0 = tip_x - half, top - 24
MW, MH = 2 * half, GROUND + 6 - MY0
mark = np.zeros((MH, MW, 4), np.float32)


def over(layer, lx, ly):
    hh, ww = layer.shape[:2]
    ax, ay = lx - MX0, ly - MY0
    sx0, sy0, sx1, sy1 = max(0, -ax), max(0, -ay), min(ww, MW - ax), min(hh, MH - ay)
    src = layer[sy0:sy1, sx0:sx1].astype(np.float32)
    dst = mark[ay + sy0:ay + sy1, ax + sx0:ax + sx1]
    k = src[..., 3:4] / 255
    dst[..., :3] = src[..., :3] * k + dst[..., :3] * (1 - k)
    dst[..., 3:4] = 255 * k + dst[..., 3:4] * (1 - k)


over(house, HX + MEET, HY)
over(church, X0, Y0)
drawing = Image.fromarray(cv2.cvtColor(np.clip(mark, 0, 255).astype(np.uint8), cv2.COLOR_BGRA2RGBA))


# ================================================================ exports
NIGHT, BLUE, WHITE_RGB = (13, 27, 64), (29, 67, 166), (255, 255, 255)


def scaled(im, height):
    """Resize with the colours premultiplied, so the transparent edge stays clean."""
    w_ = round(im.width * height / im.height)
    return im.convert("RGBa").resize((w_, round(height)), Image.LANCZOS).convert("RGBA")


def tile(size, radius, ground=BLUE, over_=4):
    """App-icon style square: the drawing on the site's blue, steeple in the middle."""
    S_ = size * over_
    im = Image.new("RGBA", (S_, S_), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 0, S_ - 1, S_ - 1], radius * over_, fill=ground + (255,))
    art = scaled(drawing, S_ * 0.78)
    im.alpha_composite(art, ((S_ - art.width) // 2, (S_ - art.height) // 2))
    return im.resize((size, size), Image.LANCZOS)


def name_font(px):
    f = ImageFont.truetype(os.path.join(SITE, "fonts", "besley-latin-wght-normal.woff2"), round(px))
    f.set_variation_by_axes([800])
    return f


def stacked(mark_h, ink, ground=None, margin=0):
    """The drawing with the name centred under the steeple."""
    art = scaled(drawing, mark_h)
    font = name_font(mark_h * 0.105)                    # the name comes out a little narrower than the drawing
    l, t, r, b = font.getbbox("Newark Coworks", anchor="ls")
    gap = round(mark_h * 0.075)
    W_, H_ = max(art.width, r - l) + 2 * margin, art.height + gap - t + b + 2 * margin
    im = Image.new("RGBA", (W_, H_), (ground + (255,)) if ground else (0, 0, 0, 0))
    im.alpha_composite(art, ((W_ - art.width) // 2, margin))
    ImageDraw.Draw(im).text(((W_ - (r - l)) // 2 - l, margin + art.height + gap - t), "Newark Coworks", font=font, fill=ink + (255,), anchor="ls")
    return im


def beside(mark_h, ink):
    """The drawing with the name beside it, standing on the same ground line."""
    art = scaled(drawing.crop(drawing.getbbox()), mark_h)
    font = name_font(mark_h * 0.29)
    l, t, r, b = font.getbbox("Newark Coworks", anchor="ls")
    gap = round(mark_h * 0.11)
    im = Image.new("RGBA", (art.width + gap + (r - l), art.height), (0, 0, 0, 0))
    im.alpha_composite(art, (0, 0))
    ImageDraw.Draw(im).text((art.width + gap - l, art.height - round(mark_h * 0.012) - 2), "Newark Coworks", font=font, fill=ink + (255,), anchor="ls")
    return im


# --- the site
scaled(drawing, 336).save(os.path.join(SITE, "images", "logo-mark.webp"), quality=92, method=6)
tile(32, 7).save(os.path.join(SITE, "favicon-32.png"), optimize=True)
tile(192, 40).save(os.path.join(SITE, "favicon-192.png"), optimize=True)
tile(180, 0).save(os.path.join(SITE, "apple-touch-icon.png"), optimize=True)

# --- logo files to hand out
if FILES:
    scaled(drawing, 2400).save(os.path.join(FILES, "newark-coworks-logo-mark.png"), optimize=True)
    stacked(1500, NIGHT).save(os.path.join(FILES, "newark-coworks-logo.png"), optimize=True)
    beside(1000, NIGHT).save(os.path.join(FILES, "newark-coworks-logo-wide.png"), optimize=True)
    sq = stacked(840, NIGHT)
    assert sq.width <= 1100 and sq.height <= 1100, sq.size
    card = Image.new("RGBA", (1200, 1200), WHITE_RGB + (255,))
    card.alpha_composite(sq, ((1200 - sq.width) // 2, (1200 - sq.height) // 2))
    card.convert("RGB").save(os.path.join(FILES, "newark-coworks-logo-square.png"), optimize=True)

if DEBUG:
    bg = Image.new("RGBA", drawing.size, (250, 250, 250, 255)); bg.alpha_composite(drawing)
    bg.convert("RGB").resize((drawing.width // 3, drawing.height // 3), Image.LANCZOS).save(os.path.join(DEBUG, "drawing.jpg"), quality=93)
    bg.convert("RGB").save(os.path.join(DEBUG, "drawing-full.png"))
print("logo written:", drawing.size, "| steeple tip at x =", tip_x - MX0, "of", MW)
