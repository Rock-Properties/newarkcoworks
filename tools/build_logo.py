"""Newark Coworks logo: the House of Prayer and the Plume House, drawn from the Broad Street
photo (IMG_5567.jpeg in the Newark Coworks Dropbox folder).

1. Clean plate. The two buildings are cut out along their real outline, and what stood in
   front of them (lamp post, traffic signal, street signs, a wire, a weeping tree, the parish
   notice board) is painted out with the building's own stone, roof and windows.
2. Drawing. The shadows are lifted, each surface is filled with its real colour as a flat
   tone, and the windows, doors and other features are traced in pencil.
3. Exports for the site (mark, favicons) and the logo files.

Usage: python3 -I tools/build_logo.py <IMG_5567.jpeg> <site_dir> [--files <dir>] [--debug <dir>]
Needs Pillow, NumPy and opencv-contrib (ximgproc). Coordinates are pixels in the half-size
upright frame of the photo (2142 x 2856).
"""
import sys, os
import numpy as np
import cv2
from PIL import Image, ImageOps

SRC, SITE = sys.argv[1], sys.argv[2]
opt = lambda flag: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None
FILES, DEBUG = opt("--files"), opt("--debug")
for d in (FILES, DEBUG):
    if d:
        os.makedirs(d, exist_ok=True)

full = ImageOps.exif_transpose(Image.open(SRC)).convert("RGB")
W, H = full.size[0] // 2, full.size[1] // 2
rgb = np.asarray(full.resize((W, H), Image.LANCZOS)).copy()


def poly(points, shape=(H, W)):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.array(points, np.float64)).astype(np.int32)], 255)
    return m > 0


def rect(x0, y0, x1, y1):
    return poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def line(points, width):
    m = np.zeros((H, W), np.uint8)
    cv2.polylines(m, [np.round(np.array(points, np.float64)).astype(np.int32)], False, 255, width, cv2.LINE_AA)
    return m > 0


# ---------------------------------------------------------------- sky
a = rgb.astype(np.int32)
sky = ((a[..., 2] - a[..., 0]) > 45) & (a[..., 2] > 95) & ((a[..., 2] - a[..., 1]) > 15)
sky = cv2.morphologyEx(sky.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
n, lab = cv2.connectedComponents(sky)
sky = np.isin(lab, list(set(np.unique(lab[0, :])) - {0}))

# ---------------------------------------------------------------- outlines
# House of Prayer: the sky gives the top and right edges; the tower's left edge (against the
# tree), the far end of the nave and the ground line above the fence are drawn by hand.
P_CHURCH = [(950, 380), (1400, 380), (1968, 1250), (1968, 2098), (916, 2076), (921, 1970), (939, 1925),
            (944, 1757), (955, 1730), (960, 1620), (967, 1596), (974, 1440), (976, 1405), (986, 1356), (950, 1342)]
# Plume House: roof verge, ridge, chimney, gable end and walls, all by hand (a clock tower,
# a lamp post and trees stand behind and in front of it).
P_PLUME = [(160, 2057), (180, 1842), (155, 1840), (152, 1822), (175, 1812), (285, 1710), (320, 1708), (320, 1672),
           (362, 1672), (362, 1680), (392, 1682), (392, 1703), (772, 1680), (812, 1674), (858, 1775), (834, 2057)]
church = poly(P_CHURCH) & ~sky
plume = poly(P_PLUME) & ~sky
church = cv2.morphologyEx(church.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
# keep the main body of each building only
for name in ("church", "plume"):
    m = locals()[name].astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    locals()[name][:] = lab == keep
mask = church | plume

# ---------------------------------------------------------------- clean plate
plate = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
yy, xx = np.indices((H, W))


def soften(m, r=2.0):
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), r)[..., None]


def put(m, layer, r=2.0):
    """Blend `layer` into the plate where mask m is set."""
    global plate
    k = soften(m, r)
    plate = (plate * (1 - k) + layer * k).astype(np.uint8)


def shifted(dx, dy):
    """The plate moved so that each pixel shows what is (dx, dy) away from it."""
    return np.roll(plate, (-dy, -dx), (0, 1))


def tiled(y0, y1, x0, x1):
    t = plate[y0:y1, x0:x1]
    t = np.concatenate([t, t[:, ::-1]], 1)
    t = np.concatenate([t, t[::-1]], 0)
    return np.tile(t, (H // t.shape[0] + 2, W // t.shape[1] + 2, 1))[:H, :W]


def grow(m, n):
    return cv2.dilate(m.astype(np.uint8), np.ones((n, n), np.uint8)) > 0


def clone(dst_xy, src_rect, feather=2):
    """Copy a rectangle of the plate to a new place, with a soft edge."""
    x0, y0, x1, y1 = src_rect
    dx, dy = dst_xy
    patch = plate[y0:y1, x0:x1].astype(np.float32)
    h, w = patch.shape[:2]
    m = np.zeros((h, w), np.float32)
    m[feather:h - feather, feather:w - feather] = 1
    m = cv2.GaussianBlur(m, (0, 0), feather)[..., None]
    region = plate[dy:dy + h, dx:dx + w].astype(np.float32)
    plate[dy:dy + h, dx:dx + w] = (region * (1 - m) + patch * m).astype(np.uint8)


def reflect(v, lo, hi):
    """Fold coordinates into [lo, hi] by mirroring, so a clean patch continues without a seam."""
    span = hi - lo
    t = np.mod(v - lo, 2 * span)
    return lo + np.where(t > span, 2 * span - t, t)


def quilt(region, src_rect, patch=(22, 24), lap=7, period=8, seed=3):
    """Cover a region with stonework: small pieces of a clean stretch of the same wall, picked at
    random, kept on the same courses and feathered into each other."""
    x0, y0, x1, y1 = src_rect
    pw, ph = patch
    rs = np.random.default_rng(seed)
    wy = np.minimum(np.arange(ph) + 1, np.arange(ph)[::-1] + 1).clip(max=lap) / lap
    wx = np.minimum(np.arange(pw) + 1, np.arange(pw)[::-1] + 1).clip(max=lap) / lap
    win = np.outer(wy, wx).astype(np.float32)
    acc = np.zeros((H, W, 3), np.float32)
    tot = np.zeros((H, W), np.float32)
    ys, xs = np.nonzero(region)
    for dy in range(ys.min() - ph // 2, ys.max() + 1, ph - lap):
        rows = [y for y in range(y0, y1 - ph + 1) if (y - dy) % period == 0]
        for dx in range(xs.min() - pw // 2, xs.max() + 1, pw - lap):
            sy, sx = rs.choice(rows), rs.integers(x0, x1 - pw + 1)
            acc[dy:dy + ph, dx:dx + pw] += plate[sy:sy + ph, sx:sx + pw] * win[..., None]
            tot[dy:dy + ph, dx:dx + pw] += win
    return acc / np.maximum(tot, 1e-3)[..., None]


def clone_window(cx, top, scale_x, scale_y, src_rect=(467, 1832, 550, 1916), src_top=1837.5, feather=2):
    """Copy the clean upstairs window to another bay, scaled for the perspective."""
    x0, y0, x1, y1 = src_rect
    w, h = round((x1 - x0) * scale_x), round((y1 - y0) * scale_y)
    patch = cv2.resize(plate[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    dx, dy = round(cx - w / 2), round(top - (src_top - y0) * scale_y)
    m = np.zeros((h, w), np.float32)
    m[feather:h - feather, feather:w - feather] = 1
    m = cv2.GaussianBlur(m, (0, 0), feather)[..., None]
    region = plate[dy:dy + h, dx:dx + w].astype(np.float32)
    plate[dy:dy + h, dx:dx + w] = (region * (1 - m) + patch * m).astype(np.uint8)


def refit_roof(top, bottom):
    """Repaint a roof plane from its own colours: a smooth fit through the clean shingles, with a
    little of the real texture kept. Wires, shadows of poles and litter fall out as outliers."""
    global plate
    area = poly(top + bottom[::-1])
    core = cv2.erode(area.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    ys, xs = np.nonzero(core)
    u, v = (xs - xs.mean()) / 300.0, (ys - ys.mean()) / 300.0
    A = np.stack([np.ones_like(u), u, v, u * v, u * u, v * v], 1)
    vals = plate[ys, xs].astype(np.float64)
    keep = np.ones(len(xs), bool)
    for _ in range(8):
        coef, *_ = np.linalg.lstsq(A[keep], vals[keep], rcond=None)
        r = np.abs(vals - A @ coef).max(1)
        keep = r < max(6.0, 1.6 * np.median(r))
    U, V = (xx - xs.mean()) / 300.0, (yy - ys.mean()) / 300.0
    fit = sum(c[None, None, :] * b[..., None] for c, b in zip(coef, (np.ones_like(U), U, V, U * V, U * U, V * V)))
    tex = cv2.medianBlur(plate, 5).astype(np.float64) - fit
    tex = cv2.GaussianBlur(np.clip(tex, -4, 4), (0, 0), 1.0) * 0.35
    put(area, np.clip(fit + tex, 0, 255), 1.2)


# --- Plume House. A lamp post, a signal pole and arm, road signs, a wire and a weeping tree
# stand in front of it. Roof and cornice are patched from the stretch beside each pole; the
# wall is patched with its own stonework; hidden windows are copied from their neighbours.
lamp = poly([(455, 1620), (482, 1620), (468, 1842), (435, 1842)])
put(grow(lamp, 5), shifted(48, -2))
sigpole = poly([(394, 1774), (424, 1774), (419, 1842), (389, 1842)])
put(grow(sigpole, 5), shifted(-38, 1))
refit_roof(top=[(290, 1720), (300, 1719), (420, 1715), (520, 1711), (600, 1703), (680, 1699), (762, 1692)],
           bottom=[(209, 1788), (178, 1815), (200, 1819), (240, 1818), (300, 1814), (420, 1805), (520, 1805),
                   (600, 1801), (680, 1796), (690, 1794)])
put(rect(338, 1803, 432, 1838), shifted(100, -4), 1.5)                                # cornice where the poles stood
arm = poly([(160, 1876), (404, 1880), (404, 1897), (160, 1894)])
put(arm, shifted(0, -21), 1.5)
put(rect(243, 1836, 254, 1876), shifted(-11, 0), 1.5)
put(rect(342, 1892, 359, 1912), shifted(0, -20), 1.5)                                  # shutter corner behind the signal

stone = tiled(1917, 1944, 474, 654)
band = poly([(361, 1839), (468, 1837), (466, 1945), (448, 1948), (446, 2068), (340, 2068), (342, 1948), (359, 1945)])
wall_fix = band | rect(176, 1894, 234, 1927) | rect(298, 1928, 346, 1954) | rect(446, 1919, 469, 1950) \
    | rect(628, 1981, 659, 2044) | rect(626, 2038, 642, 2072) | rect(338, 1912, 364, 1952) | rect(357, 1886, 364, 1914) \
    | rect(246, 1910, 286, 1954) | rect(254, 1954, 266, 2060)                          # signal, its pole's shadow
put(wall_fix, stone)
clone((263, 1953), (178, 1953, 206, 2056))        # 2nd window downstairs: left shutter was in the pole's shadow
clone_window(414.5, 1842.0, 0.96, 0.96)           # centre window upstairs, from its clean neighbour
clone_window(321.75, 1846.0, 0.917, 0.933)        # 2nd upstairs: the signal arm crossed it
clone_window(240.0, 1849.5, 0.872, 0.90)          # 1st upstairs: arm, pole and a street sign
clone((449, 1958), (447, 2004, 474, 2050))        # 4th window downstairs: top of its left shutter
clone((598, 1945), (543, 1945, 571, 2055))        # 5th window downstairs: right shutter, from the left one

gable = poly([(684, 1832), (722, 1822), (775, 1692), (812, 1682), (855, 1779), (832, 2068), (662, 2068)])
hidden = gable & (((xx > 738) & (yy > 1814)) | ((xx > 778) & (yy > 1786)) | (yy > 1925))
put(hidden, quilt(hidden, (699, 1832, 738, 1925)), 2.0)

# Front door: hidden behind the signal and signs. Painted in from what shows of it: a white
# surround with a small pediment over a panelled door.
cx, top, bot = 395, 1966, 2062
white, leaf = (224, 229, 232), (118, 128, 136)
for pts, col in (([(cx - 25, top), (cx + 25, top), (cx + 23, bot), (cx - 27, bot)], white),
                 ([(cx - 31, top + 1), (cx + 31, top + 1), (cx + 31, top - 3), (cx, top - 13), (cx - 31, top - 3)], white),
                 ([(cx - 14, top + 9), (cx + 14, top + 9), (cx + 12, bot), (cx - 16, bot)], leaf),
                 ([(cx - 10, top + 14), (cx - 2, top + 14), (cx - 3, top + 50), (cx - 11, top + 50)], (150, 160, 168)),
                 ([(cx + 2, top + 14), (cx + 10, top + 14), (cx + 9, top + 50), (cx + 1, top + 50)], (150, 160, 168)),
                 ([(cx - 12, top + 56), (cx - 4, top + 56), (cx - 5, bot - 6), (cx - 13, bot - 6)], (150, 160, 168)),
                 ([(cx, top + 56), (cx + 8, top + 56), (cx + 7, bot - 6), (cx - 1, bot - 6)], (150, 160, 168))):
    cv2.fillPoly(plate, [np.array(pts, np.int32)], col, cv2.LINE_AA)

# --- House of Prayer: only the parish notice board at the foot of the tower is in the way
put(rect(961, 2006, 1010, 2084), plate[reflect(yy, 1966, 2004), xx], 2.0)


# ================================================================ the drawing
X0, Y0, X1, Y1 = 130, 385, 1985, 2112          # the two buildings with a little room around them
plate, mask = plate[Y0:Y1, X0:X1], mask[Y0:Y1, X0:X1]
rng = np.random.default_rng(7)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# ---- extend the building colours outwards so the filters never drag sky or trees in
inside = mask.astype(np.uint8)
_, idx = cv2.distanceTransformWithLabels(1 - inside, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
ys, xs = np.nonzero(inside)
lut = np.zeros((idx.max() + 1, 2), np.int32)
lut[idx[ys, xs]] = np.stack([ys, xs], 1)
work = np.where(mask[..., None], plate, plate[lut[idx][..., 0], lut[idx][..., 1]])

# ---- lift the shadows
lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
L = lab[..., 0] / 255
# shadow = anything darker than the darkest sunlit surface; a median keeps the edge of each
# shadow crisp, so narrow shadows are lifted as much as broad ones
Lm = cv2.medianBlur((L * 255).astype(np.uint8), 7).astype(np.float32) / 255
w = cv2.GaussianBlur(1 - smoothstep(0.16, 0.26, Lm), (0, 0), 1.2)   # 1 in shadow, 0 in sunlight
C, G = 0.20, 2.2
L2 = L * (1 - w) + (C + (1 - C) * (1 - (1 - L) ** G)) * w   # raise the black point of the shadows
gain = (L2 + 0.04) / (L + 0.04)
# colour in the shadows. Open shade photographs blue; the stone is not blue, the slate roofs are.
lin = cv2.GaussianBlur((work.astype(np.float32) / 255) ** 2.2, (0, 0), 3.5)
blue = (lin[..., 0] - lin[..., 2]) / (lin.sum(2) + 1e-4)
slate = smoothstep(0.27, 0.37, blue)
chroma = np.clip(1 + 0.4 * (gain - 1), 1, 2.2 - 0.7 * slate)
A_, B_ = (lab[..., 1] - 128) * chroma, (lab[..., 2] - 128) * chroma
stone = w * (1 - slate)
A_ += stone * (np.clip(A_, 2.5, 7) - A_)                     # shaded stone: a muted version of its own colour
B_ += stone * (np.clip(B_, 6, 13) - B_)
lab2 = np.dstack([L2 * 255, A_ + 128, B_ + 128])
lifted = cv2.cvtColor(np.clip(lab2, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
# the same picture with the shadows opened right up, used only to find the features to trace
lab3 = np.dstack([(L * (1 - w) + (0.30 + 0.70 * L ** 0.5) * w) * 255, A_ + 128, B_ + 128])
opened = cv2.cvtColor(np.clip(lab3, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

# ---- colour fill: flat areas of the real colours (L0 smoothing removes the photo texture
# and leaves each surface as one tone with a clean edge)
fill = cv2.ximgproc.l0Smooth(lifted, None, 0.028, 2.0)
fill = cv2.bilateralFilter(fill, 0, 14, 2.5)
calm = cv2.ximgproc.l0Smooth(opened, None, 0.028, 2.0)


def valleys(gray, sigma, lo, hi, k=1.6):
    """Thin dark lines only: a difference of Gaussians that ignores flat dark areas."""
    d = cv2.GaussianBlur(gray, (0, 0), sigma * k) - cv2.GaussianBlur(gray, (0, 0), sigma)
    return smoothstep(lo, hi, d)


def drop_specks(ink, min_len):
    """Keep strokes, drop dots: remove line fragments shorter than min_len."""
    b = (ink > 0.35).astype(np.uint8)
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    long_enough = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= min_len
    long_enough[0] = False
    keep = cv2.dilate(long_enough[lab_].astype(np.uint8), np.ones((3, 3), np.uint8))
    return ink * keep


# ---- pencil lines
gray_s = cv2.cvtColor(calm, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
gray_f = cv2.cvtColor(cv2.bilateralFilter(opened, 0, 12, 2.0), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
# contours: where one surface meets another, traced as a line of even weight
blab = cv2.cvtColor(cv2.GaussianBlur(calm, (0, 0), 1.0), cv2.COLOR_BGR2LAB)
trace = np.zeros(mask.shape, np.uint8)
for c, (lo, hi) in enumerate(((30, 75), (12, 28), (12, 28))):
    trace |= cv2.Canny(blab[..., c], lo, hi, L2gradient=True)
trace = drop_specks(trace.astype(np.float32) / 255, 12)
trace = cv2.GaussianBlur(cv2.dilate(trace, np.ones((2, 2), np.uint8)), (0, 0), 0.5)
struct = drop_specks(valleys(gray_s, 1.2, 0.012, 0.04), 9)      # dark seams: reveals, eaves, louvres
fine = drop_specks(valleys(gray_f, 0.8, 0.010, 0.03), 7)        # stone courses, slats, tracery
ink = np.maximum.reduce([trace * 0.85, struct * 0.7, fine * 0.3])

# graphite: soft-edged, broken up by the tooth of the paper, pressure easing on and off
tooth = cv2.GaussianBlur(rng.normal(0, 1, ink.shape).astype(np.float32), (0, 0), 0.9)
tooth = 0.5 + 0.5 * np.tanh(tooth * 2.6)
press = cv2.GaussianBlur(rng.normal(0, 1, ink.shape).astype(np.float32), (0, 0), 16)
press = 0.82 + 0.18 * np.tanh(press * 30)
ink = cv2.GaussianBlur(ink, (0, 0), 0.6) * (0.5 + 0.5 * tooth) * press

# ---- the outline of the two buildings, drawn just inside their edge
edge = cv2.morphologyEx(inside, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
outline = cv2.GaussianBlur(cv2.dilate(edge, np.ones((3, 3), np.uint8)), (0, 0), 0.9) * (0.7 + 0.3 * tooth) * press
ink = np.clip(np.maximum(ink, outline * 0.95), 0, 1)

# pencil over colour: the graphite darkens what is under it rather than covering it
graphite = np.array([70, 64, 62], np.float32) / 255              # BGR
f32 = fill.astype(np.float32)
out = f32 * (1 - ink[..., None] * 0.92 * (1 - graphite * 0.55))
out = out * (1 - ink[..., None] * 0.25) + graphite * 255 * ink[..., None] * 0.25

alpha = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), 0.8)
alpha = np.clip((alpha - 0.25) / 0.5, 0, 1)
rgba = np.dstack([np.clip(out, 0, 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])


# ================================================================ exports
from PIL import ImageDraw, ImageFont

NIGHT, BLUE, WHITE = (13, 27, 64), (29, 67, 166), (255, 255, 255)
ys, xs = np.nonzero(rgba[..., 3] > 8)
drawing = Image.fromarray(cv2.cvtColor(rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1], cv2.COLOR_BGRA2RGBA))


def scaled(im, height):
    """Resize with the colours premultiplied, so the transparent edge stays clean."""
    w = round(im.width * height / im.height)
    return im.convert("RGBa").resize((w, round(height)), Image.LANCZOS).convert("RGBA")


def keyline(im, px, colour=WHITE):
    """A paper-white edge around the cut-out, so it reads on the site's blue and on dark grounds."""
    pad = int(np.ceil(px)) + 2
    a = np.zeros((im.height + 2 * pad, im.width + 2 * pad), np.uint8)
    a[pad:-pad, pad:-pad] = np.asarray(im)[..., 3]
    k = 2 * int(round(px)) + 1
    ring = cv2.GaussianBlur(cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), (0, 0), 0.7)
    out = Image.new("RGBA", (a.shape[1], a.shape[0]), colour + (0,))
    out.putalpha(Image.fromarray(ring))
    out.alpha_composite(im, (pad, pad))
    return out


def tile(size, radius, ground=BLUE, over=4):
    """App-icon style square: the drawing with its paper edge on the site's blue."""
    S = size * over
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 0, S - 1, S - 1], radius * over, fill=ground + (255,))
    art = keyline(scaled(drawing, S * 0.74), S * 0.022)
    im.alpha_composite(art, ((S - art.width) // 2, (S - art.height) // 2))
    return im.resize((size, size), Image.LANCZOS)


def lockup(mark_h, ink, edge=0.0, stacked=False, ground=None, margin=0):
    """The drawing with the name set in the site's display face (Besley, weight 800)."""
    art = scaled(drawing, mark_h)
    if edge:
        art = keyline(art, edge)
    font = ImageFont.truetype(os.path.join(SITE, "fonts", "besley-latin-wght-normal.woff2"), round(mark_h * (0.14 if stacked else 0.29)))
    font.set_variation_by_axes([800])
    l, t, r, b = font.getbbox("Newark Coworks", anchor="ls")
    if stacked:
        gap = round(mark_h * 0.085)
        W_, H_ = max(art.width, r - l) + 2 * margin, art.height + gap - t + b + 2 * margin
        im = Image.new("RGBA", (W_, H_), (ground + (255,)) if ground else (0, 0, 0, 0))
        im.alpha_composite(art, ((W_ - art.width) // 2, margin))
        ImageDraw.Draw(im).text(((W_ - (r - l)) // 2 - l, margin + art.height + gap - t), "Newark Coworks", font=font, fill=ink + (255,), anchor="ls")
    else:
        gap = round(mark_h * 0.11)
        W_, H_ = art.width + gap + (r - l) + 2 * margin, art.height + 2 * margin
        im = Image.new("RGBA", (W_, H_), (ground + (255,)) if ground else (0, 0, 0, 0))
        im.alpha_composite(art, (margin, margin))
        # the name stands on the same ground line as the buildings
        ImageDraw.Draw(im).text((margin + art.width + gap - l, margin + art.height - round(mark_h * 0.012) - (2 if not edge else int(np.ceil(edge)) + 2)), "Newark Coworks", font=font, fill=ink + (255,), anchor="ls")
    return im


# --- the site
keyline(scaled(drawing, 320), 6.0).save(os.path.join(SITE, "images", "logo-mark.webp"), quality=90, method=6)
tile(32, 7).save(os.path.join(SITE, "favicon-32.png"), optimize=True)
tile(192, 40).save(os.path.join(SITE, "favicon-192.png"), optimize=True)
tile(180, 0).save(os.path.join(SITE, "apple-touch-icon.png"), optimize=True)

# --- logo files to hand out
if FILES:
    drawing.save(os.path.join(FILES, "newark-coworks-logo-mark.png"), optimize=True)
    lockup(900, NIGHT).save(os.path.join(FILES, "newark-coworks-logo.png"), optimize=True)
    lockup(900, WHITE, edge=14).save(os.path.join(FILES, "newark-coworks-logo-for-dark-backgrounds.png"), optimize=True)
    sq = lockup(740, NIGHT, stacked=True)
    assert sq.width <= 1080, sq.size
    card = Image.new("RGBA", (1200, 1200), WHITE + (255,))
    card.alpha_composite(sq, ((1200 - sq.width) // 2, (1200 - sq.height) // 2))
    card.convert("RGB").save(os.path.join(FILES, "newark-coworks-logo-square.png"), optimize=True)

if DEBUG:
    b = np.full(out.shape, 250.0, np.float32)
    paper = (out * alpha[..., None] + b * (1 - alpha[..., None])).astype(np.uint8)
    cv2.imwrite(os.path.join(DEBUG, "drawing.jpg"), cv2.resize(paper, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 93])
    cv2.imwrite(os.path.join(DEBUG, "plate.jpg"), np.where(mask[..., None], plate, (plate * 0.25 + 190).astype(np.uint8)), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("logo written:", drawing.size)
