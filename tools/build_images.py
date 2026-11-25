"""Build web-ready photos for newarkcoworks.com from the Dropbox originals.

Usage: python3 -I build_images.py <source_dir> <out_dir>
Crops are fractions of the original frame (x0, y0, x1, y1).
An optional fifth value in a crop rotates the frame first (degrees, clockwise) to level it.
A soft blur covers a colleague's face in the hero shot.
"""
import sys, os
from PIL import Image, ImageFilter, ImageDraw, ImageOps

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

# name: (source file, crop, [blurs], [output widths])
SPEC = {
    "hero":       ("IMG_5573.jpeg", (0.00, 0.00, 1.00, 0.80),
                   [((0.556, 0.328, 0.583, 0.376), 16, "ellipse")],  # colleague's face
                   [1800, 900]),
    "window-desk": ("IMG_5568.jpeg", (0.30, 0.12, 0.97, 0.80, 2.2), [], [1600, 800]),
    "meeting-table": ("IMG_5569.jpeg", (0.155, 0.06, 0.90, 0.92), [], [1600, 800]),
    "conference-room": ("IMG_5576.jpeg", (0.00, 0.04, 0.96, 0.96), [], [1600, 800]),
    "nook":       ("IMG_5578.jpeg", (0.262, 0.215, 0.770, 0.74), [], [1200, 600]),   # table centred
    "snacks":     ("IMG_5572.jpeg", (0.474, 0.588, 0.885, 0.900), [], [1400, 700]),
    "reception":  ("IMG_5574.jpeg", (0.20, 0.02, 1.00, 0.92), [], [1600, 800]),
    "mascot":     ("IMG_5575.jpeg", (0.00, 0.00, 0.60, 1.00), [], [1200, 600]),
    "broad-street": ("IMG_5567.jpeg", (0.00, 0.06, 1.00, 0.86), [], [1400, 700]),
}

os.makedirs(OUT, exist_ok=True)
for name, (fn, crop, blurs, widths) in SPEC.items():
    im = ImageOps.exif_transpose(Image.open(os.path.join(SRC, fn))).convert("RGB")
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
