"""Map a photo of a real face onto a game head bitmap.

Pipeline: landmarks on photo + game face -> per-triangle warp of the photo onto the
game face's landmark layout (painter-sorted by depth so the far cheek tucks away
properly in the 3/4 view) -> skin-tone transfer onto the rest of the game head ->
feathered blend inside the face oval.
"""
import numpy as np
import cv2
from PIL import Image, ImageOps

from . import landmarks as LM

_TRIS = None

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
             54, 103, 67, 109]
# small patches on both cheeks + forehead + chin used for skin colour sampling
SKIN_SAMPLE = [50, 101, 118, 205, 280, 330, 347, 425, 151, 9, 108, 337, 199, 175]
# 3D-ish landmarks for the eye regions (used to make blink patches)
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
HOLES = [LEFT_EYE, RIGHT_EYE, INNER_LIPS]  # the mesh has holes here; we fan-fill them
N_LM = 478


def with_hole_centres(pts):
    """Append one centre point per hole (eye, eye, mouth) so they can be triangulated."""
    extra = np.array([pts[h].mean(0) for h in HOLES])
    return np.vstack([pts[:N_LM], extra])


def triangles():
    global _TRIS
    if _TRIS is None:
        from mediapipe.tasks.python.vision import FaceLandmarksConnections as F
        edges = [(c.start, c.end) for c in F.FACE_LANDMARKS_TESSELATION]
        adj = {}
        for a, b in edges:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        tris = set()
        for a, b in edges:
            for c in adj[a] & adj[b]:
                tris.add(tuple(sorted((a, b, c))))
        tris = sorted(tris)
        for hi, ring in enumerate(HOLES):
            c = N_LM + hi
            for i in range(len(ring)):
                tris.append((ring[i], ring[(i + 1) % len(ring)], c))
        _TRIS = np.array(tris, np.int32)
    return _TRIS


def load_photo(path_or_img):
    im = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    im = ImageOps.exif_transpose(im).convert("RGB")
    if max(im.size) > 2400:
        s = 2400 / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    return im


def analyse_photo(im):
    """Detect landmarks in the photo and its mirror. Returns list of candidate dicts."""
    out = []
    r = LM.detect(im)
    if r:
        out.append({"img": im, "mirrored": False, **r})
        mim = ImageOps.mirror(im)
        w = im.width
        pts = r["pts"].copy()
        pts[:, 0] = w - 1 - pts[:, 0]
        # mirroring swaps left/right landmark identities; re-detect for correct indices
        r2 = LM.detect(mim)
        if r2:
            out.append({"img": mim, "mirrored": True, **r2})
    return out


def pick_photo(candidates, target_yaw, target_pitch=0.0):
    """Choose the photo/mirror whose head pose best matches the game face."""
    best, bs = None, 1e9
    for c in candidates:
        s = abs(c["yaw"] - target_yaw) + 0.5 * abs(c["pitch"] - target_pitch)
        if s < bs:
            best, bs = c, s
    return best


def _signed_area(p):
    return (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])


def warp_mesh(src_img, src_pts, dst_pts, dst_size, dst_z=None, src_z=None):
    """Piecewise-affine warp of src_img so src_pts land on dst_pts.

    Triangles facing away in either image are skipped; the rest are painted
    back-to-front using dst_z. Returns (rgb float32 HxWx3, coverage HxW float32).
    """
    W, H = dst_size
    src = np.asarray(src_img, np.float32)
    out = np.zeros((H, W, 3), np.float32)
    cov = np.zeros((H, W), np.float32)
    tris = triangles()
    # triangles whose orientation flips between photo and game face are folded over
    # (hidden in one of the views) - drop them
    sa = np.array([_signed_area(src_pts[t, :2]) for t in tris])
    da = np.array([_signed_area(dst_pts[t, :2]) for t in tris])
    keep = (np.sign(da) == np.sign(sa)) & (np.abs(da) > 1e-3)
    keep |= tris.max(1) >= N_LM  # eye/mouth fills always drawn
    keep &= np.abs(da) > 1e-4
    order = np.arange(len(tris))
    if dst_z is not None:
        order = np.argsort(-dst_z[tris].mean(1))  # far (large z) first
    for i in order:
        if not keep[i]:
            continue
        t = tris[i]
        s = src_pts[t, :2].astype(np.float32)
        d = dst_pts[t, :2].astype(np.float32)
        x0, y0 = np.floor(d.min(0)).astype(int) - 1
        x1, y1 = np.ceil(d.max(0)).astype(int) + 2
        x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1c <= x0c or y1c <= y0c:
            continue
        sx0, sy0 = np.floor(s.min(0)).astype(int) - 2
        sx1, sy1 = np.ceil(s.max(0)).astype(int) + 3
        sx0, sy0 = max(sx0, 0), max(sy0, 0)
        sx1, sy1 = min(sx1, src.shape[1]), min(sy1, src.shape[0])
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        patch = src[sy0:sy1, sx0:sx1]
        M = cv2.getAffineTransform((s - [sx0, sy0]).astype(np.float32),
                                   (d - [x0c, y0c]).astype(np.float32))
        warped = cv2.warpAffine(patch, M, (x1c - x0c, y1c - y0c), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
        mask = np.zeros((y1c - y0c, x1c - x0c), np.uint8)
        cv2.fillConvexPoly(mask, np.round((d - [x0c, y0c]) * 16).astype(np.int32), 255,
                           lineType=cv2.LINE_AA, shift=4)
        m = mask.astype(np.float32)[..., None] / 255.0
        region = out[y0c:y1c, x0c:x1c]
        region[:] = region * (1 - m) + warped * m
        cv2.max(cov[y0c:y1c, x0c:x1c], m[..., 0], dst=cov[y0c:y1c, x0c:x1c])
    return out, cov


def _sample(img, pts, idx, radius):
    """Median colour in small discs around landmarks."""
    H, W = img.shape[:2]
    m = np.zeros((H, W), np.uint8)
    for i in idx:
        cv2.circle(m, (int(pts[i, 0]), int(pts[i, 1])), max(1, int(radius)), 255, -1)
    sel = img[m > 0]
    if len(sel) == 0:
        return np.array([128, 128, 128], np.float32)
    return np.median(sel.reshape(-1, img.shape[2])[:, :3], axis=0).astype(np.float32)


def oval_mask(pts, size, shrink=0.0, forehead_drop=0.0):
    W, H = size
    poly = pts[FACE_OVAL, :2].copy()
    c = pts[[1, 4, 5, 195]].mean(0)[:2]
    if shrink:
        poly = c + (poly - c) * (1 - shrink)
    if forehead_drop:
        # pull the top of the oval down toward the brows (removes fringe hair)
        top = pts[10, 1]
        brow = pts[[105, 334, 66, 296]].mean(0)[1]
        ys = poly[:, 1]
        lim = top + (brow - top) * forehead_drop
        poly[:, 1] = np.maximum(ys, np.minimum(lim, ys + (lim - top)))
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(poly * 16).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=4)
    return m.astype(np.float32) / 255.0


class FaceJob:
    """Settings for one face build."""

    def __init__(self, scale=3, feather=0.08, shrink=0.07, forehead_drop=0.3,
                 tone="photo", strength=1.0, saturation=1.0, contrast=1.0, brightness=1.0):
        self.scale = scale
        self.feather = feather
        self.shrink = shrink
        self.forehead_drop = forehead_drop
        self.tone = tone  # "photo": head takes the photo's skin; "game": photo takes head's skin
        self.strength = strength
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness


def build_face(template, tpl_lm, photo, job=None):
    """template: PIL RGBA game face bitmap. tpl_lm: landmarks dict for template (pixel coords).
    photo: candidate dict from analyse_photo. Returns (face RGBA PIL at job.scale x size,
    skin_ratio rgb array used to recolour the other head parts)."""
    job = job or FaceJob()
    k = job.scale
    tw, th = template.size
    W, H = tw * k, th * k
    tpl = np.asarray(template.convert("RGBA").resize((W, H), Image.BICUBIC), np.float32)
    tpts = with_hole_centres(tpl_lm["pts"] * k)
    ppts = with_hole_centres(photo["pts"])
    src = np.asarray(photo["img"], np.float32)

    # Similarity-normalise: nothing needed, the triangle warp handles scale/rotation.
    warped, cov = warp_mesh(src, ppts, tpts, (W, H), dst_z=tpts[:, 2])
    # edge pixels were blended against black - undo that
    warped = np.where(cov[..., None] > 0.02, warped / np.maximum(cov[..., None], 0.02), 0)

    face_w = np.linalg.norm(tpts[234, :2] - tpts[454, :2])
    rad = max(2, face_w * 0.035)
    p_skin = _sample(warped, tpts, SKIN_SAMPLE, rad)
    t_skin = _sample(tpl, tpts, SKIN_SAMPLE, rad)
    ratio = (p_skin + 1) / (t_skin + 1)

    # Photo grading
    if job.brightness != 1.0:
        warped *= job.brightness
    if job.contrast != 1.0:
        mu = warped[cov > 0.5].mean(0) if (cov > 0.5).any() else 128
        warped = (warped - mu) * job.contrast + mu
    if job.saturation != 1.0:
        g = warped.mean(2, keepdims=True)
        warped = g + (warped - g) * job.saturation

    base = tpl[..., :3].copy()
    if job.tone == "photo":
        base = base * ratio  # recolour head to the photo's skin
        ratio_out = ratio
    else:
        warped = warped / ratio  # recolour photo to the game's skin
        ratio_out = np.ones(3, np.float32)

    m0 = oval_mask(tpts, (W, H), job.shrink, job.forehead_drop)
    # always take both eyes from the photo (the far eye sits right on the oval edge)
    eyes = np.zeros((H, W), np.uint8)
    for ring in (LEFT_EYE, RIGHT_EYE):
        poly = tpts[ring, :2]
        c = poly.mean(0)
        cv2.fillPoly(eyes, [np.round((c + (poly - c) * 1.6) * 16).astype(np.int32)], 255,
                     lineType=cv2.LINE_AA, shift=4)
    m0 = np.maximum(m0, eyes.astype(np.float32) / 255.0) * (cov > 0.99)
    fr = max(1, face_w * job.feather)
    alpha = tpl[..., 3] / 255.0
    m = np.clip(cv2.GaussianBlur(m0, (0, 0), fr / 2.0) * alpha, 0, 1) * job.strength
    # two-band blend: low frequencies (lighting/colour) blend over a wide band so the
    # seam disappears, detail blends over the narrow band so the photo stays crisp
    filled = warped * cov[..., None] + base * (1 - cov[..., None])
    sig = face_w * 0.12
    low_b = cv2.GaussianBlur(base, (0, 0), sig)
    # normalised blur: only pixels inside the face contribute, so hair/background
    # just outside the oval can't bleed a dark halo into the blend band
    den = cv2.GaussianBlur(m0, (0, 0), sig)[..., None]
    low_w = cv2.GaussianBlur(filled * m0[..., None], (0, 0), sig) / np.maximum(den, 1e-4)
    low_w = np.where(den > 1e-3, low_w, low_b)
    mw = np.clip(cv2.GaussianBlur(m0, (0, 0), face_w * 0.15) * 1.3, 0, 1) * job.strength
    mw = np.maximum(mw, m)
    low = low_b * (1 - mw[..., None]) + low_w * mw[..., None]
    high = (base - low_b) * (1 - m[..., None]) + (filled - low_w) * m[..., None]
    rgb = low + high
    out = np.dstack([np.clip(rgb, 0, 255), tpl[..., 3]]).astype(np.uint8)
    face = Image.fromarray(out, "RGBA")
    # eyes-shut version (for the blink frame's eyelid patch): paint the eyes out
    em = np.zeros((H, W), np.uint8)
    for ring in (LEFT_EYE, RIGHT_EYE):
        poly = tpts[ring, :2]
        c = poly.mean(0)
        cv2.fillPoly(em, [np.round((c + (poly - c) * 1.35) * 16).astype(np.int32)], 255, shift=4)
    closed = cv2.inpaint(np.ascontiguousarray(out[..., :3]), em, max(3, int(face_w * 0.05)),
                         cv2.INPAINT_TELEA)
    face.info["closed"] = Image.fromarray(np.dstack([closed, out[..., 3]]), "RGBA")
    return face, ratio_out


def sample_hair(cand):
    """Median colour of the band above the forehead in the photo (0-255 RGB)."""
    im = np.asarray(cand["img"], np.float32)
    p = cand["pts"]
    top = p[10, :2]
    chin = p[152, :2]
    fh = np.linalg.norm(chin - top)
    x0, x1 = p[[67, 297], 0].min(), p[[67, 297], 0].max()
    y1 = top[1] - fh * 0.06
    y0 = top[1] - fh * 0.22
    x0, x1 = int(max(0, x0)), int(min(im.shape[1], x1))
    y0, y1 = int(max(0, y0)), int(max(0, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    reg = im[y0:y1, x0:x1].reshape(-1, 3)
    return np.median(reg, axis=0)


def recolour(img, ratio, scale=1):
    """Multiply an RGBA image's colour by ratio (and optionally upscale)."""
    im = img.convert("RGBA")
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.BICUBIC)
    a = np.asarray(im, np.float32)
    a[..., :3] = np.clip(a[..., :3] * ratio, 0, 255)
    return Image.fromarray(a.astype(np.uint8), "RGBA")
