"""Tiny software renderer for bitmap-based SWF sprites (enough for previews)."""
import numpy as np
from PIL import Image
from . import swf as S
from .swfstruct import shape_info, sprite_timeline

SHAPES = (S.DEFINE_SHAPE, S.DEFINE_SHAPE2, S.DEFINE_SHAPE3, S.DEFINE_SHAPE4)
IDENT = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def mul(m, n):
    """Compose: apply n first, then m. Matrices are (a, b, c, d, tx, ty) in SWF order,
    x' = a*x + c*y + tx ; y' = b*x + d*y + ty."""
    a, b, c, d, tx, ty = m
    a2, b2, c2, d2, tx2, ty2 = n
    return (a * a2 + c * b2, b * a2 + d * b2,
            a * c2 + c * d2, b * c2 + d * d2,
            a * tx2 + c * ty2 + tx, b * tx2 + d * ty2 + ty)


class Renderer:
    def __init__(self, swf):
        self.swf = swf
        self.by_id = swf.by_id()
        self.jt = next((t.data for t in swf.tags if t.code == S.JPEG_TABLES), None)
        self._bmp = {}
        self._tl = {}
        self.overrides = {}  # bitmap id -> PIL image (for previewing replacements)
        self.frame_choice = {}  # instance name or sprite id -> frame index (0-based); None hides it

    def bitmap(self, bid):
        if bid in self.overrides:
            return self.overrides[bid]
        if bid not in self._bmp:
            self._bmp[bid] = S.decode_bitmap(self.swf.tags[self.by_id[bid]], self.jt)
        return self._bmp[bid]

    def timeline(self, sid):
        if sid not in self._tl:
            self._tl[sid] = sprite_timeline(self.swf.tags[self.by_id[sid]])
        return self._tl[sid]

    def collect(self, cid, frame=0, m=IDENT, out=None, path=()):
        """Flatten into a list of (bitmap_id, matrix bitmap-px -> twips, path, shape bounds)."""
        if out is None:
            out = []
        if cid not in self.by_id:
            return out
        t = self.swf.tags[self.by_id[cid]]
        if t.code in SHAPES:
            info = shape_info(t)
            for f in info["fills"]:
                if f[1] is not None and f[1] in self.by_id:
                    out.append((f[1], mul(m, f[2]), path, info["bounds"]))
        elif t.code == S.DEFINE_SPRITE:
            _, frames, _ = self.timeline(cid)
            if not frames:
                return out
            fr = frames[min(frame, len(frames) - 1)]
            for depth in sorted(fr):
                e = fr[depth]
                if "id" not in e:
                    continue
                nm = e.get("name")
                sub = self.frame_choice.get(nm, self.frame_choice.get(e["id"], 0))
                if sub is None:
                    continue
                self.collect(e["id"], sub, mul(m, e.get("matrix", IDENT)), out,
                             path + ((depth, nm, e["id"]),))
        return out

    def render(self, cid, frame=0, scale=4.0, pad=4, items=None, bg=None):
        items = items if items is not None else self.collect(cid, frame)
        if not items:
            return Image.new("RGBA", (1, 1)), (0, 0)
        xs, ys = [], []
        for bid, m, _, _ in items:
            w, h = self.bitmap(bid).size
            for px, py in ((0, 0), (w, 0), (0, h), (w, h)):
                xs.append(m[0] * px + m[2] * py + m[4])
                ys.append(m[1] * px + m[3] * py + m[5])
        k = scale / 20.0
        x0, y0 = min(xs) * k - pad, min(ys) * k - pad
        W = int(max(xs) * k - x0 + pad) + 1
        H = int(max(ys) * k - y0 + pad) + 1
        canvas = Image.new("RGBA", (W, H), bg or (0, 0, 0, 0))
        for bid, m, _, _ in items:
            im = self.bitmap(bid)
            a, b, c, d, tx, ty = m
            A = np.array([[a * k, c * k, tx * k - x0], [b * k, d * k, ty * k - y0], [0, 0, 1]])
            inv = np.linalg.inv(A)
            layer = im.transform((W, H), Image.AFFINE, tuple(inv[:2].flatten()), Image.BILINEAR)
            canvas.alpha_composite(layer)
        return canvas, (x0, y0)
