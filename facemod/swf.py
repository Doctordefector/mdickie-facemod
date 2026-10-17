"""Minimal SWF reader/writer: enough to list tags, pull bitmaps out and put them back."""
import struct
import zlib

# Tag codes we care about
END = 0
SHOW_FRAME = 1
DEFINE_SHAPE = 2
DEFINE_BITS = 6
JPEG_TABLES = 8
DEFINE_BITS_LOSSLESS = 20
DEFINE_BITS_JPEG2 = 21
DEFINE_SHAPE2 = 22
DEFINE_SHAPE3 = 32
DEFINE_BITS_JPEG3 = 35
DEFINE_BITS_LOSSLESS2 = 36
DEFINE_SPRITE = 39
EXPORT_ASSETS = 56
DO_ABC = 82
DEFINE_SHAPE4 = 83
SYMBOL_CLASS = 76
DEFINE_BITS_JPEG4 = 90

BITMAP_TAGS = {DEFINE_BITS, DEFINE_BITS_JPEG2, DEFINE_BITS_JPEG3, DEFINE_BITS_JPEG4,
               DEFINE_BITS_LOSSLESS, DEFINE_BITS_LOSSLESS2}


class Tag:
    __slots__ = ("code", "data")

    def __init__(self, code, data):
        self.code = code
        self.data = data

    @property
    def char_id(self):
        return struct.unpack_from("<H", self.data, 0)[0]

    def encode(self):
        n = len(self.data)
        # Bitmap tags must always use the long header form
        if n >= 0x3F or self.code in BITMAP_TAGS:
            return struct.pack("<HI", (self.code << 6) | 0x3F, n) + self.data
        return struct.pack("<H", (self.code << 6) | n) + self.data


class SWF:
    def __init__(self, path):
        with open(path, "rb") as f:
            raw = f.read()
        sig = raw[:3]
        self.version = raw[3]
        if sig == b"CWS":
            body = zlib.decompress(raw[8:])
        elif sig == b"FWS":
            body = raw[8:]
        elif sig == b"ZWS":
            import lzma
            props = raw[12:17]
            dec = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=[
                lzma._decode_filter_properties(lzma.FILTER_LZMA1, props)])
            body = dec.decompress(raw[17:])
        else:
            raise ValueError("not a SWF: %r" % sig)
        self.compressed = sig != b"FWS"
        # RECT header is bit-packed: 5 bits of nbits then 4 fields
        nbits = body[0] >> 3
        rect_len = (5 + 4 * nbits + 7) // 8
        self.header = body[:rect_len + 4]  # rect + frame rate + frame count
        self.tags = list(self._parse(body, rect_len + 4))

    @staticmethod
    def _parse(buf, pos):
        end = len(buf)
        while pos < end:
            code_len, = struct.unpack_from("<H", buf, pos)
            pos += 2
            code, n = code_len >> 6, code_len & 0x3F
            if n == 0x3F:
                n, = struct.unpack_from("<I", buf, pos)
                pos += 4
            yield Tag(code, buf[pos:pos + n])
            pos += n
            if code == END:
                break

    def save(self, path, level=6):
        body = self.header + b"".join(t.encode() for t in self.tags)
        length = 8 + len(body)
        if self.compressed:
            out = b"CWS" + bytes([self.version]) + struct.pack("<I", length) + zlib.compress(body, level)
        else:
            out = b"FWS" + bytes([self.version]) + struct.pack("<I", length) + body
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(out)
        import os
        os.replace(tmp, path)

    # ---- helpers -------------------------------------------------------
    def symbol_classes(self):
        """char id -> class name, from SymbolClass/ExportAssets tags."""
        out = {}
        for t in self.tags:
            if t.code in (SYMBOL_CLASS, EXPORT_ASSETS):
                d = t.data
                count, = struct.unpack_from("<H", d, 0)
                p = 2
                for _ in range(count):
                    cid, = struct.unpack_from("<H", d, p)
                    p += 2
                    e = d.index(b"\0", p)
                    out[cid] = d[p:e].decode("utf-8", "replace")
                    p = e + 1
        return out

    def by_id(self):
        out = {}
        for i, t in enumerate(self.tags):
            if t.code in BITMAP_TAGS or t.code in (DEFINE_SHAPE, DEFINE_SHAPE2, DEFINE_SHAPE3,
                                                   DEFINE_SHAPE4, DEFINE_SPRITE):
                out[t.char_id] = i
        return out


# ---- bitmap decode/encode -------------------------------------------------

def decode_bitmap(tag, jpeg_tables=None):
    """Return a PIL RGBA image for a bitmap tag."""
    from PIL import Image
    import io
    import numpy as np
    d = tag.data
    c = tag.code
    if c in (DEFINE_BITS_LOSSLESS, DEFINE_BITS_LOSSLESS2):
        cid, fmt, w, h = struct.unpack_from("<HBHH", d, 0)
        alpha = c == DEFINE_BITS_LOSSLESS2
        if fmt == 3:
            ncolors = d[7] + 1
            raw = zlib.decompress(d[8:])
            csize = 4 if alpha else 3
            pal = np.frombuffer(raw[:ncolors * csize], np.uint8).reshape(-1, csize)
            stride = (w + 3) & ~3
            idx = np.frombuffer(raw[ncolors * csize:ncolors * csize + stride * h], np.uint8)
            idx = idx.reshape(h, stride)[:, :w]
            px = pal[idx]
            if not alpha:
                px = np.dstack([px, np.full((h, w), 255, np.uint8)])
            else:
                px = _unpremultiply(px)
            return Image.fromarray(px, "RGBA")
        if fmt == 5:
            raw = zlib.decompress(d[7:])
            a = np.frombuffer(raw[:w * h * 4], np.uint8).reshape(h, w, 4)
            argb = a[..., [1, 2, 3, 0]].copy()  # ARGB -> RGBA
            if alpha:
                argb = _unpremultiply(argb)
            else:
                argb[..., 3] = 255
            return Image.fromarray(argb, "RGBA")
        if fmt == 4:
            raw = zlib.decompress(d[7:])
            stride = ((w * 2) + 3) & ~3
            a = np.frombuffer(raw, np.uint8)[:stride * h].reshape(h, stride)[:, :w * 2]
            v = (a[:, 0::2].astype(np.uint16) << 8) | a[:, 1::2]
            r = ((v >> 10) & 31) * 255 // 31
            g = ((v >> 5) & 31) * 255 // 31
            b = (v & 31) * 255 // 31
            px = np.dstack([r, g, b, np.full_like(r, 255)]).astype(np.uint8)
            return Image.fromarray(px, "RGBA")
        raise ValueError("lossless fmt %d" % fmt)
    if c == DEFINE_BITS_JPEG2 or c == DEFINE_BITS:
        jpg = d[2:]
        if c == DEFINE_BITS and jpeg_tables:
            jpg = jpeg_tables[:-2] + jpg[2:]
        return Image.open(io.BytesIO(_fix_jpeg(jpg))).convert("RGBA")
    if c in (DEFINE_BITS_JPEG3, DEFINE_BITS_JPEG4):
        off, = struct.unpack_from("<I", d, 2)
        p = 6 + (2 if c == DEFINE_BITS_JPEG4 else 0)
        img_data = d[p:p + off]
        im = Image.open(io.BytesIO(_fix_jpeg(img_data))).convert("RGBA")
        if img_data[:4] == b"\xff\xd8\xff\xe0" or img_data[:2] == b"\xff\xd8":
            alpha_raw = d[p + off:]
            if alpha_raw:
                a = np.frombuffer(zlib.decompress(alpha_raw), np.uint8)
                px = np.array(im)
                px[..., 3] = a[:px.shape[0] * px.shape[1]].reshape(px.shape[:2])
                im = Image.fromarray(px, "RGBA")
        return im
    raise ValueError("not a bitmap tag %d" % c)


def _fix_jpeg(b):
    # Old SWFs sometimes have an extra EOI/SOI pair at the start
    if b[:4] == b"\xff\xd9\xff\xd8":
        b = b[4:]
    return b


def _unpremultiply(px):
    import numpy as np
    px = px.astype(np.float32)
    a = px[..., 3:4]
    rgb = np.where(a > 0, px[..., :3] * 255.0 / np.maximum(a, 1), 0)
    return np.dstack([np.clip(rgb, 0, 255), a]).astype(np.uint8)


def encode_lossless2(char_id, img):
    """Build DefineBitsLossless2 tag data (format 5, premultiplied ARGB) from a PIL image."""
    import numpy as np
    px = np.array(img.convert("RGBA")).astype(np.float32)
    a = px[..., 3:4]
    rgb = np.round(px[..., :3] * a / 255.0)
    argb = np.dstack([a, rgb]).astype(np.uint8)
    h, w = argb.shape[:2]
    return struct.pack("<HBHH", char_id, 5, w, h) + zlib.compress(argb.tobytes(), 9)
