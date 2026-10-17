"""Structure helpers: sprite timelines, shape bitmap fills, matrices."""
import struct
from . import swf as S

PLACE_OBJECT = 4
PLACE_OBJECT2 = 26
PLACE_OBJECT3 = 70
REMOVE_OBJECT2 = 28
FRAME_LABEL = 43


class Bits:
    def __init__(self, data, pos=0):
        self.d = data
        self.bit = pos * 8

    def ub(self, n):
        v = 0
        for _ in range(n):
            byte = self.d[self.bit >> 3]
            v = (v << 1) | ((byte >> (7 - (self.bit & 7))) & 1)
            self.bit += 1
        return v

    def sb(self, n):
        if n == 0:
            return 0
        v = self.ub(n)
        if v & (1 << (n - 1)):
            v -= 1 << n
        return v

    def align(self):
        self.bit = (self.bit + 7) & ~7

    @property
    def pos(self):
        return (self.bit + 7) >> 3


def read_rect(b):
    n = b.ub(5)
    r = [b.sb(n) for _ in range(4)]
    b.align()
    return r  # xmin xmax ymin ymax (twips)


def read_matrix(b):
    """Returns (a, b, c, d, tx, ty) floats (tx/ty in twips). Leaves reader byte-aligned."""
    a = d = 1.0
    bb = c = 0.0
    if b.ub(1):
        n = b.ub(5)
        a = b.sb(n) / 65536.0
        d = b.sb(n) / 65536.0
    if b.ub(1):
        n = b.ub(5)
        bb = b.sb(n) / 65536.0
        c = b.sb(n) / 65536.0
    n = b.ub(5)
    tx = b.sb(n)
    ty = b.sb(n)
    b.align()
    return (a, bb, c, d, tx, ty)


def write_matrix(m):
    """Encode matrix (a, b, c, d, tx, ty) to bytes."""
    a, bb, c, d, tx, ty = m
    bits = []

    def put(v, n):
        for i in range(n - 1, -1, -1):
            bits.append((v >> i) & 1)

    def nbits_s(*vals):
        n = 1
        for v in vals:
            while not (-(1 << (n - 1)) <= v < (1 << (n - 1))):
                n += 1
        return n

    A, D = int(round(a * 65536)), int(round(d * 65536))
    if (A, D) != (65536, 65536):
        put(1, 1)
        n = nbits_s(A, D)
        put(n, 5); put(A & ((1 << n) - 1), n); put(D & ((1 << n) - 1), n)
    else:
        put(0, 1)
    B, C = int(round(bb * 65536)), int(round(c * 65536))
    if (B, C) != (0, 0):
        put(1, 1)
        n = nbits_s(B, C)
        put(n, 5); put(B & ((1 << n) - 1), n); put(C & ((1 << n) - 1), n)
    else:
        put(0, 1)
    tx, ty = int(tx), int(ty)
    n = nbits_s(tx, ty) if (tx or ty) else 0
    put(n, 5)
    if n:
        put(tx & ((1 << n) - 1), n); put(ty & ((1 << n) - 1), n)
    while len(bits) % 8:
        bits.append(0)
    out = bytearray()
    for i in range(0, len(bits), 8):
        v = 0
        for x in bits[i:i + 8]:
            v = (v << 1) | x
        out.append(v)
    return bytes(out)


def shape_info(tag):
    """Parse a DefineShape* header + first fill style array.

    Returns dict: id, bounds, fills=[(type, bitmap_id, matrix, byte_offset_of_matrix, matrix_len)].
    """
    d = tag.data
    cid, = struct.unpack_from("<H", d, 0)
    b = Bits(d, 2)
    bounds = read_rect(b)
    p = b.pos
    if tag.code == S.DEFINE_SHAPE4:
        b = Bits(d, p)
        read_rect(b)  # edge bounds
        p = b.pos + 1  # flags byte
    count = d[p]; p += 1
    if count == 0xFF and tag.code != S.DEFINE_SHAPE:
        count, = struct.unpack_from("<H", d, p); p += 2
    fills = []
    rgba = tag.code in (S.DEFINE_SHAPE3, S.DEFINE_SHAPE4)
    for _ in range(count):
        ft = d[p]; p += 1
        if ft == 0x00:
            p += 4 if rgba else 3
            fills.append((ft, None, None, None, None))
        elif ft in (0x10, 0x12, 0x13):
            b = Bits(d, p); read_matrix(b); p = b.pos
            n = d[p] & 0x0F; p += 1
            p += n * (1 + (4 if rgba else 3))
            if ft == 0x13:
                p += 2
            fills.append((ft, None, None, None, None))
        elif 0x40 <= ft <= 0x43:
            bid, = struct.unpack_from("<H", d, p); p += 2
            b = Bits(d, p); m = read_matrix(b)
            fills.append((ft, bid, m, p, b.pos - p))
            p = b.pos
        else:
            break
    return {"id": cid, "bounds": bounds, "fills": fills}


def sprite_timeline(tag):
    """Parse a DefineSprite. Returns (sprite_id, frames) where frames is a list of
    dicts depth -> {"id", "matrix", "name"} representing the display list per frame,
    plus labels."""
    d = tag.data
    sid, nframes = struct.unpack_from("<HH", d, 0)
    inner = list(S.SWF._parse(d, 4))
    frames = []
    labels = {}
    disp = {}
    for t in inner:
        if t.code == S.SHOW_FRAME:
            frames.append({k: dict(v) for k, v in disp.items()})
        elif t.code in (PLACE_OBJECT2, PLACE_OBJECT3):
            pd = t.data
            flags = pd[0]
            p = 1
            flags2 = 0
            if t.code == PLACE_OBJECT3:
                flags2 = pd[1]; p = 2
            depth, = struct.unpack_from("<H", pd, p); p += 2
            if t.code == PLACE_OBJECT3 and (flags2 & 0x08 or (flags2 & 0x10 and flags & 0x02)):
                e = pd.index(b"\0", p); p = e + 1  # class name
            ent = disp.get(depth, {}) if (flags & 0x01) else {}
            if flags & 0x02:
                ent["id"], = struct.unpack_from("<H", pd, p); p += 2
            if flags & 0x04:
                b = Bits(pd, p); ent["matrix"] = read_matrix(b); p = b.pos
            if flags & 0x08:
                b = Bits(pd, p)
                has_add = b.ub(1); has_mult = b.ub(1); n = b.ub(4)
                mult = [b.sb(n) for _ in range(4)] if has_mult else None
                add = [b.sb(n) for _ in range(4)] if has_add else None
                b.align(); p = b.pos
                ent["cx"] = (mult, add)
            if flags & 0x10:
                p += 2  # ratio
            if flags & 0x20:
                e = pd.index(b"\0", p); ent["name"] = pd[p:e].decode("utf-8", "replace"); p = e + 1
            disp[depth] = ent
        elif t.code == REMOVE_OBJECT2:
            depth, = struct.unpack_from("<H", t.data, 0)
            disp.pop(depth, None)
        elif t.code == PLACE_OBJECT:
            cid, depth = struct.unpack_from("<HH", t.data, 0)
            b = Bits(t.data, 4)
            disp[depth] = {"id": cid, "matrix": read_matrix(b)}
        elif t.code == FRAME_LABEL:
            labels[len(frames)] = t.data.split(b"\0")[0].decode()
    return sid, frames, labels
