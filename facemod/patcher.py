"""Rebuild a HeadNN sprite in a game SWF with custom bitmaps.

How a head is built (same in all four games):
  HeadNN = DefineSprite with 3 frames
    frame 1: back of head      -> unnamed shape with a bitmap fill
    frame 2: face (3/4 view)   -> unnamed shape + named overlays Mouth/Eyebrows/Blood/Beard/Hair/Hat
    frame 3: blinking face     -> frame 2 + (sometimes) a small eyelid patch shape
Bitmaps/shapes are often shared between heads, so we never edit in place: we clone
the template head's timeline into the target slot's sprite id, pointing at fresh
shapes and bitmaps.
"""
import struct
from . import swf as S
from .swfstruct import Bits, read_matrix, write_matrix, shape_info, sprite_timeline, \
    PLACE_OBJECT2, PLACE_OBJECT3

DEFINE_CODES = {2, 6, 7, 10, 11, 13, 14, 20, 21, 22, 32, 33, 34, 35, 36, 37, 39, 46, 48, 60,
                75, 83, 84, 87, 90, 91}
SHAPES = (S.DEFINE_SHAPE, S.DEFINE_SHAPE2, S.DEFINE_SHAPE3, S.DEFINE_SHAPE4)


def head_class(n):
    return "Head%02d" % n


class HeadInfo:
    """What's inside one HeadNN sprite."""

    def __init__(self, swf, n):
        self.n = n
        sc = swf.symbol_classes()
        inv = {v: k for k, v in sc.items()}
        self.sprite_id = inv[head_class(n)]
        bi = swf.by_id()
        self.tag = swf.tags[bi[self.sprite_id]]
        _, self.frames, _ = sprite_timeline(self.tag)
        self.roles = {}  # shape id -> role ("back" | "face" | "blink")
        self.shape_bitmaps = {}  # shape id -> [bitmap ids]
        self.named = {}  # instance name -> sprite id
        for fi, fr in enumerate(self.frames):
            for depth, e in fr.items():
                if "id" not in e:
                    continue
                if e.get("name"):
                    self.named[e["name"]] = e["id"]
                    continue
                cid = e["id"]
                t = swf.tags[bi[cid]] if cid in bi else None
                if t is None or t.code not in SHAPES:
                    continue
                bids = [f[1] for f in shape_info(t)["fills"] if f[1] is not None and f[1] in bi]
                self.shape_bitmaps[cid] = bids
                if cid in self.roles:
                    continue
                if fi == 0:
                    self.roles[cid] = "back"
                elif fi == 1:
                    self.roles[cid] = "face"
                else:
                    face_bmps = {b for s, r in self.roles.items() if r == "face" for b in self.shape_bitmaps[s]}
                    self.roles[cid] = "face" if set(bids) <= face_bmps else "blink"

    def bitmap_for(self, role):
        for s, r in self.roles.items():
            if r == role and self.shape_bitmaps.get(s):
                return self.shape_bitmaps[s][0]
        return None


def _max_id(swf):
    m = 0
    for t in swf.tags:
        if t.code in DEFINE_CODES and len(t.data) >= 2:
            m = max(m, struct.unpack_from("<H", t.data, 0)[0])
    return m


def _clone_shape(tag, new_id, bmp_map, scale_map):
    """Copy a shape tag with a new id; swap bitmap fill ids per bmp_map and scale their
    fill matrices by 1/scale (so a k-times larger bitmap covers the same area)."""
    d = bytearray(tag.data)
    struct.pack_into("<H", d, 0, new_id)
    info = shape_info(tag)
    # patch from the back so earlier offsets stay valid
    for ft, bid, m, off, mlen in sorted([f for f in info["fills"] if f[1] in bmp_map],
                                        key=lambda f: -f[3]):
        k = scale_map[bid]
        a, b, c, dd, tx, ty = m
        newm = write_matrix((a / k, b / k, c / k, dd / k, tx, ty))
        struct.pack_into("<H", d, off - 2, bmp_map[bid])
        d[off:off + mlen] = newm
    return S.Tag(tag.code, bytes(d))


def _remap_sprite(tag, new_sprite_id, id_map):
    """Copy a DefineSprite, giving it a new id and remapping PlaceObject character ids."""
    d = tag.data
    _, nframes = struct.unpack_from("<HH", d, 0)
    inner = list(S.SWF._parse(d, 4))
    out = []
    for t in inner:
        if t.code in (PLACE_OBJECT2, PLACE_OBJECT3):
            pd = bytearray(t.data)
            flags = pd[0]
            p = 1
            if t.code == PLACE_OBJECT3:
                flags2 = pd[1]
                p = 2
                p += 2  # depth
                if flags2 & 0x08 or (flags2 & 0x10 and flags & 0x02):
                    p = pd.index(b"\0", p) + 1
            else:
                p += 2
            if flags & 0x02:
                cid, = struct.unpack_from("<H", pd, p)
                if cid in id_map:
                    struct.pack_into("<H", pd, p, id_map[cid])
            t = S.Tag(t.code, bytes(pd))
        out.append(t)
    body = struct.pack("<HH", new_sprite_id, nframes) + b"".join(x.encode() for x in out)
    return S.Tag(S.DEFINE_SPRITE, body)


def _empty_sprite(sid, nframes):
    body = struct.pack("<HH", sid, nframes)
    body += b"".join(S.Tag(S.SHOW_FRAME, b"").encode() for _ in range(nframes))
    body += S.Tag(S.END, b"").encode()
    return S.Tag(S.DEFINE_SPRITE, body)


def install_head(swf, slot, template, images, hide=()):
    """Rebuild head `slot` as a copy of head `template` using new images.

    images: dict role -> PIL RGBA image ("face", "back", "blink"); a role that is missing
    keeps the template's original bitmap. Each image may be any integer multiple of the
    original bitmap size (higher res = sharper in game).
    hide: instance names (e.g. "Eyebrows") to replace with an empty clip.
    """
    tpl = HeadInfo(swf, template)
    dst = HeadInfo(swf, slot)
    bi = swf.by_id()
    jt = next((t.data for t in swf.tags if t.code == S.JPEG_TABLES), None)
    next_id = [_max_id(swf) + 1]

    def nid():
        v = next_id[0]
        next_id[0] += 1
        if v > 65534:
            raise RuntimeError("SWF ran out of character ids")
        return v

    new_tags = []
    bmp_map, scale_map = {}, {}
    for role in ("face", "back", "blink"):
        img = images.get(role)
        old = tpl.bitmap_for(role)
        if img is None or old is None or old in bmp_map:
            continue
        ow, oh = S.decode_bitmap(swf.tags[bi[old]], jt).size
        k = img.width / ow
        new_bid = nid()
        new_tags.append(S.Tag(S.DEFINE_BITS_LOSSLESS2, S.encode_lossless2(new_bid, img)))
        bmp_map[old] = new_bid
        scale_map[old] = k
    id_map = {}
    for shp, bids in tpl.shape_bitmaps.items():
        if any(b in bmp_map for b in bids):
            new_sid = nid()
            new_tags.append(_clone_shape(swf.tags[bi[shp]], new_sid, bmp_map, scale_map))
            id_map[shp] = new_sid
    for name in hide:
        if name in tpl.named:
            ref = swf.tags[bi[tpl.named[name]]]
            _, nf = struct.unpack_from("<HH", ref.data, 0)
            eid = nid()
            new_tags.append(_empty_sprite(eid, nf))
            id_map[tpl.named[name]] = eid
    new_sprite = _remap_sprite(tpl.tag, dst.sprite_id, id_map)

    # drop the old slot definition, add ours (plus its parts) after all other definitions
    old_index = bi[dst.sprite_id]
    del swf.tags[old_index]
    at = next(i for i, t in enumerate(swf.tags) if t.code in (S.DO_ABC, 72, S.SYMBOL_CLASS))
    swf.tags[at:at] = new_tags + [new_sprite]
    return {"new_ids": [struct.unpack_from("<H", t.data, 0)[0] for t in new_tags]}
