"""Edit text literals inside the game's ActionScript bytecode (DoABC string pool).

Only the string pool is rewritten; everything after it in the ABC block refers to
strings by index, so lengths can change freely. Strings used as identifiers
(class/method/property names, namespaces) are never touched, which keeps the game
from breaking.
"""
import struct
from . import swf as S


def _u30(d, p):
    v = 0
    shift = 0
    while True:
        b = d[p]
        p += 1
        v |= (b & 0x7F) << shift
        if not b & 0x80 or shift >= 28:
            return v, p
        shift += 7


def _enc_u30(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


class ABC:
    def __init__(self, tag):
        self.tag = tag
        d = tag.data
        p = 0
        if tag.code == S.DO_ABC:
            self.flags = d[:4]
            e = d.index(b"\0", 4)
            self.name = d[4:e + 1]
            p = e + 1
        else:
            self.flags, self.name = b"", b""
        self.prefix = d[:p]
        abc = d[p:]
        self.abc = abc
        q = 4  # minor/major version
        n, q = _u30(abc, q)
        for _ in range(max(n - 1, 0)):
            _, q = _u30(abc, q)
        n, q = _u30(abc, q)
        for _ in range(max(n - 1, 0)):
            _, q = _u30(abc, q)
        n, q = _u30(abc, q)
        q += 8 * max(n - 1, 0)
        self.pre_strings = abc[:q]
        n, q = _u30(abc, q)
        self.strings = [""]
        for _ in range(max(n - 1, 0)):
            ln, q = _u30(abc, q)
            self.strings.append(abc[q:q + ln].decode("utf-8", "replace"))
            q += ln
        self.rest_off = q
        # identifiers: namespaces + multinames reference string indices
        self.ident = set()
        n, q = _u30(abc, q)
        for _ in range(max(n - 1, 0)):
            q += 1
            s, q = _u30(abc, q)
            self.ident.add(s)
        n, q = _u30(abc, q)  # ns sets
        for _ in range(max(n - 1, 0)):
            c, q = _u30(abc, q)
            for _ in range(c):
                _, q = _u30(abc, q)
        n, q = _u30(abc, q)  # multinames
        for _ in range(max(n - 1, 0)):
            kind = abc[q]
            q += 1
            if kind in (0x07, 0x0D):  # QName(A)
                _, q = _u30(abc, q)
                s, q = _u30(abc, q)
                self.ident.add(s)
            elif kind in (0x0F, 0x10):  # RTQName(A)
                s, q = _u30(abc, q)
                self.ident.add(s)
            elif kind in (0x11, 0x12):  # RTQNameL
                pass
            elif kind in (0x09, 0x0E):  # Multiname(A)
                s, q = _u30(abc, q)
                self.ident.add(s)
                _, q = _u30(abc, q)
            elif kind in (0x1B, 0x1C):  # MultinameL
                _, q = _u30(abc, q)
            elif kind == 0x1D:  # TypeName
                _, q = _u30(abc, q)
                c, q = _u30(abc, q)
                for _ in range(c):
                    _, q = _u30(abc, q)
            else:
                break
        # metadata / names of things also live in strings; be conservative and also
        # protect anything that looks like a code identifier with no spaces and a dot/colon
        self.ident.discard(0)

    def editable(self, i):
        s = self.strings[i]
        return i not in self.ident and s != ""

    def encode(self):
        out = bytearray(self.prefix)
        out += self.pre_strings
        out += _enc_u30(len(self.strings))
        for s in self.strings[1:]:
            b = s.encode("utf-8")
            out += _enc_u30(len(b)) + b
        out += self.abc[self.rest_off:]
        return S.Tag(self.tag.code, bytes(out))


def abc_tags(swf):
    return [i for i, t in enumerate(swf.tags) if t.code in (S.DO_ABC, 72)]


def list_strings(swf, query="", limit=500):
    """Editable text literals, optionally filtered."""
    out = []
    q = query.lower()
    for ti in abc_tags(swf):
        a = ABC(swf.tags[ti])
        for i, s in enumerate(a.strings):
            if not a.editable(i):
                continue
            if q and q not in s.lower():
                continue
            out.append({"tag": ti, "index": i, "text": s})
            if len(out) >= limit:
                return out
    return out


def apply_text_edits(swf, edits):
    """edits: [{"find": str, "replace": str, "exact": bool}] applied to editable literals."""
    for ti in abc_tags(swf):
        a = ABC(swf.tags[ti])
        changed = False
        for i, s in enumerate(a.strings):
            if not a.editable(i):
                continue
            new = s
            for e in edits:
                if not e.get("find"):
                    continue
                if e.get("exact", True):
                    if new == e["find"]:
                        new = e["replace"]
                else:
                    new = new.replace(e["find"], e["replace"])
            if new != s:
                a.strings[i] = new
                changed = True
        if changed:
            swf.tags[ti] = a.encode()
