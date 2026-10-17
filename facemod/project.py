"""Mod management for one game folder.

Everything is rebuilt from pristine backups on every apply, so mods can be removed
or edited freely:
  <game>/FaceMod/original/<swf>, Characters.txt   untouched copies (made on first use)
  <game>/FaceMod/mods.json                         list of face mods + text edits
  <game>/FaceMod/faces/<id>/...                    generated bitmaps + source photos
"""
import json
import os
import shutil
import time
import uuid

import numpy as np
from PIL import Image

from . import games as G
from . import swf as S
from . import faceswap as FS
from . import landmarks as LM
from .render import Renderer
from .patcher import HeadInfo, install_head, head_class


class GameProject:
    def __init__(self, folder):
        self.folder = os.path.abspath(folder)
        self.game = G.detect_game(self.folder)
        if self.game is None:
            raise ValueError("Not a supported MDickie game folder: %s" % folder)
        self.dir = os.path.join(self.folder, "FaceMod")
        self.orig_dir = os.path.join(self.dir, "original")
        self.faces_dir = os.path.join(self.dir, "faces")
        os.makedirs(self.faces_dir, exist_ok=True)
        self._backup()
        self.mods_path = os.path.join(self.dir, "mods.json")
        self.mods = self._load_mods()
        self._swf = None
        self._renderer = None
        self._tpl_lm = {}

    # ---- files ---------------------------------------------------------
    @property
    def swf_path(self):
        return os.path.join(self.folder, self.game.swf)

    @property
    def chars_path(self):
        return os.path.join(self.folder, "Characters.txt")

    def _backup(self):
        os.makedirs(self.orig_dir, exist_ok=True)
        for name in (self.game.swf, "Characters.txt"):
            src = os.path.join(self.folder, name)
            dst = os.path.join(self.orig_dir, name)
            if not os.path.exists(dst) and os.path.exists(src):
                shutil.copy2(src, dst)

    def _load_mods(self):
        if os.path.exists(self.mods_path):
            with open(self.mods_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"faces": [], "texts": []}

    def save_mods(self):
        tmp = self.mods_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.mods, f, indent=1)
        os.replace(tmp, self.mods_path)

    # ---- original assets -------------------------------------------------
    def swf(self):
        """Pristine SWF (parsed once)."""
        if self._swf is None:
            self._swf = S.SWF(os.path.join(self.orig_dir, self.game.swf))
            self._renderer = Renderer(self._swf)
        return self._swf

    def renderer(self):
        self.swf()
        return self._renderer

    def roster(self):
        """Current Characters.txt (names you edited by hand are kept)."""
        return G.Roster(self.game, self.chars_path)

    def head_count(self):
        sc = self.swf().symbol_classes()
        n = 0
        while head_class(n + 1) in sc.values():
            n += 1
        return n

    def head_info(self, n):
        return HeadInfo(self.swf(), n)

    def template_face(self, n):
        info = self.head_info(n)
        return self.renderer().bitmap(info.bitmap_for("face"))

    def template_landmarks(self, n):
        if n not in self._tpl_lm:
            self._tpl_lm[n] = LM.detect_upscaled(self.template_face(n))
        return self._tpl_lm[n]

    def hair_base_colour(self, hair):
        """Average colour of a hairstyle's art before tinting (for colour matching)."""
        r = self.renderer()
        info = self.head_info(1)
        sid = info.named.get("Hair")
        if not sid or hair <= 0:
            return None
        items = r.collect(sid, hair - 1)
        px = []
        for bid, *_ in items:
            a = np.asarray(r.bitmap(bid), np.float32).reshape(-1, 4)
            px.append(a[a[:, 3] > 200, :3])
        if not px or not sum(len(p) for p in px):
            return None
        return np.median(np.concatenate(px), axis=0)

    # ---- slot allocation ---------------------------------------------------
    def free_slot(self, char_index, roster=None):
        roster = roster or self.roster()
        orig = G.Roster(self.game, os.path.join(self.orig_dir, "Characters.txt"))
        used = set()
        for r in (roster, orig):
            for c in r.chars:
                if c.index != char_index:
                    used |= c.heads_all()
        for m in self.mods["faces"]:
            if m["char"] != char_index:
                used.add(m["slot"])
        for n in range(self.head_count(), 0, -1):
            if n not in used:
                return n
        return None

    # ---- face building -------------------------------------------------------
    def build(self, photos, template, job=None, mirror_ok=True):
        """photos: list of candidate dicts (from FS.analyse_photo). Returns dict of images."""
        tl = self.template_landmarks(template)
        if tl is None:
            raise RuntimeError("Couldn't find the face in template head %d" % template)
        cands = photos if mirror_ok else [p for p in photos if not p["mirrored"]]
        c = FS.pick_photo(cands, tl["yaw"], tl["pitch"])
        tpl = self.template_face(template)
        face, ratio = FS.build_face(tpl, tl, c, job)
        info = self.head_info(template)
        r = self.renderer()
        out = {"face": face, "ratio": ratio, "photo": c}
        b = info.bitmap_for("back")
        if b:
            out["back"] = FS.recolour(r.bitmap(b), ratio)
        b = info.bitmap_for("blink")
        if b:
            out["blink"] = self._blink_patch(info, b, face, ratio)
        return out

    def _blink_patch(self, info, blink_bid, face, ratio):
        """Cut the eyelid patch out of the eyes-painted-out face so blinking looks right."""
        r = self.renderer()
        fb = info.bitmap_for("face")
        mats = {}
        for bid, m, _, _ in r.collect(info.sprite_id, 2):
            mats.setdefault(bid, m)
        orig = r.bitmap(blink_bid)
        closed = face.info.get("closed")
        if closed is None or fb not in mats or blink_bid not in mats:
            return FS.recolour(orig, ratio)
        k = face.width / r.bitmap(fb).width

        def full(m):
            a, b, c, d, tx, ty = m
            return np.array([[a, c, tx], [b, d, ty], [0, 0, 1]], np.float64)

        # blink_k px -> blink px -> twips -> face px -> face_k px
        A = (np.diag([k, k, 1]) @ np.linalg.inv(full(mats[fb])) @ full(mats[blink_bid])
             @ np.diag([1 / k, 1 / k, 1]))
        W, H = int(orig.width * k), int(orig.height * k)
        patch = closed.transform((W, H), Image.AFFINE, tuple(A[:2].flatten()), Image.BICUBIC)
        a = np.asarray(orig.resize((W, H), Image.BICUBIC), np.float32)[..., 3]
        px = np.asarray(patch, np.float32)
        px[..., 3] = a
        return Image.fromarray(px.astype(np.uint8), "RGBA")

    def preview(self, images, template, hair=None, hair_rgb=None, frames=(1, 0), scale=10,
                hide_brows=True, beard=0):
        """Render the built head (with optional hairstyle) like the game would."""
        r = self.renderer()
        info = self.head_info(template)
        tw = r.bitmap(info.bitmap_for("face")).size
        ov = {}
        for role in ("face", "back", "blink"):
            b = info.bitmap_for(role)
            if b and role in images:
                src = r.bitmap(b)
                ov[b] = images[role].resize(src.size, Image.LANCZOS)
        r.overrides = ov
        tiles = []
        for fr in frames:
            r.frame_choice = {"Hat": None, "HatB": None, "Blood": None, "Mouth": 4,
                              "Eyebrows": None if hide_brows else 0,
                              "Beard": (beard - 1) if beard else None,
                              "Hair": None if not hair else ((200 + hair - 1) if fr == 0 else hair - 1)}
            items = r.collect(info.sprite_id, fr)
            if hair_rgb is not None:
                pass
            im, _ = r.render(info.sprite_id, fr, scale=scale, items=items)
            if hair and hair_rgb is not None:
                im = self._tint_hair(r, info, fr, scale, items, hair_rgb)
            tiles.append(im)
        r.overrides = {}
        r.frame_choice = {}
        W = sum(t.width for t in tiles)
        H = max(t.height for t in tiles)
        out = Image.new("RGBA", (W, H))
        x = 0
        for t in tiles:
            out.alpha_composite(t, (x, (H - t.height) // 2))
            x += t.width
        return out

    def _tint_hair(self, r, info, fr, scale, items, rgb):
        # render with hair layers multiplied by rgb (mirrors the game's ColorTransform)
        hair_bids = {b for b, m, path, _ in items if any(p[1] == "Hair" for p in path)}
        saved = dict(r.overrides)
        for b in hair_bids:
            r.overrides[b] = FS.recolour(r.bitmap(b), np.asarray(rgb, np.float32))
        im, _ = r.render(info.sprite_id, fr, scale=scale, items=items)
        r.overrides = saved
        return im

    # ---- mods --------------------------------------------------------------
    def add_face_mod(self, char_index, name, template, images, photo_paths, settings,
                     slot=None, hair=None, hair_rgb=None, beard=0, match_body=True,
                     hide_brows=True):
        roster = self.roster()
        # replace existing mod for the same character
        old = [m for m in self.mods["faces"] if m["char"] == char_index]
        if slot is None and old:
            slot = old[0]["slot"]
        for m in old:
            self.mods["faces"].remove(m)
        slot = slot or self.free_slot(char_index, roster)
        if slot is None:
            slot = roster.chars[char_index - 1].head
        mid = uuid.uuid4().hex[:8]
        d = os.path.join(self.faces_dir, mid)
        os.makedirs(d, exist_ok=True)
        for role in ("face", "back", "blink"):
            if role in images:
                images[role].save(os.path.join(d, role + ".png"))
        kept = []
        for i, p in enumerate(photo_paths):
            ext = os.path.splitext(p)[1].lower() or ".jpg"
            dst = os.path.join(d, "photo%d%s" % (i + 1, ext))
            shutil.copy2(p, dst)
            kept.append(os.path.basename(dst))
        mod = {
            "id": mid, "char": char_index, "name": name, "template": template, "slot": slot,
            "ratio": [float(x) for x in images["ratio"]], "settings": settings,
            "hair": hair, "hair_rgb": [float(x) for x in hair_rgb] if hair_rgb is not None else None,
            "beard": beard, "match_body": match_body, "hide_brows": hide_brows,
            "photos": kept, "created": time.time(),
        }
        self.mods["faces"].append(mod)
        self.save_mods()
        for m in old:
            shutil.rmtree(os.path.join(self.faces_dir, m["id"]), ignore_errors=True)
        return mod

    def remove_face_mod(self, mid):
        m = next((m for m in self.mods["faces"] if m["id"] == mid), None)
        if not m:
            return False
        self.mods["faces"].remove(m)
        self.save_mods()
        # restore that character's original look
        roster = self.roster()
        orig = G.Roster(self.game, os.path.join(self.orig_dir, "Characters.txt"))
        c, o = roster.chars[m["char"] - 1], orig.chars[m["char"] - 1]
        for (coz, limb), p in c.limb_pos.items():
            if limb in G.SKIN_LIMBS or limb in (G.HAIR, G.BEARD):
                op = o.limb_pos[(coz, limb)]
                roster.tok[p:p + 4] = orig.tok[op:op + 4]
        roster.save()
        shutil.rmtree(os.path.join(self.faces_dir, mid), ignore_errors=True)
        return True

    def apply(self, progress=None):
        """Write the game files: original SWF + all face mods + text edits; roster edits."""
        say = progress or (lambda *_: None)
        say("Loading original game data")
        swf = S.SWF(os.path.join(self.orig_dir, self.game.swf))
        roster = self.roster()
        for m in self.mods["faces"]:
            say("Installing %s into head slot %d" % (m["name"], m["slot"]))
            d = os.path.join(self.faces_dir, m["id"])
            imgs = {}
            for role in ("face", "back", "blink"):
                p = os.path.join(d, role + ".png")
                if os.path.exists(p):
                    imgs[role] = Image.open(p).convert("RGBA")
            hide = ("Eyebrows",) if m.get("hide_brows", True) else ()
            install_head(swf, m["slot"], m["template"], imgs, hide=hide)
            self._apply_roster(roster, m)
        if self.mods.get("texts"):
            say("Applying text edits")
            from . import abc as ABC
            ABC.apply_text_edits(swf, self.mods["texts"])
        say("Compressing and writing %s" % self.game.swf)
        swf.save(self.swf_path)
        roster.save()
        say("Done")

    def _apply_roster(self, roster, m):
        """Edit one character. Tints are always derived from the ORIGINAL roster so
        applying again never compounds."""
        c = roster.chars[m["char"] - 1]
        orig = G.Roster(self.game, os.path.join(self.orig_dir, "Characters.txt"))
        o = orig.chars[m["char"] - 1]
        if m.get("name"):
            c.name = m["name"]
        ratio = np.array(m["ratio"], np.float32)
        for coz in range(1, self.game.coz_count + 1):
            _, hr, hg, hb = o.get(G.HEAD, coz)
            # The face was recoloured from the raw head art by `ratio`; in the original
            # game this head was drawn at raw * old_head_tint. Shift every skin limb by
            # the same factor so the body matches the new face.
            fac = ratio / np.maximum(np.array([hr, hg, hb], np.float32), 0.05)
            for limb in G.SKIN_LIMBS:
                if limb == G.HEAD or (coz, limb) not in o.limb_pos:
                    continue
                v, r, g, b = o.get(limb, coz)
                rgb = np.array([r, g, b]) * fac if m.get("match_body", True) else (r, g, b)
                c.set(limb, rgb=np.clip(rgb, 0.05, 3.0), coz=coz)
            c.set(G.HEAD, value=m["slot"], rgb=(1, 1, 1), coz=coz)
            if m.get("hair") is not None:
                _, r, g, b = c.get(G.HAIR, coz)
                rgb = m.get("hair_rgb") or (r, g, b)
                c.set(G.HAIR, value=m["hair"], rgb=rgb, coz=coz)
            if m.get("beard") is not None:
                c.set(G.BEARD, value=m["beard"], coz=coz)

    def restore_originals(self):
        for name in (self.game.swf, "Characters.txt"):
            src = os.path.join(self.orig_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(self.folder, name))
