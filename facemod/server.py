"""Local web UI backend. Run:  python -m facemod   (opens http://127.0.0.1:8765)

Everything runs on this machine; no network calls are made.
"""
import base64
import io
import json
import os
import re
import subprocess
import sys
import threading
import traceback
import uuid
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image

from . import games as G
from . import faceswap as FS
from .project import GameProject

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = getattr(sys, "frozen", False)
if FROZEN:
    # single-file exe: look for games next to the exe, keep settings in AppData
    ROOT = os.environ.get("FACEMOD_ROOT") or os.path.dirname(sys.executable)
    DATA = os.path.join(os.environ.get("APPDATA", ROOT), "MDickieFaceMod")
else:
    ROOT = os.environ.get("FACEMOD_ROOT") or os.path.dirname(HERE)
    DATA = os.path.dirname(HERE)
os.makedirs(DATA, exist_ok=True)
WEB = os.path.join(HERE, "web")
CONFIG = os.path.join(DATA, "facemod_config.json")
UPLOADS = os.path.join(DATA, "facemod_uploads")
NATIVE = {"pick_folder": None, "pick_files": None}  # filled in by the desktop window

LOCK = threading.RLock()
PROJECTS = {}      # key -> GameProject
PHOTOSETS = {}     # id -> {"paths": [...], "cands": [...]}
BUILDS = {}        # build id -> dict(images, template, game, ...)
THUMBS = {}


def load_config():
    if os.path.exists(CONFIG):
        with open(CONFIG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"folders": []}


def save_config(cfg):
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)


def scan_games():
    cfg = load_config()
    cands = [ROOT] + [os.path.join(ROOT, d) for d in os.listdir(ROOT)] + cfg.get("folders", [])
    out = []
    seen = set()
    for d in cands:
        d = os.path.abspath(d)
        if d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        g = G.detect_game(d)
        if g:
            out.append({"key": _key(d), "title": g.title, "folder": d, "game": g.key})
    return out


def _key(folder):
    return re.sub(r"[^a-z0-9]+", "-", os.path.basename(folder).lower()).strip("-")


def project(key):
    with LOCK:
        if key not in PROJECTS:
            for g in scan_games():
                if g["key"] == key:
                    PROJECTS[key] = GameProject(g["folder"])
                    break
            else:
                raise KeyError("unknown game " + key)
        return PROJECTS[key]


def png_bytes(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def data_url(img):
    return "data:image/png;base64," + base64.b64encode(png_bytes(img)).decode()


def thumb(img, size=160):
    im = img.copy()
    im.thumbnail((size, size))
    return im


# ---- API handlers ----------------------------------------------------------

def api_games(q, body):
    return {"games": scan_games(), "root": ROOT}


def api_native(q, body):
    return {"folder": bool(NATIVE["pick_folder"]), "files": bool(NATIVE["pick_files"])}


def api_pick(q, body):
    """Open a native folder/file dialog (desktop build only)."""
    kind = body.get("kind", "folder")
    fn = NATIVE["pick_folder"] if kind == "folder" else NATIVE["pick_files"]
    if not fn:
        raise ValueError("Native dialogs are only available in the desktop app")
    return {"paths": fn() or []}


def api_add_folder(q, body):
    path = body["path"].strip().strip('"')
    if not G.detect_game(path):
        raise ValueError("No supported MDickie game found in that folder")
    cfg = load_config()
    if path not in cfg["folders"]:
        cfg["folders"].append(path)
        save_config(cfg)
    return api_games(q, body)


def api_chars(q, body):
    p = project(q["game"])
    r = p.roster()
    modded = {m["char"]: m for m in p.mods["faces"]}
    usage = r.head_usage()
    chars = [{"index": c.index, "name": c.name, "head": c.head,
              "shared": len(usage.get(c.head, [])), "mod": modded.get(c.index, {}).get("id")}
             for c in r.chars]
    return {"chars": chars, "heads": p.head_count(), "title": p.game.title,
            "hairstyles": p.game.hairstyles, "beards": p.game.beards,
            "mods": p.mods["faces"], "texts": p.mods.get("texts", []), "folder": p.folder}


def api_photos(q, body):
    """Upload photos (base64) of one person; detect faces in each and its mirror."""
    pid = uuid.uuid4().hex[:10]
    d = os.path.join(UPLOADS, pid)
    os.makedirs(d, exist_ok=True)
    paths, cands, info = [], [], []
    for i, f in enumerate(body["files"]):
        raw = base64.b64decode(f["data"].split(",", 1)[-1])
        ext = os.path.splitext(f.get("name", ""))[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            ext = ".jpg"
        path = os.path.join(d, "%d%s" % (i + 1, ext))
        with open(path, "wb") as fh:
            fh.write(raw)
        im = FS.load_photo(path)
        cs = FS.analyse_photo(im)
        for c in cs:
            c["src"] = i
        ok = bool(cs)
        info.append({"name": f.get("name"), "ok": ok,
                     "yaw": round(cs[0]["yaw"]) if ok else None,
                     "pitch": round(cs[0]["pitch"]) if ok else None,
                     "thumb": data_url(thumb(im, 140))})
        if ok:
            paths.append(path)
            cands += cs
    if not cands:
        raise ValueError("Couldn't find a face in any of those photos")
    PHOTOSETS[pid] = {"paths": paths, "cands": cands}
    return {"id": pid, "photos": info}


def api_photos_path(q, body):
    """Same as api_photos but reads images from a folder (or file) path on this PC."""
    files = []
    for raw in body["paths"]:
        p = raw.strip().strip('"')
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p))
                      if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))]
        elif os.path.isfile(p):
            files.append(p)
    if not files:
        raise ValueError("No images found at that path")
    enc = []
    for f in files[:24]:
        im = FS.load_photo(f)  # downsizes huge camera shots before sending around
        b = io.BytesIO()
        im.save(b, "JPEG", quality=92)
        enc.append({"name": os.path.basename(f),
                    "data": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()})
    res = api_photos(q, {"files": enc})
    res["files"] = enc
    return res


def _job(s):
    s = s or {}
    return FS.FaceJob(
        scale=int(s.get("scale", 3)), feather=float(s.get("feather", 0.08)),
        shrink=float(s.get("shrink", 0.07)), forehead_drop=float(s.get("forehead", 0.3)),
        tone=s.get("tone", "photo"), strength=float(s.get("strength", 1.0)),
        saturation=float(s.get("saturation", 1.0)), contrast=float(s.get("contrast", 1.0)),
        brightness=float(s.get("brightness", 1.0)))


def _hair_tint(p, cand, hair):
    """Tint multipliers that make game hairstyle `hair` look like the photo's hair."""
    want = FS.sample_hair(cand)
    base = p.hair_base_colour(hair)
    if want is None or base is None:
        return None
    return [float(x) for x in np.clip((want + 4) / (base + 4), 0.05, 3.0)]


def api_build(q, body):
    p = project(body["game"])
    ps = PHOTOSETS[body["photos"]]
    template = int(body["template"])
    with LOCK:
        imgs = p.build(ps["cands"], template, _job(body.get("settings")),
                       mirror_ok=body.get("mirror", True))
        hair = body.get("hair")
        hair = int(hair) if hair not in (None, "", "keep") else None
        hair_rgb = body.get("hair_rgb")
        if hair and body.get("hair_auto", True):
            hair_rgb = _hair_tint(p, imgs["photo"], hair)
        beard = int(body.get("beard") or 0)
        prev_hair = hair
        if hair is None and body.get("char"):
            c = p.roster().chars[int(body["char"]) - 1]
            v, r, g, b = c.get(G.HAIR)
            prev_hair, hair_rgb = v, hair_rgb or [r, g, b]
        prev = p.preview(imgs, template, hair=prev_hair, hair_rgb=hair_rgb, scale=9,
                         hide_brows=body.get("hide_brows", True), beard=beard)
    bid = uuid.uuid4().hex[:10]
    BUILDS[bid] = {"images": imgs, "template": template, "game": body["game"],
                   "photos": body["photos"], "settings": body.get("settings") or {},
                   "hair": hair, "hair_rgb": hair_rgb, "beard": beard}
    c = imgs["photo"]
    return {"build": bid, "preview": data_url(prev), "face": data_url(imgs["face"]),
            "used_photo": c.get("src"), "mirrored": c["mirrored"], "yaw": round(c["yaw"]),
            "hair_rgb": hair_rgb}


def api_install(q, body):
    b = BUILDS[body["build"]]
    p = project(b["game"])
    char = int(body["char"])
    with LOCK:
        mod = p.add_face_mod(char, body.get("name", "").strip(), b["template"], b["images"],
                             PHOTOSETS[b["photos"]]["paths"], b["settings"],
                             hair=b["hair"], hair_rgb=b["hair_rgb"], beard=b["beard"],
                             match_body=body.get("match_body", True),
                             hide_brows=body.get("hide_brows", True))
        p.apply()
    return {"ok": True, "mod": mod}


def api_remove(q, body):
    p = project(body["game"])
    with LOCK:
        p.remove_face_mod(body["id"])
        p.apply()
    return {"ok": True}


def api_rename(q, body):
    p = project(body["game"])
    with LOCK:
        r = p.roster()
        for e in body["names"]:
            i = int(e["index"])
            if 1 <= i <= len(r.chars) and e.get("name", "").strip():
                r.chars[i - 1].name = e["name"]
                for m in p.mods["faces"]:
                    if m["char"] == i:
                        m["name"] = e["name"].strip()
        r.save()
        p.save_mods()
    return {"ok": True}


def api_strings(q, body):
    p = project(q["game"])
    from . import abc as ABC
    with LOCK:
        res = ABC.list_strings(p.swf(), q.get("q", ""), limit=400)
    return {"strings": [{"text": s["text"]} for s in res]}


def api_texts(q, body):
    p = project(body["game"])
    with LOCK:
        p.mods["texts"] = [e for e in body["edits"] if e.get("find")]
        p.save_mods()
        p.apply()
    return {"ok": True}


def api_apply(q, body):
    p = project(body["game"])
    with LOCK:
        p.apply()
    return {"ok": True}


def api_restore(q, body):
    p = project(body["game"])
    with LOCK:
        p.restore_originals()
        p.mods = {"faces": [], "texts": []}
        p.save_mods()
    return {"ok": True}


def api_launch(q, body):
    p = project(body["game"])
    exe = os.path.join(p.folder, p.game.exe)
    subprocess.Popen([exe], cwd=p.folder)
    return {"ok": True}


def img_head(q):
    p = project(q["game"])
    n = int(q["n"])
    key = ("head", q["game"], n)
    if key not in THUMBS:
        with LOCK:
            r = p.renderer()
            info = p.head_info(n)
            r.frame_choice = {"Hat": None, "HatB": None, "Blood": None, "Mouth": 4, "Eyebrows": None,
                              "Beard": None, "Hair": None}
            im, _ = r.render(info.sprite_id, 1, scale=5)
            r.frame_choice = {}
        THUMBS[key] = png_bytes(im)
    return THUMBS[key]


def img_hair(q):
    p = project(q["game"])
    n = int(q["n"])
    key = ("hair", q["game"], n)
    if key not in THUMBS:
        with LOCK:
            r = p.renderer()
            info = p.head_info(1)
            r.frame_choice = {"Hat": None, "HatB": None, "Blood": None, "Mouth": 4, "Eyebrows": None,
                              "Beard": None, "Hair": n - 1}
            im, _ = r.render(info.sprite_id, 1, scale=3)
            r.frame_choice = {}
        THUMBS[key] = png_bytes(im)
    return THUMBS[key]


def img_mod(q):
    p = project(q["game"])
    path = os.path.join(p.faces_dir, os.path.basename(q["id"]), "face.png")
    with open(path, "rb") as f:
        return f.read()


ROUTES = {
    ("GET", "/api/games"): api_games,
    ("POST", "/api/folder"): api_add_folder,
    ("GET", "/api/native"): api_native,
    ("POST", "/api/pick"): api_pick,
    ("GET", "/api/chars"): api_chars,
    ("POST", "/api/photos"): api_photos,
    ("POST", "/api/photos_path"): api_photos_path,
    ("POST", "/api/build"): api_build,
    ("POST", "/api/install"): api_install,
    ("POST", "/api/remove"): api_remove,
    ("POST", "/api/rename"): api_rename,
    ("GET", "/api/strings"): api_strings,
    ("POST", "/api/texts"): api_texts,
    ("POST", "/api/apply"): api_apply,
    ("POST", "/api/restore"): api_restore,
    ("POST", "/api/launch"): api_launch,
}
IMAGES = {"/img/head": img_head, "/img/hair": img_hair, "/img/mod": img_mod}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, data, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _handle(self, method):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if method == "GET" and u.path in IMAGES:
                return self._send(200, IMAGES[u.path](q), "image/png")
            fn = ROUTES.get((method, u.path))
            if fn:
                body = {}
                if method == "POST":
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(n) or b"{}")
                res = fn(q, body)
                return self._send(200, json.dumps(res).encode(), "application/json")
            if method == "GET":
                name = "index.html" if u.path in ("/", "") else u.path.lstrip("/")
                path = os.path.normpath(os.path.join(WEB, name))
                if path.startswith(WEB) and os.path.isfile(path):
                    ctype = {"html": "text/html", "js": "text/javascript", "css": "text/css",
                             "png": "image/png", "svg": "image/svg+xml", "woff2": "font/woff2"}.get(path.rsplit(".", 1)[-1],
                                                                             "application/octet-stream")
                    with open(path, "rb") as f:
                        return self._send(200, f.read(), ctype + "; charset=utf-8")
            self._send(404, b"not found", "text/plain")
        except Exception as e:
            traceback.print_exc()
            self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")


def main(port=8765, open_browser=True):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = "http://127.0.0.1:%d/" % port
    print("MDickie Face Mod running at", url, "(Ctrl+C to quit)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
