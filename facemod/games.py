"""Per-game knowledge: where files live and how Characters.txt is laid out.

Layouts were reverse engineered from each game's SaveDefaultChars() routine.
"""
import os

HEAD, HAIR, HAT, BEARD = 3, 14, 30, 31
SKIN_LIMBS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

HT_REMOVED = {16, 17, 20, 21, 19, 23, 25, 29}


class Game:
    def __init__(self, key, title, swf, exe, no_limbs, no_acc, removed, pre, post, coz_count=3,
                 yt_style=False, markers=(), hairstyles=150, beards=35):
        self.key = key
        self.title = title
        self.swf = swf
        self.exe = exe
        self.no_limbs = no_limbs
        self.no_acc = no_acc
        self.removed = removed
        self.pre = pre        # number of fields between name and the scale block (incl. name)
        self.post = post      # number of fields after the costume block
        self.coz_count = coz_count
        self.yt_style = yt_style
        self.markers = markers  # files that identify this game folder
        self.hairstyles = hairstyles
        self.beards = beards

    def limbs(self):
        return [l for l in range(1, self.no_limbs + self.no_acc + 1) if l not in self.removed]


GAMES = [
    # name gender age role gang home stat*4 | scale*14 | 3x costume | stance walk move16 gm4 gm4 attack6 crush5
    Game("hardtime", "Hard Time", "Game.swf", "Hard Time.exe", 13, 18, HT_REMOVED,
         pre=1 + 5 + 4, post=2 + 16 + 4 + 4 + 6 + 5, markers=("Hard Time.exe",), hairstyles=153, beards=35),
    # name gender age role gang class grades*16 | scale*14 | 3x costume | ... relation6
    Game("schooldays", "School Days", "Game.swf", "School Days.exe", 13, 19, HT_REMOVED,
         pre=1 + 5 + 16, post=2 + 16 + 4 + 4 + 6 + 5 + 6, markers=("School Days.exe",), hairstyles=154, beards=35),
    # name gender team stat stat | limb0 scale, then (scale?) limb r g b for limbs 1..32 | eyes move10 gm3 gm3 attack5 crush5 stance prop propB
    Game("youtestament", "The You Testament 2D", "Game.swf", "The You Testament.exe", 13, 19, set(),
         pre=1 + 4, post=1 + 10 + 3 + 3 + 5 + 5 + 3, coz_count=1, yt_style=True,
         markers=("The You Testament.exe",), hairstyles=89, beards=36),
    # name gender age role heel eyes theme light prop rel6 stat8 | scale*14 | 3x costume (no removed) | stance walk taunt4 move16 gm4 gm4 attack6 crush5
    Game("wrestlingrev", "Wrestling Revolution 2D", "Wrestling Revolution.swf", "Wrestling Revolution.exe",
         13, 18, set(), pre=1 + 8 + 6 + 8, post=2 + 4 + 16 + 4 + 4 + 6 + 5,
         markers=("Wrestling Revolution.exe",), hairstyles=153, beards=35),
]


def detect_game(folder):
    for g in GAMES:
        if any(os.path.exists(os.path.join(folder, m)) for m in g.markers):
            return g
    # fall back on swf name
    for g in GAMES:
        if os.path.exists(os.path.join(folder, g.swf)) and g.key == "wrestlingrev":
            return g
    return None


class Character:
    """A view onto one record in the token list. Edits go straight into the tokens."""

    def __init__(self, roster, index, start):
        self.roster = roster
        self.index = index  # 1-based like the game
        self.start = start
        self.limb_pos = {}  # (coz, limb) -> token index of the limb value (r,g,b follow)

    @property
    def name(self):
        return self.roster.tok[self.start]

    @name.setter
    def name(self, v):
        v = v.replace("|", "/").strip()
        self.roster.tok[self.start] = v

    def get(self, limb, coz=1):
        p = self.limb_pos[(coz, limb)]
        t = self.roster.tok
        return int(float(t[p])), float(t[p + 1]), float(t[p + 2]), float(t[p + 3])

    def set(self, limb, value=None, rgb=None, coz=None):
        cozs = [coz] if coz else range(1, self.roster.game.coz_count + 1)
        t = self.roster.tok
        for c in cozs:
            p = self.limb_pos.get((c, limb))
            if p is None:
                continue
            if value is not None:
                t[p] = str(int(value))
            if rgb is not None:
                for i in range(3):
                    t[p + 1 + i] = _fmt(rgb[i])

    @property
    def head(self):
        return self.get(HEAD)[0]

    def heads_all(self):
        return {self.get(HEAD, c)[0] for c in range(1, self.roster.game.coz_count + 1)}


def _fmt(x):
    x = round(float(x), 2)
    return str(int(x)) if x == int(x) else repr(x)


class Roster:
    def __init__(self, game, path):
        self.game = game
        self.path = path
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            self.raw = f.read()
        self.tok = self.raw.split("|")
        self.chars = []
        self._parse()

    def _parse(self):
        g = self.game
        t = self.tok
        n = int(t[0])
        p = 1
        for i in range(1, n + 1):
            if p >= len(t) - 1:
                break
            ch = Character(self, i, p)
            if g.yt_style:
                p += g.pre
                for limb in range(0, g.no_limbs + g.no_acc + 1):
                    if limb <= g.no_limbs:
                        p += 1  # scale
                    if limb > 0:
                        ch.limb_pos[(1, limb)] = p
                        p += 4
                p += g.post
            else:
                p += g.pre + g.no_limbs + 1
                for coz in range(1, g.coz_count + 1):
                    for limb in g.limbs():
                        ch.limb_pos[(coz, limb)] = p
                        p += 4
                p += g.post
            self.chars.append(ch)
        self.end = p

    def validate(self):
        """Sanity check: head values should be small positive ints and names non-numeric."""
        bad = 0
        for c in self.chars:
            try:
                h = c.head
                if not (0 <= h <= 400):
                    bad += 1
            except Exception:
                bad += 1
        return bad == 0 and self.end <= len(self.tok)

    def head_usage(self):
        use = {}
        for c in self.chars:
            for h in c.heads_all():
                use.setdefault(h, []).append(c.index)
        return use

    def save(self, path=None):
        path = path or self.path
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write("|".join(self.tok))
        os.replace(tmp, path)
