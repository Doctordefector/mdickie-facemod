# MDickie FaceMod

Put your friends into MDickie's 2D AIR games — their **faces, names and in-game text** — from a few photos.
Runs 100% locally (no APIs, no uploads).

Supported (desktop AIR versions):

| Game | SWF | Head slots | Characters |
|---|---|---|---|
| Hard Time | `Game.swf` | 103 | 150 |
| School Days | `Game.swf` | 104 | 150 |
| The You Testament 2D | `Game.swf` | 120 | 200 |
| Wrestling Revolution 2D | `Wrestling Revolution.swf` | 116 | 350 |

## Using it

1. Download `FaceMod.exe` from Releases (or build it, below).
2. Put it next to your game folders (or inside one), or use **+ Game folder** in the app.
3. **Faces** tab: pick a character → drop in photos of your friend (or paste a folder path) → tweak → **Put them in the game**.
   - Give it a few angles. The games draw faces turned ~25° to the right; the app picks (or mirrors) the closest shot automatically.
   - Pick a game hairstyle and it gets coloured from the photo; skin tone of the whole body is matched to the face.
4. **Names** tab: rename anyone, or bulk‑paste a list of names.
5. **Text** tab: search any dialogue/menu/move text and replace it ("Officer" → "Sergeant", a gang or school name, …).
6. **Installed mods**: remove a face, re-apply, or restore the original files.

Originals are backed up to `<game>/FaceMod/original/` the first time, and every apply rebuilds from those, so nothing is ever stacked or lost.
If you have a save in progress, use the game's restore-characters option (or start fresh) to see roster changes — saves keep their own copy of the roster.

## Building

Needs Python 3.10+ on Windows.

```
build.bat
```

→ `dist\FaceMod.exe` (single file). For development: `pip install -r requirements.txt` then `python FaceMod.py` (native window) or `python -m facemod` (browser).

## How the games store faces (reverse-engineering notes)

All four are AS3 SWFs on the Adobe AIR captive runtime; the runtime does **not** verify `META-INF/signatures.xml`, so edited files load fine.

**Heads.** Each head is an exported `MovieClip` symbol `Head01…HeadNN` (`DefineSprite`) with 3 frames:

| Frame | Shows | Contents |
|---|---|---|
| 1 | back of head (facing away) | one shape with a bitmap fill (~115×155 px) |
| 2 | face, 3/4 view turned ~25° right | one shape with a bitmap fill (62×84 … 126×172 px) + named child clips `Mouth`, `Eyebrows`, `Blood`, `Beard`, `Hair`, `Hat` (+`HatB` in You Testament) |
| 3 | blink / eyes shut | frame 2, sometimes plus a small eyelid patch bitmap |

Bitmaps are `DefineBitsLossless2` (premultiplied ARGB) drawn through a fill matrix that scales them down ~3.6–6.6× (the art is supersampled). Back-of-head and eyelid bitmaps are **shared** across many heads, so the modder never edits in place: it clones the template head's timeline into the target slot with fresh bitmaps/shapes.
`Mouth`/`Eyebrows` are shared expression overlays — frame 5 is empty (neutral), the code calls `gotoAndStop(5 - expression)`.
The `Hair` clip has frames 1…N for front views and 201…N+200 for the back view; `Hat` uses 100+n for the back, `Beard` frame 100 for the back.

**Which head a character uses.** `limbCostume[3]` in the game code is `[0, Head01, Head02, …]`, so head value *n* → `HeadNN`. The per-character value lives in `Characters.txt`.

**Characters.txt** is one line of `|`-separated fields: `count|` then per character (from each game's `SaveDefaultChars()`):

- Hard Time: `name gender age role gang home stat×4 | scale×14 | 3 costumes × limbs 1–31 (skipping 16,17,19,20,21,23,25,29) × (value r g b) | stance walk move×16 ground×4 ground×4 attack×6 crush×5`
- School Days: `name gender age role gang class grade×16 | scale×14 | 3 costumes × limbs 1–32 (same skips) × 4 | stance walk move×16 ground×4 ground×4 attack×6 crush×5 relation×6`
- Wrestling Revolution: `name gender age role heel eyes theme light prop relationship×6 stat×8 | scale×14 | 3 costumes × limbs 1–31 × 4 | stance walk taunt×4 move×16 ground×4 ground×4 attack×6 crush×5`
- You Testament: `name gender team stat stat | for limb 0–32: (scale if limb≤13) (value r g b if limb≥1) | eyes move×10 ground×3 ground×3 attack×5 crush×5 stance prop propB` (one costume)

Limb ids: 1 hips, 2 body, **3 head**, 4–7 legs, 8–13 arms, **14 hair**, **30 hat**, **31 beard** (32 = HatB in You Testament). `r g b` are `ColorTransform` multipliers; accessory colours are divided by their parent limb's tint, so setting the head tint to 1 doesn't disturb hair colour.

**Text** is in the `DoABC` string pool. Only literals are edited — strings referenced as namespaces/multinames (identifiers) are protected.

## How a face is made

1. MediaPipe Face Landmarker (local `.task` model) finds 478 landmarks on each photo, its mirror, and on the game's face bitmap (upscaled; detects all heads).
2. Pick the photo whose yaw/pitch is closest to the game face.
3. Warp the photo onto the game face with the 854-triangle face mesh (plus fans for the eye/mouth holes), painted back-to-front by depth so the far cheek tucks away correctly.
4. Skin ratio (photo ÷ game skin) recolours the rest of the head, the back-of-head bitmap and the body limb tints.
5. Two-band blend inside the face oval (wide for lighting, narrow for detail) hides the seam. The result is written at 3× resolution.

The `facemod/` package can also be used as a library (`swf.py` SWF reader/writer, `render.py` sprite renderer, `abc.py` string-pool editor, `games.py` roster parser).

## Disclaimer

Fan tool, not affiliated with MDickie. Don't redistribute game files. Only mod people who are cool with it.

## Third-party

Uses [MediaPipe](https://github.com/google-ai-edge/mediapipe) and its Face Landmarker model (Apache 2.0), OpenCV, NumPy, Pillow and pywebview. Bundles the Space Mono and Inter fonts (SIL Open Font License).
