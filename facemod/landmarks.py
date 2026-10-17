"""Local face landmark detection (MediaPipe Face Landmarker, runs fully offline)."""
import os
import numpy as np
from PIL import Image

MODEL = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")
_det = None


def _detector():
    global _det
    if _det is None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL),
            num_faces=1,
            min_face_detection_confidence=0.2,
            min_face_presence_confidence=0.2,
            output_facial_transformation_matrixes=True,
        )
        _det = vision.FaceLandmarker.create_from_options(opts)
    return _det


def detect(img, max_side=1600):
    """img: PIL image. Returns dict(pts=Nx3 array in pixel coords of img, yaw=deg) or None.

    z is in the same scale as x (pixels), negative = toward camera.
    """
    import mediapipe as mp
    im = img.convert("RGB")
    s = 1.0
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    arr = np.ascontiguousarray(np.array(im))
    res = _detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    if not res.face_landmarks:
        return None
    w, h = im.size
    pts = np.array([(p.x * w, p.y * h, p.z * w) for p in res.face_landmarks[0]], np.float64) / s
    yaw = pitch = 0.0
    if res.facial_transformation_matrixes:
        R = np.array(res.facial_transformation_matrixes[0])[:3, :3]
        yaw = float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))
        pitch = float(np.degrees(np.arcsin(-np.clip(R[1, 2], -1, 1))))
    return {"pts": pts, "yaw": yaw, "pitch": pitch}


def detect_upscaled(img, factor=6, pad=0.35):
    """For tiny game sprites: put on a neutral background with padding, upscale, detect."""
    im = img.convert("RGBA")
    w, h = im.size
    pw, ph = int(w * pad), int(h * pad)
    bg = Image.new("RGBA", (w + 2 * pw, h + 2 * ph), (128, 128, 128, 255))
    bg.alpha_composite(im, (pw, ph))
    big = bg.convert("RGB").resize((bg.width * factor, bg.height * factor), Image.BICUBIC)
    r = detect(big)
    if r is None:
        return None
    r["pts"] = r["pts"] / factor - np.array([pw, ph, 0.0])
    r["pts"][:, 2] = r["pts"][:, 2]
    return r
