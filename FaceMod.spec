# PyInstaller spec - builds a single FaceMod.exe.  Run:  build.bat
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = [
    ("facemod/web", "facemod/web"),
    ("facemod/models/face_landmarker.task", "facemod/models"),
]
datas += collect_data_files("mediapipe", includes=["**/*.binarypb", "**/*.tflite", "**/*.txt"])
binaries = collect_dynamic_libs("mediapipe")

a = Analysis(
    ["FaceMod.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=["webview.platforms.edgechromium", "mediapipe.tasks.python.vision"],
    excludes=["torch", "tensorflow", "scipy", "pandas", "tkinter", "IPython",
              "jax", "sounddevice", "PyQt5", "PyQt6", "PySide6"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="FaceMod",
    icon="assets/icon.ico",
    console=False,
    upx=False,
)
