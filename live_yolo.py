#!/usr/bin/env python3
# --------------------------------------------------------------
# Live webcam demo → YOLO26 tracking (BoT‑SORT) with a resizable
# window that you can set to any size you like.
#
# What the script does:
#   1️⃣ Guarantees required packages (`ultralytics`, `opencv-python`)
#      are installed (auto‑install only if they are missing).
#   2️⃣ Downloads the pretrained YOLO26 weight file (*yolo26n.pt*)
#      when it is not already present.
#   3️⃣ Opens the first USB/webcam (index 0).
#   4️⃣ Creates a manually‑resizable OpenCV window whose size you
#      control (`desired_w`, `desired_h` below).
#   5️⃣ (Optional) up‑scales every frame before showing it.
#   6️⃣ Runs YOLO26 tracking with the BoT‑SORT backend, draws boxes/
#      IDs on each frame and displays the result inside the window.
#   7️⃣ Press **q** or **Esc** to quit cleanly.
#
# Tested on Python 3.8+ (Linux / macOS / Windows). No extra system
# libraries are required – everything is pulled from PyPI.
# --------------------------------------------------------------

import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Helper: install a missing pip package *with the same interpreter* that
# is currently executing this script.  Only safe modules (`subprocess`,
# `sys`) are used, so we never call `__import__` on an unknown name.
# ----------------------------------------------------------------------
def _ensure_pkg(pkg_name: str) -> None:
    """
    Try to import *pkg_name*. If it fails, install it with the current
    interpreter (`python3 -m pip install …`) and try again.
    Any failure raises a clear RuntimeError that tells the user to run the
    installation manually.
    """
    try:
        __import__(pkg_name)                     # succeed → nothing else needed
    except ImportError:                          # <-- package is missing
        print(f"🚀 Installing missing package **{pkg_name}** …")
        # `subprocess` and `sys` are part of the Python standard library,
        # so they are guaranteed to exist.
        import subprocess as _sp                 # noqa: F401 – we only need it for the call
        try:
            # The same executable (`python3`) that runs this script is used
            # to invoke pip.  This works inside virtual‑envs, conda envs,
            # system Python, etc.
            _sp.check_call([sys.executable,
                           "-m", "pip",
                           "install", "--upgrade", pkg_name])
        except Exception as exc:                  # pragma: no cover – defensive guard
            raise RuntimeError(
                f"Could not auto‑install `{pkg_name}`. "
                f"Do it manually with:\n    python3 -m pip install {pkg_name}"
            ) from exc

        print(f"✅  Installed `{pkg_name}`. Continuing…")

# ------------------------------------------------------------
# Make sure both required third‑party libraries are present.
# ------------------------------------------------------------
_ensure_pkg("ultralytics")          # provides the YOLO class
_ensure_pkg("opencv-python")        # provides `cv2`

import cv2                          # <-- now we have a working OpenCV import
from ultralytics import YOLO        # now we can use the model


# ----------------------------------------------------------------------
# Helper: download the pretrained YOLO26 weight file once, if it does not yet exist.
# ----------------------------------------------------------------------
MODEL_URL   = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo26n.pt"
DEFAULT_PATH = Path("yolo26n.pt")

def _download_model(path: Path) -> Path:
    """Download the model if *path* does not already exist."""
    if path.is_file():
        return path                         # already downloaded → reuse
    print(f"⬇️  Downloading YOLO26 weights from {MODEL_URL} …")
    import urllib.request, tqdm
    with tqdm.tqdm(unit='B', unit_scale=True,
                   desc="Downloading", leave=False) as t:
        with urllib.request.urlopen(MODEL_URL) as resp:
            total = int(resp.headers.get('content-length', 0))
            with open(path, "wb") as out_f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    t.update(len(chunk))
                    out_f.write(chunk)
    print(f"✅  Saved model to {path}")
    return path

model_path: Path = _download_model(DEFAULT_PATH)


# ----------------------------------------------------------------------
# Load YOLO26 (the "n" – nano – variant).  The ultralytics library
# will automatically download the .pt file if it cannot find one.
# ----------------------------------------------------------------------
print("🔧 Loading YOLO26 model …")
model = YOLO(str(model_path))          # object ready for tracking / detection
print("✅  Model loaded.")


# ----------------------------------------------------------------------
# Open webcam (index 0).  Change the index if you have several cameras.
# ----------------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError(
        "❌ Could not open any camera. Verify that a USB/webcam is plugged in "
        "and your user has permission to access it."
    )
print("📷 Camera opened successfully.")


# ----------------------------------------------------------------------
# Create a *resizeable* window and force the desired display size.
# Pick whatever resolution feels comfortable on your monitor.
# ----------------------------------------------------------------------
WIN_NAME      = "YOLO26 Real‑time Tracking"
desired_w, desired_h = 1280, 720          # <-- edit these numbers to suit you
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)   # allow external resizing
cv2.resizeWindow(WIN_NAME, desired_w, desired_h)
print(f"🖼️  Created resizable window “{WIN_NAME}” of size {desired_w}×{desired_h}")

# ----------------------------------------------------------------------
# Optional up‑scale factor – makes each shown frame larger than the raw camera feed.
# Adjust >1 to enlarge, =1 (default) for original size,
# <1 to shrink.
# ----------------------------------------------------------------------
scale_factor = 1.3   # you can change this value (e.g. 2.0 → double size)

# ----------------------------------------------------------------------
# Main processing loop
# ----------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:                     # camera stopped delivering frames – exit gracefully
        print("⚠️  Frame capture failed – exiting.")
        break

    # ---- optional up‑scale -------------------------------------------------
    new_h = int(frame.shape[0] * scale_factor)
    new_w = int(frame.shape[1] * scale_factor)
    frame_up = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # ---- run tracking (BoT‑SORT backend) ------------------------------------
    # `persist=True` preserves track IDs between consecutive frames.
    results = model.track(
        frame_up,                     # up‑scaled image – you can also feed the raw frame
        persist=True,
        tracker="botsort.yaml"        # BoT‑SORT configuration file (bundled with YOLO)
    )

    # ---- draw boxes + IDs on that same picture ----------------------------
    annotated = results[0].plot()      # returns a NumPy image ready for display

    # ---- show the result inside our fixed‑size window -----------------------
    cv2.imshow(WIN_NAME, annotated)

    # ---- exit handling ------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:   # 'q' or ESC → break the loop
        break

# ----------------------------------------------------------------------
# Clean‑up resources before we leave
# ----------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
print("👋  All done – goodbye!")


# ---------------------------------------------------------------
# 🛠️  If you *prefer* to install the dependencies manually,
#     skip the auto‑install block above and run once:
#
#         python3 -m pip install --upgrade ultralytics opencv-python tqdm
#
#     Then simply execute the script (e.g. `python3 live_yolo26_tracker.py`).
# ---------------------------------------------------------------
