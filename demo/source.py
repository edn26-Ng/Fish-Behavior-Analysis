# Video source abstraction — swap webcam for Pi camera or recorded file
# without changing the caller.

import cv2


# Wraps a single webcam as a reusable source object.
# Exposes read(), release(), and context-manager support so callers
# never interact with cv2.VideoCapture directly.
class WebcamSource:
    # Opens the webcam at the given index and requests the target resolution
    # and FPS. The driver may not honor all values; actual_fps reflects what
    # was granted.
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        # CAP_DSHOW (DirectShow) is more reliable than the default MSMF backend on Windows.
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam at index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    # Queries the driver for the FPS it actually granted after __init__ set it.
    @property
    def actual_fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)

    # Returns (success, frame), mirroring cv2.VideoCapture.read() so callers
    # can be swapped to any other source without changing their read logic.
    def read(self):
        return self.cap.read()

    # Frees the webcam hardware handle; called automatically via __exit__.
    def release(self):
        self.cap.release()

    # Allows use as a context manager (with WebcamSource(...) as src:).
    def __enter__(self):
        return self

    # Guarantees release() runs on exit, even if the body raises an exception.
    def __exit__(self, *_):
        self.release()


# Reads the source type from the config dict and returns the matching source
# object. Adding a new source (file, Picamera2) means adding a branch here;
# no other file needs to change.
def build_source(cfg: dict):
    src_type = cfg.get("type", "webcam")
    if src_type == "webcam":
        return WebcamSource(
            index=cfg.get("webcam_index", 0),
            width=cfg.get("width", 1280),
            height=cfg.get("height", 720),
            fps=cfg.get("fps", 30),
        )
    raise ValueError(f"Unknown source type: {src_type!r}")
