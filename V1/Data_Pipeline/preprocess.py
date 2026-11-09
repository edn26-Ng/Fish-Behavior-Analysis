"""
Preprocessing for the fish-behavior pipeline: frame in -> cleaned frame out,
before the detector. Returns the crop offset so boxes can be logged in
original-frame pixels.

DEFAULT IMPLEMENTATION: pass-through. Every step below needs tank footage or
camera calibration before it can be implemented and tuned. Steps, in order:
  1. Undistort / rectify  - needs checkerboard calibration through water
  2. Tank mask (circle)   - needs a real frame to set center + radius
  3. Crop to mask         - needs the mask; return the real crop offset
  4. CLAHE brightness     - enable only if testing shows it helps detection
  5. Denoise              - mainly for IR night footage
Set the tank mask BEFORE extracting labeling frames: training and inference
must use identical preprocessing.
"""

import numpy as np


def preprocess(image: np.ndarray, cfg: dict) -> tuple[np.ndarray, tuple[int, int]]:
    # DEFAULT: cfg (preprocess: section) is ignored until the steps above are implemented.
    # TODO: undistort, tank mask, crop, CLAHE, denoise (blocked on tank footage + calibration)
    return image, (0, 0)  # DEFAULT: frame unchanged, crop offset (0, 0). Return the real crop offset once crop_to_mask exists.
