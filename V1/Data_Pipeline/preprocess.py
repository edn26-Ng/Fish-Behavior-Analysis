import numpy as np
def preprocess(image: np.ndarray, cfg: dict) -> tuple[np.ndarray, tuple[int, int]]:
    # TODO: undistort, tank mask, crop, CLAHE, denoise (blocked on tank footage + calibration)
    return image, (0, 0)
