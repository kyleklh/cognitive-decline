import numpy as np
import math

def exit_center(exit_region):
    x1, y1, x2, y2 = exit_region

    x_min = min(x1, x2)
    x_max = max(x1, x2)

    y_min = min(y1, y2)
    y_max = max(y1, y2)

    return (x_min + x_max) / 2, (y_min + y_max) / 2


def mask_centroid(mask):
    ys, xs = np.where(mask > 0.5)

    if len(xs) == 0:
        return None
    
    cx = float(xs.mean())
    cy = float(ys.mean())

    return cx, cy

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)