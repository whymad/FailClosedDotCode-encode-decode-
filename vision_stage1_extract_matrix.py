
# vision_stage1_extract_matrix.py
#
# ============================================================
# RuggedDot-11 — Vision Stage 1
#   Fiducials (3 solid L corners) -> normalize (warp) -> 11x11 matrix
# ============================================================
#
# OUTPUT:
#   Prints ONLY the 11x11 matrix:
#     1 = dot present, 0 = empty
#
# This script intentionally does NOT do ECC/CRC decoding.
#
# Dependencies:
#   pip install opencv-python numpy
#
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List

import cv2
import numpy as np


# =========================
# USER SETTINGS (edit here)
# =========================
IMAGE_PATH = "ruggeddot11_111.png"   # <-- change to your filename

GRID_N = 11

# This is the "normalized" data-square size after warp.
# Larger => more stable sampling (recommended 500~1000).
WARP_SIZE = 660

# IMPORTANT:
# This must match your encoder design (RenderConfig.fiducial_gap_cells).
# If you change the encoder gap, update this too.
FID_GAP_CELLS = 0.8

# Disk sampling radius relative to one cell size in the normalized square
SAMPLE_RADIUS_RATIO = 0.30

# Morphology closing kernel (solidify L fiducials)
MORPH_CLOSE_K = 5  # try 3~9 if needed


# =========================
# Internal helpers
# =========================

@dataclass
class Component:
    label: int
    area: int
    bbox: Tuple[int, int, int, int]       # x,y,w,h
    centroid: Tuple[float, float]         # cx,cy


def otsu_binary_foreground(gray: np.ndarray) -> np.ndarray:
    """
    Otsu threshold, returning a binary image where:
      foreground (black ink / dots / fiducials) = 255
      background = 0
    We use THRESH_BINARY_INV because your tags are black on white.
    """
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def morph_close(bw: np.ndarray, k: int) -> np.ndarray:
    """
    Morphological closing (dilation then erosion) to make fiducials more solid.
    This helps when there are small holes or anti-alias edges.
    """
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)


def get_components(bw: np.ndarray) -> List[Component]:
    """
    Connected components on a binary (0/255) image.
    Returns a list of components excluding background label 0.
    """
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)

    comps: List[Component] = []
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[lab]
        comps.append(Component(lab, area, (x, y, w, h), (float(cx), float(cy))))
    return comps


def pick_fiducials_three_corners(comps: List[Component], img_w: int, img_h: int) -> Optional[Tuple[Component, Component, Component]]:
    """
    Pick TL/TR/BL fiducials.

    Key assumption for your current design:
      - Fiducials are the largest connected components (big solid L blocks),
        much larger than circular dots.

    We select candidate components by area first, then assign by centroid position.
    """
    if not comps:
        return None

    # Sort by area descending; L fiducials should be near the top.
    comps_sorted = sorted(comps, key=lambda c: c.area, reverse=True)

    # Consider top K largest components only (avoid scanning thousands of dots)
    K = min(40, len(comps_sorted))
    cand = comps_sorted[:K]

    # Define corner regions by centroid (with generous margins)
    # TL: left-top, TR: right-top, BL: left-bottom
    TL_cands = [c for c in cand if c.centroid[0] < 0.45 * img_w and c.centroid[1] < 0.45 * img_h]
    TR_cands = [c for c in cand if c.centroid[0] > 0.55 * img_w and c.centroid[1] < 0.45 * img_h]
    BL_cands = [c for c in cand if c.centroid[0] < 0.45 * img_w and c.centroid[1] > 0.55 * img_h]

    if not TL_cands or not TR_cands or not BL_cands:
        return None

    # Choose the largest in each region (most likely the fiducial)
    TL = max(TL_cands, key=lambda c: c.area)
    TR = max(TR_cands, key=lambda c: c.area)
    BL = max(BL_cands, key=lambda c: c.area)

    return TL, TR, BL


def fiducial_outer_points(TL: Component, TR: Component, BL: Component) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Convert each fiducial's bounding box into a single anchor point:
      TL uses bbox top-left corner
      TR uses bbox top-right corner
      BL uses bbox bottom-left corner

    These are stable, easy to detect, and consistent for your current images.
    """
    x, y, w, h = TL.bbox
    TLp = (float(x), float(y))

    x, y, w, h = TR.bbox
    TRp = (float(x + w - 1), float(y))

    x, y, w, h = BL.bbox
    BLp = (float(x), float(y + h - 1))

    return TLp, TRp, BLp


def warp_to_data_square(gray: np.ndarray, TLp, TRp, BLp, out_size: int, gap_cells: float) -> np.ndarray:
    """
    Warp the image so the INTERNAL data grid becomes a normalized out_size x out_size square.

    Important:
    - Our detected points are fiducial "outer corners".
    - In the encoder, those outer corners sit outside the data grid by "gap".
      So in destination coordinates, we map:
          TL_fid_outer -> (-gap, -gap)
          TR_fid_outer -> (S+gap, -gap)
          BL_fid_outer -> (-gap, S+gap)

      Then we crop the [0..S]x[0..S] region as the data square.

    This needs gap_cells consistent with encoder's fiducial_gap_cells.
    """
    S = float(out_size)  # data square size
    cell = S / GRID_N
    gap = float(gap_cells) * cell

    # Destination canvas includes extra margin to hold the negative coords
    canvas = int(np.ceil(S + 2.0 * gap))

    # Shift everything by +gap so coords become non-negative in the canvas
    shift = gap
    dst_TL = (shift + (-gap), shift + (-gap))
    dst_TR = (shift + (S + gap), shift + (-gap))
    dst_BL = (shift + (-gap), shift + (S + gap))

    src = np.array([TLp, TRp, BLp], dtype=np.float32)
    dst = np.array([dst_TL, dst_TR, dst_BL], dtype=np.float32)

    M = cv2.getAffineTransform(src, dst)

    warped_full = cv2.warpAffine(
        gray, M, (canvas, canvas),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )

    # Crop internal data square [shift .. shift+S)
    s0 = int(round(shift))
    s1 = s0 + out_size
    data_square = warped_full[s0:s1, s0:s1].copy()
    return data_square


def cell_means_disk(warped: np.ndarray, n: int, radius_ratio: float) -> np.ndarray:
    """
    Compute mean grayscale value inside a disk around each cell center.
    warped is already a normalized data-only square.

    This is robust to slight blur and dot shape variations.
    """
    H, W = warped.shape[:2]
    assert H == W
    S = H
    cell = S / n
    radius = max(2, int(cell * radius_ratio))

    disk = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    cv2.circle(disk, (radius, radius), radius, 1, -1)

    means = np.zeros((n, n), dtype=np.float32)

    for r in range(n):
        for c in range(n):
            cx = int((c + 0.5) * cell)
            cy = int((r + 0.5) * cell)

            x0 = max(0, cx - radius)
            x1 = min(S - 1, cx + radius)
            y0 = max(0, cy - radius)
            y1 = min(S - 1, cy + radius)

            patch = warped[y0:y1 + 1, x0:x1 + 1]

            dy0 = y0 - (cy - radius)
            dx0 = x0 - (cx - radius)
            d = disk[dy0:dy0 + patch.shape[0], dx0:dx0 + patch.shape[1]]

            vals = patch[d > 0].astype(np.float32)
            means[r, c] = float(vals.mean()) if vals.size else 255.0

    return means


def otsu_threshold_on_means(means: np.ndarray) -> float:
    """
    Otsu threshold on the 121 cell means (0..255).
    This works well because cells are roughly bimodal: dot-cells darker, empty-cells brighter.
    """
    v = means.reshape(-1)
    v8 = np.clip(v, 0, 255).astype(np.uint8)

    hist = np.bincount(v8, minlength=256).astype(np.float64)
    total = hist.sum()
    sum_total = (np.arange(256) * hist).sum()

    sumB = 0.0
    wB = 0.0
    max_var = -1.0
    thr = 127.0

    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > max_var:
            max_var = var_between
            thr = float(t)

    return thr


def means_to_matrix(means: np.ndarray, thr: float, dark_is_1: bool = True) -> List[List[int]]:
    """
    Convert means to a binary 11x11 matrix.

    dark_is_1=True:
      mean < thr  => 1 (dot present)
      mean >= thr => 0
    """
    n = means.shape[0]
    mat = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            m = float(means[r, c])
            if dark_is_1:
                mat[r][c] = 1 if m < thr else 0
            else:
                mat[r][c] = 1 if m > thr else 0
    return mat


def print_matrix(mat: List[List[int]]) -> None:
    for r in range(len(mat)):
        print(" ".join(str(int(x)) for x in mat[r]))


# =========================
# Main
# =========================
def main() -> None:
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img is None:
        print("REJECT: cannot read image:", IMAGE_PATH)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 1) Binary foreground mask (dots + fiducials)
    bw = otsu_binary_foreground(gray)

    # 2) Close to solidify fiducials (helps contour/component stability)
    bw = morph_close(bw, MORPH_CLOSE_K)

    # 3) Connected components
    comps = get_components(bw)

    # 4) Pick TL/TR/BL fiducials
    h, w = gray.shape[:2]
    picked = pick_fiducials_three_corners(comps, img_w=w, img_h=h)
    if picked is None:
        print("REJECT: cannot locate 3 fiducials (TL/TR/BL)")
        return

    TLc, TRc, BLc = picked
    TLp, TRp, BLp = fiducial_outer_points(TLc, TRc, BLc)

    # 5) Warp so the internal data grid becomes a normalized square
    data_square = warp_to_data_square(
        gray=gray,
        TLp=TLp,
        TRp=TRp,
        BLp=BLp,
        out_size=WARP_SIZE,
        gap_cells=FID_GAP_CELLS,
    )

    # 6) Sample 11x11 means
    means = cell_means_disk(data_square, n=GRID_N, radius_ratio=SAMPLE_RADIUS_RATIO)

    # 7) Threshold on means (Otsu)
    thr = otsu_threshold_on_means(means)

    # 8) Convert to matrix
    # Your printed dots are black, so "dark_is_1" is the correct polarity for these PNGs.
    mat = means_to_matrix(means, thr, dark_is_1=True)

    # 9) Output only matrix (as requested)
    print_matrix(mat)


if __name__ == "__main__":
    main()
