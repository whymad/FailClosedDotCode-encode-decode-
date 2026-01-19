# FCDDecoder.py
#
# ============================================================
# RuggedDot-11 Fail-Closed Decoder (2-stage DEBUG version)
# ============================================================
#
# Stage 1 (VISION -> MATRIX):
#   image -> 11x11 matrix (dot=1, empty=0)
#   Print the matrix for debugging.
#
# Stage 2 (MATRIX -> NUMBER):
#   11x11 matrix -> 121 physical bits -> deinterleave -> ECC -> CRC32 gate
#   Output number ONLY if all checks pass (fail-closed).
#
# Why this split?
#   Your current failures are very likely caused by the VISION step
#   extracting the wrong 121 bits (misaligned ROI / bad threshold).
#   Printing the 11x11 matrix lets us verify the sampling first.
#
# Dependencies:
#   pip install opencv-python numpy
#
# ============================================================

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np


# ============================================================
# ===================== USER EDIT AREA =======================
# ============================================================

IMAGE_PATH = "ruggeddot11_123454321.png"

# Must match encoder
INTERLEAVE_STEP = 37

# ---- Vision settings ----
# For your rendered PNG (white background + sparse black dots), this is usually best:
ROI_MODE = "full"     # "full" or "bbox"
GRID_SIZE = 11

# If ROI_MODE="full":
#   We take a centered square crop from the full image.
#   Works well when the code occupies most of the image (like your PNG).
FULL_ROI_MARGIN_FRAC = 0.00
# (0.00 means use the whole centered square. Increase slightly if you have extra border.)

# If ROI_MODE="bbox":
#   We find the bounding box of dark pixels then expand. Good for photos,
#   but can misalign if outer cells are blank (no dots).
BBOX_EXPAND = 0.40
BBOX_MIN_SIDE_FRAC = 0.60  # for photos; for rendered PNG can be 0.90+

# Sampling
SAMPLE_RADIUS_RATIO = 0.28

# Threshold method on "cell means" (NOT pixel-level Otsu):
#   "otsu_means": Otsu on the 121 cell-mean values
#   "kmeans2"   : 2-cluster kmeans on the 121 means, then choose dark cluster as "dot"
MEAN_THRESHOLD_MODE = "otsu_means"  # "otsu_means" or "kmeans2"

# Polarity tries:
#   - dark_is_1: dark dot -> bit 1  (most common)
#   - bright_is_1: bright dot -> bit 1 (contrast inverted)
TRY_INVERT_POLARITY = True

# ============================================================
# =================== END USER EDIT AREA ======================
# ============================================================


# ============================================================
# ---------------------- Utilities: bits/CRC ------------------
# ============================================================

def bits_to_int(bits: List[int]) -> int:
    """Big-endian bit list -> integer."""
    x = 0
    for b in bits:
        x = (x << 1) | (b & 1)
    return x

def bits_to_bytes(bits: List[int]) -> bytes:
    """
    Pack bits into bytes (big-endian per byte), right-pad zeros.
    Must match encoder.
    """
    pad = (-len(bits)) % 8
    b2 = bits + [0] * pad
    out = bytearray()
    for i in range(0, len(b2), 8):
        out.append(bits_to_int(b2[i:i + 8]))
    return bytes(out)

def crc32_of_bits(bits: List[int]) -> int:
    """CRC32 over packed bytes of bits."""
    return zlib.crc32(bits_to_bytes(bits)) & 0xFFFFFFFF


# ============================================================
# --------------------- Hamming(15,11) ------------------------
# ============================================================

HAMMING_DATA_POS = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
HAMMING_PARITY_POS = [1, 2, 4, 8]

def hamming15_11_syndrome(code15: List[int]) -> int:
    """
    Return syndrome (0..15).
    0 => parity checks pass.
    non-zero => suggested 1-based bit position to flip for single-bit correction.
    """
    if len(code15) != 15:
        raise ValueError("hamming15_11_syndrome expects 15 bits")

    code = [0] + [b & 1 for b in code15]  # 1..15 indexing
    syn = 0
    for p in HAMMING_PARITY_POS:
        parity = 0
        for i in range(1, 16):
            if i & p:
                parity ^= code[i]
        if parity != 0:
            syn |= p
    return syn

def hamming15_11_decode(code15: List[int]) -> Tuple[List[int], Dict[str, Any]]:
    """
    Decode one 15-bit codeword to 11 data bits.
    Correct up to 1 bit error.
    """
    code = [b & 1 for b in code15]
    syn_before = hamming15_11_syndrome(code)

    corrected = False
    corrected_pos = 0
    if syn_before != 0:
        idx = syn_before - 1
        if 0 <= idx < 15:
            code[idx] ^= 1
            corrected = True
            corrected_pos = syn_before

    syn_after = hamming15_11_syndrome(code)

    code1 = [0] + code
    data11 = [code1[pos] for pos in HAMMING_DATA_POS]

    return data11, {
        "syndrome_before": syn_before,
        "syndrome_after": syn_after,
        "corrected": corrected,
        "corrected_pos_1based": corrected_pos,
    }


# ============================================================
# ------------------- Interleave / Deinterleave --------------
# ============================================================

def make_interleaver(step: int = 37, n: int = 121) -> List[int]:
    """Permutation P[i] = (i*step) mod n. Requires gcd(step,n)=1."""
    def gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a
    if gcd(step, n) != 1:
        raise ValueError(f"step={step} must be coprime with n={n}")
    return [(i * step) % n for i in range(n)]

def deinterleave_physical_to_logical(physical121: List[int], step: int) -> List[int]:
    """
    Encoder wrote: physical[P[i]] = logical[i]
    Decoder reads: logical[i] = physical[P[i]]
    """
    P = make_interleaver(step=step, n=121)
    logical = [0] * 121
    for i, p in enumerate(P):
        logical[i] = physical121[p]
    return logical


# ============================================================
# ======================= Stage 1: VISION =====================
# ============================================================

@dataclass
class VisionDebug:
    roi: Tuple[int, int, int, int]
    cell_means: np.ndarray  # shape (11,11), float
    threshold: float
    polarity: str
    mode: str  # threshold mode

def _center_square_roi(gray: np.ndarray, margin_frac: float) -> Tuple[int, int, int, int]:
    """
    Take centered square ROI from full image.

    This is the most reliable method for your current rendered PNG because:
      - the code is already aligned and centered,
      - bounding-box methods can shrink when outer cells are blank (bit=0),
      - shrink => sampling misalignment => CRC fails.

    margin_frac:
      0.00 => use the largest centered square fully inside the image.
      >0   => shrink slightly to avoid outside border.
    """
    H, W = gray.shape[:2]
    side = min(H, W)
    # shrink a little if requested
    side = int(round(side * (1.0 - margin_frac)))
    side = max(20, side)

    cx = W // 2
    cy = H // 2
    half = side // 2

    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(W - 1, cx + half)
    y1 = min(H - 1, cy + half)
    return (x0, y0, x1, y1)

def _binarize_dark_pixels(gray: np.ndarray) -> np.ndarray:
    """
    Find dark pixels (dots) for bbox ROI mode.

    We use a simple Otsu on pixels only for bbox detection, not for bit decision.
    """
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # dots are dark => bw==0
    mask = (bw == 0).astype(np.uint8)
    return mask

def _bbox_square_roi(gray: np.ndarray, expand: float, min_side_frac: float) -> Optional[Tuple[int, int, int, int]]:
    """
    Bounding-box ROI mode. Good for photos when code doesn't fill the image.
    But can be risky when outer cells are blank.

    We enforce a minimum ROI size fraction to avoid over-shrinking.
    """
    H, W = gray.shape[:2]
    mask = _binarize_dark_pixels(gray)

    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    bbox_side = max(x_max - x_min + 1, y_max - y_min + 1)
    side = bbox_side * (1.0 + 2.0 * expand)

    min_side = min_side_frac * float(min(H, W))
    side = max(side, min_side)

    half = 0.5 * side
    x0 = int(round(cx - half))
    x1 = int(round(cx + half))
    y0 = int(round(cy - half))
    y1 = int(round(cy + half))

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W - 1, x1)
    y1 = min(H - 1, y1)

    if (x1 - x0) < 50 or (y1 - y0) < 50:
        return None
    return (x0, y0, x1, y1)

def _cell_means_from_roi(gray: np.ndarray, roi: Tuple[int, int, int, int], sample_radius_ratio: float) -> np.ndarray:
    """
    Compute mean intensity for each of the 11x11 cells.

    Pipeline:
      - crop ROI
      - resize ROI to canonical size (N*80)
      - for each cell center, average a disk neighborhood

    Output:
      means[r,c] in [0..255], float
    """
    x0, y0, x1, y1 = roi
    roi_gray = gray[y0:y1 + 1, x0:x1 + 1]

    target = GRID_SIZE * 80
    roi_resized = cv2.resize(roi_gray, (target, target), interpolation=cv2.INTER_AREA)

    cell = target / GRID_SIZE
    radius = max(2, int(cell * sample_radius_ratio))

    disk = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    cv2.circle(disk, (radius, radius), radius, 1, -1)

    means = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cx = int((c + 0.5) * cell)
            cy = int((r + 0.5) * cell)

            xA = max(0, cx - radius)
            yA = max(0, cy - radius)
            xB = min(target - 1, cx + radius)
            yB = min(target - 1, cy + radius)

            patch = roi_resized[yA:yB + 1, xA:xB + 1]

            d = disk[
                (yA - (cy - radius)):(yA - (cy - radius)) + patch.shape[0],
                (xA - (cx - radius)):(xA - (cx - radius)) + patch.shape[1],
            ]
            ys, xs = np.where(d > 0)
            vals = patch[ys, xs].astype(np.float32)
            m = float(vals.mean()) if vals.size else 255.0
            means[r, c] = m

    return means

def _threshold_on_means(means: np.ndarray, mode: str) -> float:
    """
    Decide a threshold using the 121 per-cell mean intensities (NOT pixel histogram).

    Why:
      - Your image is mostly white with a few black dots.
      - Pixel-level Otsu can become unstable / pick extreme threshold.
      - But the 121 cell-means are bimodal (dot cells vs empty cells) -> stable split.
    """
    v = means.reshape(-1).astype(np.float32)

    if mode == "otsu_means":
        # Otsu on the 121 values (we scale to 0..255 uint8 for OpenCV threshold)
        v8 = np.clip(v, 0, 255).astype(np.uint8)
        # cv2.threshold expects a "1-channel image", so shape it
        _, _bw = cv2.threshold(v8.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Unfortunately OpenCV doesn't return threshold directly here in a clean way.
        # So we recompute by scanning a simple Otsu ourselves on 256 bins.
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

    elif mode == "kmeans2":
        # 2-cluster kmeans on 121 means; pick split between centers
        # We implement tiny kmeans manually to avoid sklearn dependency.
        x = v.copy()
        # init centers: min and max
        c0, c1 = float(x.min()), float(x.max())
        for _ in range(20):
            # assign
            d0 = np.abs(x - c0)
            d1 = np.abs(x - c1)
            a = (d1 < d0).astype(np.int32)  # 0 -> cluster0, 1 -> cluster1
            if (a == 0).any():
                c0_new = float(x[a == 0].mean())
            else:
                c0_new = c0
            if (a == 1).any():
                c1_new = float(x[a == 1].mean())
            else:
                c1_new = c1
            if abs(c0_new - c0) < 1e-3 and abs(c1_new - c1) < 1e-3:
                break
            c0, c1 = c0_new, c1_new
        # threshold at midpoint
        return 0.5 * (c0 + c1)

    else:
        raise ValueError("MEAN_THRESHOLD_MODE must be 'otsu_means' or 'kmeans2'")

def image_to_matrix_11x11(gray: np.ndarray, polarity: str) -> Tuple[List[List[int]], VisionDebug]:
    """
    Stage 1: image -> 11x11 matrix.

    polarity:
      - "dark_is_1": if cell mean < threshold => dot => 1
      - "bright_is_1": if cell mean > threshold => dot => 1
    """
    # 1) Choose ROI
    if ROI_MODE == "full":
        roi = _center_square_roi(gray, margin_frac=FULL_ROI_MARGIN_FRAC)
    elif ROI_MODE == "bbox":
        roi2 = _bbox_square_roi(gray, expand=BBOX_EXPAND, min_side_frac=BBOX_MIN_SIDE_FRAC)
        if roi2 is None:
            raise ValueError("bbox ROI failed. Try ROI_MODE='full' for rendered PNGs.")
        roi = roi2
    else:
        raise ValueError("ROI_MODE must be 'full' or 'bbox'")

    # 2) Compute per-cell means
    means = _cell_means_from_roi(gray, roi, sample_radius_ratio=SAMPLE_RADIUS_RATIO)

    # 3) Threshold on means (robust for sparse dots)
    thr = _threshold_on_means(means, mode=MEAN_THRESHOLD_MODE)

    # 4) Convert to bits
    mat = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            m = float(means[r, c])
            if polarity == "dark_is_1":
                mat[r][c] = 1 if m < thr else 0
            elif polarity == "bright_is_1":
                mat[r][c] = 1 if m > thr else 0
            else:
                raise ValueError("polarity must be 'dark_is_1' or 'bright_is_1'")

    dbg = VisionDebug(
        roi=roi,
        cell_means=means,
        threshold=float(thr),
        polarity=polarity,
        mode=MEAN_THRESHOLD_MODE,
    )
    return mat, dbg

def print_matrix(mat: List[List[int]]) -> None:
    """
    Pretty-print 11x11 matrix as 0/1 grid.
    """
    print("11x11 MATRIX (1=dot, 0=empty):")
    for r in range(GRID_SIZE):
        line = " ".join(str(mat[r][c]) for c in range(GRID_SIZE))
        print(line)

def print_means_summary(dbg: VisionDebug) -> None:
    """
    Print helpful vision debug stats.
    """
    means = dbg.cell_means
    print("\nVISION DEBUG:")
    print("  ROI:", dbg.roi, f"(ROI_MODE={ROI_MODE})")
    print("  threshold_mode:", dbg.mode)
    print("  polarity:", dbg.polarity)
    print("  threshold_on_means:", dbg.threshold)
    print("  cell_means min/max:", float(means.min()), float(means.max()))
    # Optional: show the means matrix (rounded) for deep debugging
    print("\nCELL MEANS (rounded):")
    for r in range(GRID_SIZE):
        row = " ".join(f"{int(round(means[r,c])):3d}" for c in range(GRID_SIZE))
        print(row)


# ============================================================
# ======================= Stage 2: DECODE =====================
# ============================================================

@dataclass
class DecodeResult:
    ok: bool
    value_str: Optional[str]
    version: Optional[int]
    reason: str
    debug: Dict[str, Any]

def matrix_to_physical_bits(mat: List[List[int]]) -> List[int]:
    """
    Convert 11x11 matrix to 121 physical bits in row-major order.
    physical_idx = r*11 + c
    """
    bits = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            bits.append(int(mat[r][c]) & 1)
    if len(bits) != 121:
        raise ValueError("matrix_to_physical_bits: unexpected length")
    return bits

def decode_matrix_failclosed(mat: List[List[int]]) -> DecodeResult:
    """
    Stage 2: matrix -> number with fail-closed checks.

    Decoding layout (must match encoder):
      physical121 --deinterleave--> logical121
      logical121:
        [0..119] = 8 * Hamming(15,11) codewords concatenated (120 bits)
        [120]    = sentinel bit (must be 1)

      After Hamming decode:
        info88 = data54 (54) + version2 (2) + crc32 (32)
      Finally:
        CRC32(data54||version2) must equal stored crc32.
    """
    physical121 = matrix_to_physical_bits(mat)
    logical121 = deinterleave_physical_to_logical(physical121, step=INTERLEAVE_STEP)

    # Sentinel check
    sentinel = logical121[120]
    if sentinel != 1:
        return DecodeResult(False, None, None, "REJECT: sentinel != 1", {"sentinel": sentinel})

    code120 = logical121[:120]

    info88: List[int] = []
    corrections = 0
    nonzero_after = 0

    for i in range(8):
        cw15 = code120[i * 15:(i + 1) * 15]
        data11, dbg = hamming15_11_decode(cw15)
        info88.extend(data11)
        if dbg["corrected"]:
            corrections += 1
        if dbg["syndrome_after"] != 0:
            nonzero_after += 1

    if nonzero_after != 0:
        return DecodeResult(
            False, None, None,
            "REJECT: ECC parity still failing after correction (severe corruption / wrong sampling)",
            {"ecc_bad_after": nonzero_after, "ecc_corrections": corrections}
        )

    if len(info88) != 88:
        return DecodeResult(False, None, None, "REJECT: info88 length error", {"len": len(info88)})

    data54 = info88[:54]
    ver2 = info88[54:56]
    crc_bits = info88[56:88]

    version = bits_to_int(ver2)
    stored_crc = bits_to_int(crc_bits)
    calc_crc = crc32_of_bits(data54 + ver2)

    if calc_crc != stored_crc:
        return DecodeResult(
            False, None, version,
            "REJECT: CRC32 mismatch (fail-closed)",
            {
                "stored_crc": hex(stored_crc),
                "calc_crc": hex(calc_crc),
                "ecc_corrections": corrections
            }
        )

    value = bits_to_int(data54)
    return DecodeResult(
        True, str(value), version,
        "OK",
        {"crc32": hex(calc_crc), "ecc_corrections": corrections}
    )


# ============================================================
# ================================ MAIN =======================
# ============================================================

def main():
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img is None:
        print("ERROR: cannot read image:", IMAGE_PATH)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Light blur reduces aliasing noise for photos; for PNG it usually doesn't hurt.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # ---- Trial A: dark dot => 1 ----
    matA, dbgA = image_to_matrix_11x11(gray, polarity="dark_is_1")
    print("\n================ TRIAL A: polarity = dark_is_1 ================")
    print_means_summary(dbgA)
    print()
    print_matrix(matA)

    resA = decode_matrix_failclosed(matA)
    print("\nDECODE RESULT (Trial A):")
    if resA.ok:
        print("  DECODE OK")
        print("  value   :", resA.value_str)
        print("  version :", resA.version)
        print("  crc32   :", resA.debug.get("crc32"))
        print("  ecc_corrections:", resA.debug.get("ecc_corrections"))
        return
    else:
        print("  DECODE REJECT")
        print("  reason:", resA.reason)
        # print("  debug:", resA.debug)

    # ---- Trial B: bright dot => 1 (optional) ----
    if TRY_INVERT_POLARITY:
        matB, dbgB = image_to_matrix_11x11(gray, polarity="bright_is_1")
        print("\n================ TRIAL B: polarity = bright_is_1 ================")
        print_means_summary(dbgB)
        print()
        print_matrix(matB)

        resB = decode_matrix_failclosed(matB)
        print("\nDECODE RESULT (Trial B):")
        if resB.ok:
            print("  DECODE OK")
            print("  value   :", resB.value_str)
            print("  version :", resB.version)
            print("  crc32   :", resB.debug.get("crc32"))
            print("  ecc_corrections:", resB.debug.get("ecc_corrections"))
            return
        else:
            print("  DECODE REJECT")
            print("  reason:", resB.reason)
            # print("  debug:", resB.debug)

    print("\nFINAL: both polarities rejected (fail-closed).")

if __name__ == "__main__":
    main()
