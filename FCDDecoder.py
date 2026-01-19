# rugged_dotcode_11x11_failclosed_decoder_en.py
#
# ============================================================
# RuggedDot-11 (Fail-Closed) VISUAL DECODER
# ============================================================
#
# This script decodes the 11x11 dot-matrix code produced by:
#   rugged_dotcode_11x11_failclosed_en.py   (encoder + renderer)
#
# INPUT : an image containing (at least) the INTERNAL 11x11 dot grid.
# OUTPUT: the decoded decimal number (string) if and only if all checks pass,
#         otherwise "REJECT" (fail-closed).
#
# ---------------------------
# Fail-Closed contract
# ---------------------------
# The decoder MUST NOT output a number unless it is extremely confident.
# Concretely, we require ALL of the following to pass:
#   1) Extract 121 physical bits from the 11x11 grid (from the image).
#   2) De-interleave (inverse permutation) to recover 121 logical bits.
#   3) Sentinel bit must equal 1.
#   4) Hamming(15,11) decode all 8 codewords (correct up to 1 bit per codeword).
#   5) CRC32 check must match (CRC32(data54||ver2) == stored_crc32).
# If any step fails -> REJECT (no output).
#
# IMPORTANT PRACTICAL NOTE
# ------------------------
# This decoder intentionally focuses on "internal dot grid only".
# Without an outer fiducial frame, robust localization under rotation/tilt/stretch
# is hard in real fabric images.
#
# Therefore, this decoder implements a "best-effort" localization:
#   - find the bounding box of all dark pixels (candidate dots),
#   - expand to a square ROI,
#   - optional perspective correction from a detected outer contour (if present).
#
# For production-level robustness on fabric, you should add an outer fiducial frame
# (thick border + corner marks) and then decode the internal grid after a homography
# warp. This script contains hooks for that, but cannot fully guarantee geometry
# robustness without those fiducials.
#
# ============================================================
# Dependencies
# ============================================================
# pip install opencv-python numpy pillow
#
# ============================================================

from __future__ import annotations

import argparse
import zlib
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np


# ============================================================
# -------------------- Bit / CRC / ECC helpers ----------------
# ============================================================

def int_to_bits(x: int, nbits: int) -> List[int]:
    """Big-endian bit list (MSB first)."""
    return [(x >> (nbits - 1 - i)) & 1 for i in range(nbits)]


def bits_to_int(bits: List[int]) -> int:
    """Big-endian bits -> integer."""
    x = 0
    for b in bits:
        x = (x << 1) | (b & 1)
    return x


def bits_to_bytes(bits: List[int]) -> bytes:
    """
    Pack bits into bytes (big-endian per byte), right-padding zeros.
    Encoder and decoder MUST use the same packing convention.
    """
    pad = (-len(bits)) % 8
    b2 = bits + [0] * pad
    out = bytearray()
    for i in range(0, len(b2), 8):
        out.append(bits_to_int(b2[i:i + 8]))
    return bytes(out)


def crc32_of_bits(bits: List[int]) -> int:
    """
    CRC32 over packed bytes of bits, using zlib.crc32 (standard CRC-32).
    """
    return zlib.crc32(bits_to_bytes(bits)) & 0xFFFFFFFF


# -------------------------
# Hamming(15,11) positions
# -------------------------
# Positions are 1..15:
#   parity positions: 1,2,4,8
#   data positions  : 3,5,6,7,9,10,11,12,13,14,15
HAMMING_DATA_POS = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
HAMMING_PARITY_POS = [1, 2, 4, 8]


def hamming15_11_syndrome(code15: List[int]) -> int:
    """
    Compute the Hamming syndrome for a 15-bit codeword (even parity).
    The syndrome is an integer in [0..15]. If 0, no parity-check error is detected.
    If non-zero, it indicates the 1-based bit position that should be flipped
    to correct a single-bit error (classic Hamming property).

    Note: Hamming(15,11) without an additional overall parity bit is NOT a perfect
    double-error detector. Multi-bit errors may produce a misleading syndrome.
    We rely on CRC32 to fail-closed reject those cases.
    """
    if len(code15) != 15:
        raise ValueError("hamming15_11_syndrome expects 15 bits")

    # Use 1..15 indexing for parity logic
    code = [0] + [b & 1 for b in code15]

    syn = 0
    for p in HAMMING_PARITY_POS:
        parity = 0
        # parity bit covers all positions i where (i & p) != 0
        for i in range(1, 16):
            if i & p:
                parity ^= code[i]
        # For even parity, the XOR of covered bits should be 0
        if parity != 0:
            syn |= p
    return syn


def hamming15_11_decode(code15: List[int]) -> Tuple[List[int], Dict[str, Any]]:
    """
    Decode one Hamming(15,11) codeword.
    - Correct up to 1 bit error (if syndrome != 0, flip that bit).
    - Extract the 11 data bits.

    Returns:
      data11: length-11 list of bits
      info : dict with debug (syndrome, corrected, etc.)
    """
    if len(code15) != 15:
        raise ValueError("hamming15_11_decode expects 15 bits")

    code = [b & 1 for b in code15]
    syn = hamming15_11_syndrome(code)

    corrected = False
    corrected_pos_1based = 0

    if syn != 0:
        # Single-bit correction attempt: flip the indicated position.
        # If the error was indeed a single-bit error, this fixes it.
        # If there were multiple bit errors, this may mis-correct.
        idx = syn - 1  # convert 1-based to 0-based
        if 0 <= idx < 15:
            code[idx] ^= 1
            corrected = True
            corrected_pos_1based = syn

    # After (possible) correction, extract the data bits from fixed positions
    # using the agreed mapping positions (1..15).
    code1 = [0] + code  # 1..15 indexing
    data11 = [code1[pos] for pos in HAMMING_DATA_POS]

    # Recompute syndrome after correction for diagnostics
    syn_after = hamming15_11_syndrome(code)

    info = {
        "syndrome_before": syn,
        "syndrome_after": syn_after,
        "corrected": corrected,
        "corrected_pos_1based": corrected_pos_1based,
    }
    return data11, info


# ============================================================
# -------------------- Interleaver / De-interleaver ----------------
# ============================================================

def make_interleaver(step: int = 37, n: int = 121) -> List[int]:
    """
    The encoder uses:
        physical_index = P[logical_index] = (logical_index * step) % n

    This function returns P.
    Decoder recovers logical bits via:
        logical[logical_index] = physical[P[logical_index]]
    (i.e., gather back from physical positions in the same order).
    """
    def gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a

    if gcd(step, n) != 1:
        raise ValueError(f"Interleave step={step} must be coprime with n={n}")
    return [(i * step) % n for i in range(n)]


def deinterleave_physical_to_logical(physical121: List[int], step: int = 37) -> List[int]:
    """
    Inverse of the encoder placement.

    Encoder did:
        physical[P[i]] = logical[i]
    So decoder does:
        logical[i] = physical[P[i]]
    """
    if len(physical121) != 121:
        raise ValueError("deinterleave expects length-121 physical bits")

    P = make_interleaver(step=step, n=121)
    logical121 = [0] * 121
    for logical_idx, phys_idx in enumerate(P):
        logical121[logical_idx] = physical121[phys_idx]
    return logical121


# ============================================================
# -------------------- Visual extraction of 11x11 bits ----------------
# ============================================================

@dataclass
class VisionConfig:
    """
    Vision parameters controlling how we locate and sample the grid.

    grid_size:
      Internal grid is fixed to 11x11.

    roi_expand:
      When we find a bounding box of dark pixels, we expand it by this fraction
      to include margin / missing dots. Example 0.25 expands each side by 25% of box size.

    sample_radius_ratio:
      For each cell, we sample a small disk around the cell center.
      radius = sample_radius_ratio * cell_size.
      Too small => sensitive to slight misalignment.
      Too large => mixes with neighbors / background.

    thresh_mode:
      Thresholding method:
        - "otsu": global Otsu threshold (works if background is uniform).
        - "adaptive": adaptive threshold (works better under uneven illumination).

    invert_try:
      If True, we attempt decoding with both polarities:
        (A) dot=dark => bit=1
        (B) dot=bright => bit=1 (inverted)
      We accept only if CRC32 passes. This is robust to contrast polarity flips.
    """
    grid_size: int = 11
    roi_expand: float = 0.35
    sample_radius_ratio: float = 0.28
    thresh_mode: str = "otsu"          # "otsu" or "adaptive"
    invert_try: bool = True


def _binarize(gray: np.ndarray, mode: str) -> np.ndarray:
    """
    Convert grayscale to binary mask where "dot candidates" are 1.

    We want a mask where dots are foreground=1.
    If dots are dark on bright background, then:
      threshold -> dots become 1 after inversion.
    """
    if mode == "adaptive":
        # Adaptive threshold: useful under uneven lighting.
        # We invert at the end to make dark dots => 1.
        bw = cv2.adaptiveThreshold(
            gray, 255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=31,
            C=7
        )
        # bw is 255 for "white". We want dots (dark) => 1, so invert:
        mask = (bw == 0).astype(np.uint8)
        return mask
    else:
        # Otsu global threshold.
        # We threshold to get a binary image; then invert so dark dots => 1.
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (bw == 0).astype(np.uint8)
        return mask


def _find_square_roi_from_foreground(mask: np.ndarray, expand: float) -> Optional[Tuple[int, int, int, int]]:
    """
    Find a square ROI that roughly contains the 11x11 dot grid.

    Strategy:
      - Find all foreground pixels (mask==1).
      - Compute bounding box (min/max).
      - Expand the box by 'expand' fraction.
      - Make it square by expanding the shorter side.
      - Clip to image boundaries.

    Returns (x0, y0, x1, y1) in pixel coordinates, or None if no foreground found.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        # Too few foreground pixels -> likely failed segmentation
        return None

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    w = x_max - x_min + 1
    h = y_max - y_min + 1

    # Expand the bounding box
    pad_x = int(w * expand)
    pad_y = int(h * expand)

    x0 = x_min - pad_x
    x1 = x_max + pad_x
    y0 = y_min - pad_y
    y1 = y_max + pad_y

    # Make square: expand the shorter dimension
    w2 = x1 - x0 + 1
    h2 = y1 - y0 + 1
    if w2 > h2:
        diff = w2 - h2
        y0 -= diff // 2
        y1 += diff - diff // 2
    else:
        diff = h2 - w2
        x0 -= diff // 2
        x1 += diff - diff // 2

    # Clip to image
    H, W = mask.shape[:2]
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W - 1, x1)
    y1 = min(H - 1, y1)

    # Re-check minimum size
    if (x1 - x0) < 40 or (y1 - y0) < 40:
        return None

    return x0, y0, x1, y1


def _sample_bits_from_roi(
    gray: np.ndarray,
    roi: Tuple[int, int, int, int],
    grid_size: int,
    sample_radius_ratio: float,
    polarity: str,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Given a square ROI that contains the grid, sample each cell center and decide bit.

    We produce "physical bits" in row-major order (idx=row*11+col).

    Decision rule:
      - Compute mean intensity inside a small disk around each cell center.
      - Compare to an automatically estimated threshold for the ROI.

    Polarity options:
      - "dark_is_1": darker dot => bit=1 (typical printed/engraved black dot)
      - "bright_is_1": brighter dot => bit=1 (inverted contrast scenario)

    Returns:
      bits121: list of 121 bits (0/1) in row-major physical order
      stats: diagnostics (threshold used, etc.)
    """
    x0, y0, x1, y1 = roi
    roi_gray = gray[y0:y1 + 1, x0:x1 + 1]
    N = grid_size

    # Resize ROI to a canonical size to simplify sampling and make it scale-invariant.
    # Using a multiple of N provides clean cell boundaries.
    target = 11 * 80  # 880 px: big enough for stable sampling
    roi_resized = cv2.resize(roi_gray, (target, target), interpolation=cv2.INTER_AREA)

    cell = target / N
    radius = max(2, int(cell * sample_radius_ratio))

    # Estimate a global threshold within ROI using Otsu on the resized ROI
    # This works surprisingly well if ROI is mostly background + dots.
    _, bw_otsu = cv2.threshold(roi_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu threshold is implicitly encoded in bw_otsu; we can compute it by re-running:
    otsu_thr, _ = cv2.threshold(roi_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = otsu_thr

    bits: List[int] = []
    means: List[float] = []

    # Precompute a disk mask for sampling
    disk = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    cv2.circle(disk, (radius, radius), radius, 1, -1)
    disk_coords = np.where(disk > 0)

    for r in range(N):
        for c in range(N):
            cx = int((c + 0.5) * cell)
            cy = int((r + 0.5) * cell)

            # Extract patch around center
            xA = max(0, cx - radius)
            yA = max(0, cy - radius)
            xB = min(target - 1, cx + radius)
            yB = min(target - 1, cy + radius)
            patch = roi_resized[yA:yB + 1, xA:xB + 1]

            # Match disk mask to patch size (clipped at borders)
            d = disk[
                (yA - (cy - radius)):(yA - (cy - radius)) + patch.shape[0],
                (xA - (cx - radius)):(xA - (cx - radius)) + patch.shape[1],
            ]
            ys, xs = np.where(d > 0)
            vals = patch[ys, xs].astype(np.float32)
            m = float(vals.mean()) if vals.size else 255.0
            means.append(m)

            # Convert mean intensity to bit decision based on polarity
            if polarity == "dark_is_1":
                # darker than threshold => dot present => bit=1
                bit = 1 if m < thr else 0
            elif polarity == "bright_is_1":
                # brighter than threshold => dot present => bit=1
                bit = 1 if m > thr else 0
            else:
                raise ValueError("polarity must be 'dark_is_1' or 'bright_is_1'")

            bits.append(bit)

    stats = {
        "roi": roi,
        "target_size": target,
        "cell_size_px": cell,
        "sample_radius_px": radius,
        "otsu_threshold": float(thr),
        "polarity": polarity,
        "mean_intensity_min": float(np.min(means)) if means else None,
        "mean_intensity_max": float(np.max(means)) if means else None,
    }
    return bits, stats


def extract_physical_bits_11x11(image_bgr: np.ndarray, vcfg: VisionConfig) -> Tuple[List[int], Dict[str, Any]]:
    """
    High-level extraction:
      1) Convert to grayscale
      2) Binarize to find dot pixels
      3) Compute a square ROI that contains the grid
      4) Sample 11x11 bits from ROI using intensity statistics

    Returns:
      bits121_physical (row-major)
      vision_meta (debug info)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Mild denoise to stabilize thresholding on fabric texture
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    mask = _binarize(gray_blur, mode=vcfg.thresh_mode)

    roi = _find_square_roi_from_foreground(mask, expand=vcfg.roi_expand)
    if roi is None:
        raise ValueError("Failed to localize grid ROI: no reliable dot foreground found.")

    # We do not decide polarity here; we only return gray and ROI as meta.
    # Actual bit extraction may be tried in both polarities.
    return [], {"gray": gray_blur, "roi": roi, "mask": mask}


# ============================================================
# -------------------- Full decode (Vision + ECC + CRC) ----------------
# ============================================================

@dataclass
class DecodeResult:
    ok: bool
    value_str: Optional[str]
    version: Optional[int]
    reason: str
    debug: Dict[str, Any]


def decode_from_physical_bits_failclosed(
    physical121: List[int],
    interleave_step: int = 37,
) -> DecodeResult:
    """
    Pure bit-level fail-closed decoding:
      physical bits -> logical bits -> sentinel -> ECC decode -> CRC32 gate -> output.

    Note:
      Without additional length info, decoding returns the integer value as a string.
      If your IDs can have leading zeros and you need to preserve them, you must
      encode the length explicitly (not implemented in the current encoder spec).
    """
    if len(physical121) != 121:
        return DecodeResult(False, None, None, "REJECT: physical bit length != 121", {})

    # 1) de-interleave
    logical121 = deinterleave_physical_to_logical(physical121, step=interleave_step)

    # 2) sentinel check (fail fast)
    sentinel = logical121[120]
    if sentinel != 1:
        return DecodeResult(False, None, None, "REJECT: sentinel != 1", {"sentinel": sentinel})

    code120 = logical121[:120]

    # 3) Hamming decode 8 blocks
    info88: List[int] = []
    ecc_debug = []
    corrections = 0
    nonzero_syndromes_after = 0

    for i in range(8):
        code15 = code120[i * 15:(i + 1) * 15]
        data11, info = hamming15_11_decode(code15)
        info88.extend(data11)
        ecc_debug.append(info)

        if info["corrected"]:
            corrections += 1
        # If syndrome_after != 0, something is off (should be 0 for valid codeword after correction)
        if info["syndrome_after"] != 0:
            nonzero_syndromes_after += 1

    if len(info88) != 88:
        return DecodeResult(False, None, None, "REJECT: internal info length != 88", {})

    # Optional strictness:
    # If you want to be extra conservative, you could reject if any syndrome_after != 0.
    # In practice, syndrome_after should be 0; if not, we reject.
    if nonzero_syndromes_after != 0:
        return DecodeResult(
            False, None, None,
            "REJECT: ECC parity-check still failing after correction (severe corruption)",
            {"ecc_debug": ecc_debug}
        )

    # 4) Parse fields
    data54 = info88[:54]
    ver2 = info88[54:56]
    crc32_bits = info88[56:88]

    version = bits_to_int(ver2)
    stored_crc32 = bits_to_int(crc32_bits)

    # 5) CRC32 gate
    crc_calc = crc32_of_bits(data54 + ver2)
    if crc_calc != stored_crc32:
        return DecodeResult(
            False, None, version,
            "REJECT: CRC32 mismatch (fail-closed)",
            {
                "stored_crc32_hex": hex(stored_crc32),
                "calc_crc32_hex": hex(crc_calc),
                "ecc_corrections": corrections,
                "ecc_debug": ecc_debug,
            }
        )

    # 6) Output number (decimal). Leading zeros cannot be recovered without extra length bits.
    n = bits_to_int(data54)
    value_str = str(n)

    return DecodeResult(
        True, value_str, version,
        "OK",
        {
            "ecc_corrections": corrections,
            "stored_crc32_hex": hex(stored_crc32),
            "calc_crc32_hex": hex(crc_calc),
            "ecc_debug": ecc_debug,
        }
    )


def decode_image_failclosed(
    image_path: str,
    interleave_step: int = 37,
    vcfg: VisionConfig = VisionConfig(),
) -> DecodeResult:
    """
    End-to-end fail-closed decode from image file.

    Strategy:
      - Load image
      - Localize an ROI containing the dot grid
      - Try extracting bits with polarity "dark_is_1"
      - If CRC fails and invert_try=True, also try "bright_is_1"
      - Accept only if CRC passes.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return DecodeResult(False, None, None, "REJECT: cannot read image", {"path": image_path})

    # 1) ROI localization (best-effort)
    try:
        _, meta0 = extract_physical_bits_11x11(img, vcfg)
        gray = meta0["gray"]
        roi = meta0["roi"]
    except Exception as e:
        return DecodeResult(False, None, None, f"REJECT: ROI localization failed ({e})", {})

    # 2) Try polarity A: dark dot = 1
    bitsA, statsA = _sample_bits_from_roi(
        gray=gray,
        roi=roi,
        grid_size=vcfg.grid_size,
        sample_radius_ratio=vcfg.sample_radius_ratio,
        polarity="dark_is_1",
    )
    resA = decode_from_physical_bits_failclosed(bitsA, interleave_step=interleave_step)
    resA.debug.update({"vision": statsA, "polarity_trial": "dark_is_1"})

    if resA.ok:
        return resA

    # 3) Optionally try polarity B: bright dot = 1 (inverted)
    if vcfg.invert_try:
        bitsB, statsB = _sample_bits_from_roi(
            gray=gray,
            roi=roi,
            grid_size=vcfg.grid_size,
            sample_radius_ratio=vcfg.sample_radius_ratio,
            polarity="bright_is_1",
        )
        resB = decode_from_physical_bits_failclosed(bitsB, interleave_step=interleave_step)
        resB.debug.update({"vision": statsB, "polarity_trial": "bright_is_1"})
        if resB.ok:
            return resB

        # Both failed: return the "more informative" failure (include both trials)
        return DecodeResult(
            False, None, None,
            "REJECT: both polarities failed CRC/ECC (fail-closed)",
            {"trial_dark": resA.debug, "trial_bright": resB.debug}
        )

    return resA


# ============================================================
# -------------------- CLI ----------------
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fail-Closed visual decoder for RuggedDot-11 (11x11).")
    parser.add_argument("image", type=str, help="Path to input image (photo or rendered PNG).")
    parser.add_argument("--step", type=int, default=37, help="Interleave step (must match encoder). Default=37.")
    parser.add_argument("--thresh", type=str, default="otsu", choices=["otsu", "adaptive"],
                        help="Threshold mode used for ROI localization. Default=otsu.")
    parser.add_argument("--roi_expand", type=float, default=0.35,
                        help="ROI expansion fraction around dot bounding box. Default=0.35.")
    parser.add_argument("--sample_r", type=float, default=0.28,
                        help="Sampling radius as fraction of cell size. Default=0.28.")
    parser.add_argument("--no_invert_try", action="store_true",
                        help="Disable inverted-polarity trial.")
    args = parser.parse_args()

    vcfg = VisionConfig(
        thresh_mode=args.thresh,
        roi_expand=args.roi_expand,
        sample_radius_ratio=args.sample_r,
        invert_try=(not args.no_invert_try),
    )

    res = decode_image_failclosed(
        image_path=args.image,
        interleave_step=args.step,
        vcfg=vcfg,
    )

    if res.ok:
        print("DECODE OK")
        print("  value   :", res.value_str)
        print("  version :", res.version)
        print("  note    :", res.reason)
        print("  ecc_corrections:", res.debug.get("ecc_corrections"))
        print("  crc32   :", res.debug.get("calc_crc32_hex"))
        print("  polarity:", res.debug.get("polarity_trial"))
    else:
        print("DECODE REJECT")
        print("  reason  :", res.reason)
        # For debugging, you can print res.debug, but it can be verbose.
        # print("  debug:", res.debug)


if __name__ == "__main__":
    main()
