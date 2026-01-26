# vision_stage2_decode_number.py
#
# ============================================================
# RuggedDot-11 — Stage 2 Decoder (Fail-Closed)
#
# Calls Stage1 (vision_stage1_extract_matrix.py) to get a 11x11 matrix,
# then decodes it into the numeric ID using:
#   - de-interleaving (same STEP as encoder)
#   - sentinel check
#   - Hamming(15,11) decode (single-bit correction per codeword)
#   - CRC32 gate (NO FALSE ACCEPT)
#
# OUTPUT:
#   - If success: prints decoded decimal number
#   - If fail: prints "DECODE REJECT" with reason
#
# Dependencies:
#   pip install opencv-python numpy
#
# ------------------------------------------------------------

from __future__ import annotations

import sys
import zlib
from typing import List, Tuple, Optional

import cv2
import numpy as np

# Import your Stage1 module (must be in same folder)
import vision_stage1_extract_matrix as S1


# =========================
# USER SETTINGS
# =========================

# If you don't pass argv, edit it here.
IMAGE_PATH = "ruggeddot11_1234512345.png"

# Must match encoder (the STEP used in make_interleaver)
INTERLEAVE_STEP = 37

# Your encoder layout:
#   logical121 = code120 + [sentinel=1]
#   code120 = 8 * Hamming(15,11) codewords
#   info88 = data54 + ver2 + crc32
#   crc32 is computed over (data54 + ver2) with zlib.crc32 on bits_to_bytes packing
EXPECTED_SENTINEL = 1


# ============================================================
# Bit packing helpers (MUST MATCH encoder exactly)
# ============================================================

def bits_to_int(bits: List[int]) -> int:
    x = 0
    for b in bits:
        x = (x << 1) | (b & 1)
    return x


def int_to_bits(x: int, nbits: int) -> List[int]:
    return [(x >> (nbits - 1 - i)) & 1 for i in range(nbits)]


def bits_to_bytes(bits: List[int]) -> bytes:
    """
    Pack bits into bytes (big-endian per byte), right-pad zeros to byte boundary.
    MUST be identical to encoder's bits_to_bytes().
    """
    pad = (-len(bits)) % 8
    b2 = bits + [0] * pad
    out = bytearray()
    for i in range(0, len(b2), 8):
        out.append(bits_to_int(b2[i:i + 8]))
    return bytes(out)


def crc32_of_bits(bits: List[int]) -> int:
    """CRC32 gate for no-false-accept."""
    return zlib.crc32(bits_to_bytes(bits)) & 0xFFFFFFFF


# ============================================================
# Interleaving / de-interleaving (must match encoder)
# ============================================================

def make_interleaver(step: int, n: int) -> List[int]:
    """
    P[logical] = physical = (logical * step) % n
    Requires gcd(step,n)=1
    """
    def gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a

    if gcd(step, n) != 1:
        raise ValueError(f"step={step} must be coprime with n={n}")
    return [(i * step) % n for i in range(n)]


def deinterleave_physical_to_logical(physical: List[int], step: int) -> List[int]:
    """
    Encoder did: physical[P[logical]] = logical_bit
    So decoder recovers: logical[logical] = physical[P[logical]]
    """
    n = len(physical)
    P = make_interleaver(step, n)
    logical = [0] * n
    for logical_idx, phys_idx in enumerate(P):
        logical[logical_idx] = int(physical[phys_idx]) & 1
    return logical


# ============================================================
# Hamming(15,11) decode (single-error correction, fail-closed)
# ============================================================

HAMMING_DATA_POS = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
HAMMING_PARITY_POS = [1, 2, 4, 8]

def hamming15_11_syndrome(code15: List[int]) -> int:
    """
    Compute syndrome (1..15). 0 means parity checks pass.

    For even parity:
      For each parity position p, compute XOR of all code[i] where (i&p)!=0.
      If result is 1, that parity check fails -> syndrome accumulates p.
    """
    if len(code15) != 15:
        raise ValueError("code15 must be length 15")

    # Use 1-based indexing for clarity
    code = [0] + [int(b) & 1 for b in code15]

    syn = 0
    for p in HAMMING_PARITY_POS:
        parity = 0
        for i in range(1, 16):
            if i & p:
                parity ^= code[i]
        if parity != 0:
            syn |= p
    return syn


def hamming15_11_decode_failclosed(code15: List[int]) -> Tuple[Optional[List[int]], int]:
    """
    Decode one Hamming(15,11) codeword in fail-closed manner.

    Returns:
      (data11, corrected_pos)
        - data11: list of 11 bits, or None if reject
        - corrected_pos: 0 if no correction, 1..15 if corrected that position
    Fail-closed policy:
      - If syndrome==0 => accept as-is
      - If syndrome in 1..15 => flip that bit, then re-check syndrome must become 0
        otherwise reject (ill-conditioned / too many errors)
    """
    if len(code15) != 15:
        return None, 0

    code = [int(b) & 1 for b in code15]
    syn = hamming15_11_syndrome(code)

    if syn == 0:
        corrected_pos = 0
    else:
        # single-bit correction attempt
        pos = syn  # 1..15
        code[pos - 1] ^= 1
        # must become consistent, else reject
        syn2 = hamming15_11_syndrome(code)
        if syn2 != 0:
            return None, 0
        corrected_pos = pos

    # Extract data bits from positions
    code1 = [0] + code
    data11 = [code1[pos] for pos in HAMMING_DATA_POS]
    return data11, corrected_pos


# ============================================================
# Stage1 call: image -> 11x11 matrix
# ============================================================

def stage1_extract_matrix(image_path: str) -> List[List[int]]:
    """
    Call Stage1's existing functions (not re-implementing the logic).
    This mirrors S1.main() but returns the matrix instead of printing it.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Use Stage1 methods
    bw = S1.otsu_binary_foreground(gray)
    bw = S1.morph_close(bw, S1.MORPH_CLOSE_K)

    comps = S1.get_components(bw)
    h, w = gray.shape[:2]
    picked = S1.pick_fiducials_three_corners(comps, img_w=w, img_h=h)
    if picked is None:
        raise ValueError("stage1 reject: cannot locate 3 fiducials (TL/TR/BL)")

    TLc, TRc, BLc = picked
    TLp, TRp, BLp = S1.fiducial_outer_points(TLc, TRc, BLc)

    data_square = S1.warp_to_data_square(
        gray=gray,
        TLp=TLp, TRp=TRp, BLp=BLp,
        out_size=S1.WARP_SIZE,
        gap_cells=S1.FID_GAP_CELLS,
    )

    means = S1.cell_means_disk(data_square, n=S1.GRID_N, radius_ratio=S1.SAMPLE_RADIUS_RATIO)
    thr = S1.otsu_threshold_on_means(means)
    mat = S1.means_to_matrix(means, thr, dark_is_1=True)
    return mat


def matrix_to_physical_bits(mat11: List[List[int]]) -> List[int]:
    """
    Convert 11x11 matrix to 121 physical bits in row-major order.
    """
    if len(mat11) != 11 or any(len(row) != 11 for row in mat11):
        raise ValueError("matrix must be 11x11")
    out: List[int] = []
    for r in range(11):
        for c in range(11):
            out.append(int(mat11[r][c]) & 1)
    return out


# ============================================================
# Stage2 decode: 121 physical bits -> number
# ============================================================

def decode_bits121_failclosed(physical121: List[int], step: int) -> Tuple[bool, str]:
    """
    Returns (ok, message_or_number).
      - ok=False => message_or_reason
      - ok=True  => decoded decimal string (no leading zeros preserved)

    Fail-closed behavior:
      - any inconsistency => reject
    """
    if len(physical121) != 121:
        return False, "reject: physical bits length != 121"

    logical121 = deinterleave_physical_to_logical(physical121, step=step)

    # Sentinel is the last logical bit
    sentinel = logical121[120]
    if sentinel != EXPECTED_SENTINEL:
        return False, f"reject: sentinel != {EXPECTED_SENTINEL} (got {sentinel})"

    code120 = logical121[:120]

    # Split into 8 codewords of 15
    info_bits: List[int] = []
    total_corrected = 0

    for i in range(8):
        cw = code120[i * 15:(i + 1) * 15]
        data11, corrected_pos = hamming15_11_decode_failclosed(cw)
        if data11 is None:
            return False, f"reject: Hamming decode failed at codeword {i}"
        info_bits.extend(data11)
        if corrected_pos != 0:
            total_corrected += 1

    if len(info_bits) != 88:
        return False, "reject: info bits length != 88"

    data54 = info_bits[:54]
    ver2 = info_bits[54:56]
    crc32_bits = info_bits[56:88]

    crc_expected = bits_to_int(crc32_bits)
    crc_got = crc32_of_bits(data54 + ver2)

    if crc_got != crc_expected:
        return False, "reject: CRC32 mismatch (fail-closed)"

    # Decode numeric
    num_int = bits_to_int(data54)

    # NOTE: leading zeros from original input cannot be recovered
    # because encoder used int(num_str). If you need to preserve
    # leading zeros, you must include a length field in the payload.
    return True, str(num_int)


# ============================================================
# Main
# ============================================================

def main() -> None:
    path = IMAGE_PATH
    if len(sys.argv) >= 2:
        path = sys.argv[1]

    try:
        mat = stage1_extract_matrix(path)
        physical = matrix_to_physical_bits(mat)
    except Exception as e:
        print("DECODE REJECT")
        print("  reason:", str(e))
        return

    ok, msg = decode_bits121_failclosed(physical, step=INTERLEAVE_STEP)
    if ok:
        print("DECODE OK")
        print("  number:", msg)
    else:
        print("DECODE REJECT")
        print("  reason:", msg)


if __name__ == "__main__":
    main()