# rugged_dotcode_11x11_failclosed_en.py
#
# ============================================================
# RuggedDot-11 (Fail-Closed) Encoder + Renderer (INTERNAL GRID ONLY)
# ============================================================
#
# INPUT : a pure numeric string with 1~16 decimal digits
# OUTPUT: an 11x11 dot-matrix PNG (optionally also an inverted-polarity PNG)
#
# ---------------------------
# Design goals (Fail-Closed)
# ---------------------------
# (1) Never output a wrong ID (NO FALSE ACCEPT):
#     Any uncertainty / inconsistency / check failure MUST result in "reject",
#     not a wrong decoded number.
#
# (2) Read as often as possible (HIGH DECODE SUCCESS):
#     We tolerate a small number of random bit errors and repair them with ECC.
#     We also mitigate local, contiguous damage (smear/burn/occlusion) by interleaving.
#
# Important: This script only generates the INTERNAL 11x11 data grid.
# In practical deployment you should add an outer fiducial frame / border / corner marks
# for robust localization and perspective correction on fabric (stretch, tilt, wrinkles).
#
# ============================================================
# Bit budget and layout: 121 bits = 11 x 11
# ============================================================
# We fill the 121 cells with 121 "physical bits".
#
# (A) 120 bits  = 8 codewords of Hamming(15,11), concatenated
#     - Each Hamming(15,11) codeword carries 11 "information bits" (payload)
#     - Each codeword expands 11 -> 15 bits, allowing correction of up to 1 random bit error
#       per 15-bit codeword (classic Hamming single-error-correcting code).
#
# (B) 1 bit     = sentinel bit (fixed to 1)
#     - Used as a fast sanity check / early rejection at decode time.
#     - If it is not 1 (due to damage or severe misread), decoder rejects immediately.
#
# Total info bits carried by the 8 Hamming blocks:
#     8 * 11 = 88 info bits
#
# We define the 88 info bits as:
#   54 bits : data54   (integer encoding of a 16-digit decimal number)
#    2 bits : ver2     (version field, 0..3 for future evolution)
#   32 bits : crc32    (CRC32 of data54 || ver2, used as a strong "no-false-accept" gate)
#
# 54 + 2 + 32 = 88 bits exactly, so we can slice into eight 11-bit blocks.
#
# ============================================================
# Interleaving (Permutation over 121 positions)
# ============================================================
# The 121 "logical" bits (120 ECC bits + 1 sentinel) are NOT placed row-major directly.
# We apply a reversible permutation so that adjacent physical damage becomes dispersed
# across the logical stream, which:
#   - reduces the chance that one local smear destroys multiple bits within the same
#     Hamming codeword (which would exceed its 1-bit correction capability),
#   - increases real-world decode success on fabric (stains, burn spots, scratches).
#
# The permutation used is:
#   physical_pos = (logical_pos * STEP) mod 121
# where gcd(STEP, 121) = 1 to guarantee a full permutation (bijective mapping).
#
# Decoder must use the same STEP to de-interleave (apply inverse permutation).
#
# ============================================================
# Rendering
# ============================================================
# For each physical bit:
#   bit = 1  -> draw a filled circular dot
#   bit = 0  -> leave blank
#
# Optionally invert polarity:
#   invert=False: bit1=dot, bit0=blank
#   invert=True : bit1=blank, bit0=dot
#
# Inversion is useful because some marking processes invert contrast
# (e.g., engraving can make marks appear brighter or darker depending on material).
#
# ============================================================
# Dependencies
# ============================================================
# - Python 3.8+
# - Pillow (PIL): pip install pillow
#
# ============================================================

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from PIL import Image, ImageDraw


# ============================================================
# Bit utilities
# ============================================================

def int_to_bits(x: int, nbits: int) -> List[int]:
    """
    Convert a non-negative integer x to a list of bits (length = nbits),
    using big-endian bit order (MSB first).

    Example:
      x=5, nbits=4  ->  [0, 1, 0, 1]   (0101)
    """
    if x < 0:
        raise ValueError("int_to_bits expects a non-negative integer")
    return [(x >> (nbits - 1 - i)) & 1 for i in range(nbits)]


def bits_to_int(bits: List[int]) -> int:
    """
    Convert a list of bits in big-endian order (MSB first) into an integer.

    Example:
      [0, 1, 0, 1] -> 5
    """
    x = 0
    for b in bits:
        x = (x << 1) | (b & 1)
    return x


def bits_to_bytes(bits: List[int]) -> bytes:
    """
    Pack a list of bits into bytes, big-endian within each byte (MSB first).

    If len(bits) is not a multiple of 8, we right-pad zeros to the next byte boundary.
    This is important for CRC computation: encoder and decoder must use the SAME packing
    scheme, including padding direction and bit order.

    Example:
      bits = [1,0,0,0,0,0,0,1] -> b'\\x81'
    """
    pad = (-len(bits)) % 8
    b2 = bits + [0] * pad  # right-pad zeros
    out = bytearray()
    for i in range(0, len(b2), 8):
        out.append(bits_to_int(b2[i:i + 8]))
    return bytes(out)


# ============================================================
# CRC32 (strong "no false accept" gate)
# ============================================================

def crc32_of_bits(bits: List[int]) -> int:
    """
    Compute CRC32 over the packed bytes of a bit-list.

    We use zlib.crc32, which implements standard CRC-32 (IEEE 802.3) in reflected form.
    The returned value is masked to 32-bit unsigned.

    Why CRC32 here?
    - Hamming ECC can sometimes "mis-correct" when errors exceed its capability.
    - CRC32 acts as a final gate: if CRC mismatches, we reject (fail-closed).
    - For random incorrect payloads, chance to pass CRC32 is about 2^-32 (~1/4.29e9),
      which is extremely small for "no false accept" requirements.
    """
    data = bits_to_bytes(bits)
    return zlib.crc32(data) & 0xFFFFFFFF


# ============================================================
# Hamming(15,11) encoder (even parity)
# ============================================================
# Hamming(15,11) layout using positions 1..15:
#   Parity bit positions: 1, 2, 4, 8
#   Data bit positions  : 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15
#
# Even parity rule:
#   parity bit at position p covers all positions i such that (i & p) != 0, including p itself.
# Here we compute parity so that the XOR of covered bits is 0 (even parity).

HAMMING_DATA_POS = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
HAMMING_PARITY_POS = [1, 2, 4, 8]


def hamming15_11_encode(data11: List[int]) -> List[int]:
    """
    Encode 11 data bits into a 15-bit Hamming(15,11) codeword (single-error-correcting).

    Correction capability (at decode time):
      - can correct up to 1 bit error within this 15-bit codeword
      - can detect (but not reliably correct) many multi-bit error patterns

    Parameters:
      data11: list of 11 bits (0/1)

    Returns:
      list of 15 bits (positions 1..15 flattened to Python list index 0..14)
    """
    if len(data11) != 11:
        raise ValueError("hamming15_11_encode expects exactly 11 bits")

    # Use indices 1..15 for clarity; index 0 unused
    code = [0] * 16

    # 1) Place data bits into the non-parity positions
    for bit, pos in zip(data11, HAMMING_DATA_POS):
        code[pos] = bit & 1

    # 2) Compute parity bits (even parity)
    # For each parity position p, compute XOR of all code[i] where i has that bit set.
    # We exclude i==p when summing, then set code[p] to make overall parity even.
    for p in HAMMING_PARITY_POS:
        parity = 0
        for i in range(1, 16):
            if (i & p) and i != p:
                parity ^= code[i]
        code[p] = parity

    return code[1:]  # 15 bits


# ============================================================
# Interleaver permutation over 121 positions
# ============================================================

def make_interleaver(step: int = 37, n: int = 121) -> List[int]:
    """
    Build a permutation P of length n:
      P[logical_index] = physical_index = (logical_index * step) % n

    Requirement: gcd(step, n) == 1.
    If not coprime, the mapping would repeat early and not cover all positions.

    This permutation is bijective, hence invertible.
    Decoder should build inverse permutation to de-interleave.

    Note: n=121 = 11^2, so any step not divisible by 11 will be coprime with 121.
    """
    def gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a

    if gcd(step, n) != 1:
        raise ValueError(f"Interleave step={step} must be coprime with n={n}")

    return [(i * step) % n for i in range(n)]


# ============================================================
# Main encoding pipeline (Fail-Closed oriented)
# ============================================================

def encode_numeric_to_bits121_physical(
    num_str: str,
    version: int = 1,
    interleave_step: int = 37,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Encode a decimal numeric string into 121 physical bits for a 11x11 dot grid.

    Pipeline:
      1) Validate input: 1~16 digits
      2) data54  = integer(num_str) encoded in 54 bits (MSB first)
      3) ver2    = version encoded in 2 bits (0..3)
      4) crc32   = CRC32(data54 || ver2) as 32 bits
      5) info88  = data54 || ver2 || crc32 (88 bits)
      6) ECC     = split info88 into 8 blocks of 11 bits -> Hamming(15,11) -> 8*15=120 bits
      7) sentinel= 1 bit appended (fixed 1)
      8) interleave logical121 bits into physical positions (row-major indices 0..120)

    Returns:
      physical121: list of 121 bits in row-major physical placement order:
                  index 0 is (row=0,col=0), index 120 is (row=10,col=10)
      meta: debug metadata (safe to print/log, not required to embed in the tag)
    """
    # ---- (0) Validate numeric range ----
    if not re.fullmatch(r"\d{1,16}", num_str):
        raise ValueError("Input must be a pure decimal string with 1~16 digits.")

    n = int(num_str)
    if n >= 10**16:
        # Should not happen if regex is enforced, but keep explicit guard
        raise ValueError("Input out of range: must be < 10^16 (max 16 digits).")

    # version is 2 bits (0..3)
    if not (0 <= version <= 3):
        raise ValueError("version must be within 0..3 (2 bits).")

    # ---- (1) data54 ----
    # 16-digit decimal fits in 54 bits because 10^16 < 2^54.
    data54 = int_to_bits(n, 54)

    # ---- (2) ver2 ----
    ver2 = int_to_bits(version, 2)

    # ---- (3) crc32 over (data54 + ver2) ----
    crc_in = data54 + ver2
    crc32_val = crc32_of_bits(crc_in)
    crc32_bits = int_to_bits(crc32_val, 32)

    # ---- (4) info88 ----
    info88 = data54 + ver2 + crc32_bits
    assert len(info88) == 88

    # ---- (5) ECC: 8 * Hamming(15,11) -> 120 bits ----
    # Each codeword can correct <=1 random bit error in its 15-bit block.
    code120: List[int] = []
    for i in range(8):
        block11 = info88[i * 11:(i + 1) * 11]
        code120.extend(hamming15_11_encode(block11))
    assert len(code120) == 120

    # ---- (6) Sentinel bit ----
    # Fast sanity check. Decoder should reject if not 1.
    sentinel_bit = 1
    logical121 = code120 + [sentinel_bit]
    assert len(logical121) == 121

    # ---- (7) Interleave logical -> physical ----
    P = make_interleaver(step=interleave_step, n=121)
    physical121 = [0] * 121
    for logical_idx, phys_idx in enumerate(P):
        physical121[phys_idx] = logical121[logical_idx]

    meta = {
        "scheme_name": "RuggedDot-11 Fail-Closed",
        "num_str": num_str,
        "num_int": n,
        "version": version,
        "crc32_hex": hex(crc32_val),
        "interleave_step": interleave_step,
        "sentinel": sentinel_bit,
        "decoder_contract": (
            "Decoder MUST: (1) de-interleave, (2) check sentinel==1, "
            "(3) Hamming decode each block (<=1 correction), (4) recompute CRC32 "
            "over (data54||ver2) and compare; if any step fails -> REJECT."
        ),
    }
    return physical121, meta


# ============================================================
# Rendering (11x11 dot grid)
# ============================================================

@dataclass
class RenderConfig:
    """
    Rendering configuration (affects only the PNG appearance).

    cell:
      Pixel size of each logical grid cell.
      Larger cell -> larger dots -> easier for laser engraving and camera recognition.

    dot_ratio:
      Dot diameter as a fraction of cell size. Typical 0.6~0.8.
      Too large -> dots may touch/merge if printed/engraved with blur.
      Too small -> dots may disappear after washing / low contrast.

    margin:
      White border around the grid, in pixels.
      In real tags you often want a quiet zone to help segmentation.

    invert:
      If True, invert polarity: bit=1 -> blank, bit=0 -> dot.
      Useful for materials/processes where engraving flips perceived contrast.

    background / foreground:
      Grayscale values (0=black, 255=white) for background and dots.
    """
    cell: int = 28
    dot_ratio: float = 0.72
    margin: int = 84
    invert: bool = False
    background: int = 255
    foreground: int = 0

    # ---- Corner L fiducials (solid L blocks, not dots) ----
    draw_corner_L: bool = True
    fiducial_gap_cells: float = 0.8   # how far the L sits away from the data grid (in cell units)

    fiducial_arm_len_cells: float = 2  # L arm length (in cell units)
    fiducial_thickness_cells: float = 0.6  # L stroke thickness (in cell units)


def _draw_filled_dot(draw: ImageDraw.ImageDraw, cx: int, cy: int, radius: int, color: int) -> None:
    """Draw one filled circular dot centered at (cx, cy)."""
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)

def draw_three_L_corner_fiducials(
    draw: ImageDraw.ImageDraw,
    cfg: RenderConfig,
    grid_origin_x: int,
    grid_origin_y: int,
    grid_size_cells: int = 11,
) -> None:
    """
    Draw 3 SOLID L-shaped corner fiducials in the quiet zone (margin),
    NOT inside the 11x11 data grid.

    Corners used:
      - Top-Left (TL)
      - Top-Right (TR)
      - Bottom-Left (BL)
    Bottom-Right is intentionally empty for orientation ("missing corner" cue).

    Each L is drawn as two filled rectangles (horizontal arm + vertical arm).
    This produces a continuous, solid L marker (industrial fiducial style).
    """
    if not cfg.draw_corner_L:
        return

    # Safety: need enough margin to draw L without clipping
    if cfg.margin < cfg.cell:
        raise ValueError("RenderConfig.margin should be >= cell when draw_corner_L=True")

    N = grid_size_cells
    grid_w = N * cfg.cell
    grid_h = N * cfg.cell

    # Convert cell-based params to pixels
    gap = int(round(cfg.cell * cfg.fiducial_gap_cells))
    arm = int(round(cfg.cell * cfg.fiducial_arm_len_cells))
    th  = max(1, int(round(cfg.cell * cfg.fiducial_thickness_cells)))

    # Grid bounding box in image coordinates
    gx0 = grid_origin_x
    gy0 = grid_origin_y
    gx1 = grid_origin_x + grid_w
    gy1 = grid_origin_y + grid_h

    # Helper to draw a solid L given corner anchor and directions
    # dx, dy define where the arms extend (+1 or -1)
    def draw_L_at(anchor_x: int, anchor_y: int, dx: int, dy: int) -> None:
        """
        anchor_x, anchor_y: the OUTER corner point of the L (near the grid corner but in margin)
        dx: horizontal arm direction (+1 right, -1 left)
        dy: vertical arm direction (+1 down,  -1 up)
        """
        # Horizontal arm rectangle
        # It starts at anchor and extends by 'arm' in dx direction
        if dx > 0:
            x0, x1 = anchor_x, anchor_x + arm
        else:
            x0, x1 = anchor_x - arm, anchor_x
        y0, y1 = anchor_y - th // 2, anchor_y + (th - th // 2)

        draw.rectangle([x0, y0, x1, y1], fill=cfg.foreground)

        # Vertical arm rectangle
        if dy > 0:
            y0v, y1v = anchor_y, anchor_y + arm
        else:
            y0v, y1v = anchor_y - arm, anchor_y
        x0v, x1v = anchor_x - th // 2, anchor_x + (th - th // 2)

        draw.rectangle([x0v, y0v, x1v, y1v], fill=cfg.foreground)

    # Place anchors just outside the grid corners (in margin)
    # TL: anchor near (gx0, gy0) but shifted outward by gap
    TL = (gx0 - gap, gy0 - gap)
    TR = (gx1 + gap, gy0 - gap)
    BL = (gx0 - gap, gy1 + gap)

    # Draw 3 Ls
    draw_L_at(TL[0], TL[1], dx=+1, dy=+1)  # TL extends right and down
    draw_L_at(TR[0], TR[1], dx=-1, dy=+1)  # TR extends left and down
    draw_L_at(BL[0], BL[1], dx=+1, dy=-1)  # BL extends right and up


def render_11x11_dotgrid(
    bits121_physical: List[int],
    out_path: str,
    cfg: RenderConfig = RenderConfig(),
) -> None:
    """
    Render the 121 physical bits into an 11x11 dot matrix image.

    Placement convention:
      bits121_physical is row-major over 11x11:
        idx = row * 11 + col, where row,col in [0..10].

    Rendering rule:
      bit=1 -> draw dot (unless invert=True)
      bit=0 -> blank
    """
    if len(bits121_physical) != 121:
        raise ValueError("render_11x11_dotgrid expects exactly 121 bits.")

    N = 11
    W = cfg.margin * 2 + cfg.cell * N
    H = cfg.margin * 2 + cfg.cell * N

    img = Image.new("L", (W, H), color=cfg.background)
    draw = ImageDraw.Draw(img)
    grid_origin_x = cfg.margin
    grid_origin_y = cfg.margin

    dot_d = int(cfg.cell * cfg.dot_ratio)
    r = dot_d // 2

    for idx, b in enumerate(bits121_physical):
        row = idx // N
        col = idx % N

        bit_on = (b == 1)
        if cfg.invert:
            bit_on = not bit_on

        if bit_on:
            # Center of current cell
            cx = cfg.margin + col * cfg.cell + cfg.cell // 2
            cy = cfg.margin + row * cfg.cell + cfg.cell // 2

            # Draw filled circle
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=cfg.foreground)

    grid_origin_x = cfg.margin
    grid_origin_y = cfg.margin
    ...
    draw_three_L_corner_fiducials(draw, cfg, grid_origin_x, grid_origin_y, 11)

    img.save(out_path)


# ============================================================
# CLI entry point
# ============================================================

def main() -> None:
    """
    Simple CLI:
      - Ask for a numeric string
      - Encode into 121 bits
      - Render normal + inverted PNG for material polarity experiments
    """
    num = input("Enter 1~16 digits: ").strip()

    bits121, meta = encode_numeric_to_bits121_physical(
        num_str=num,
        version=1,             # keep fixed now; change in future revisions
        interleave_step=37,    # must be coprime with 121
    )

    # Normal polarity
    cfg = RenderConfig(cell=28, dot_ratio=0.72, margin=36, invert=False)
    out1 = f"ruggeddot11_{num}.png"
    render_11x11_dotgrid(bits121, out1, cfg)

    # Inverted polarity (optional)
    cfg_inv = RenderConfig(cell=28, dot_ratio=0.72, margin=36, invert=True)
    out2 = f"ruggeddot11_{num}_inv.png"
    render_11x11_dotgrid(bits121, out2, cfg_inv)

    print("\nGenerated:")
    print(" -", out1)
    print(" -", out2)

    print("\nMetadata (for debugging / traceability; not printed on the tag):")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
