# FailClosedDotCode — Encode / Decode

A small project for encoding and decoding **FailClosedDotCode** (a dot-matrix / fiducial-style code), with utilities for generating the code and reading it back from images.

## What’s in here
- **Encoder**: generate a dot-matrix code from input payload/parameters
- **Decoder**: detect the code in an image, normalize/warp it, and recover the underlying bit matrix / payload
- **Vision utilities**: helpers for detection, alignment, and sampling (OpenCV-based)

## Requirements
- Python 3.10+ (recommended)
- Common deps (typical): `opencv-python`, `numpy`  
  Install:
  ```bash
  pip install -r requirements.txt
