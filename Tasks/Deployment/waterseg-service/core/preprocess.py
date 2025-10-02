import numpy as np
import tifffile as tiff
from .bands import IDX, MNDWI_POS

def _ensure_hwc(arr: np.ndarray) -> np.ndarray:
    # If CHW (C,H,W) and looks like <=12 channels, move to HWC
    if arr.ndim == 3 and arr.shape[0] <= 12 and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr.astype(np.float32)

def _compute_mndwi(hwc: np.ndarray) -> np.ndarray:
    g  = hwc[..., IDX["green"]]
    s1 = hwc[..., IDX["swir1"]]
    m  = (g - s1) / (g + s1 + 1e-6)
    return np.clip(m, -1.0, 1.0).astype(np.float32)

def _per_image_percentile_scale(x: np.ndarray, idxs):
    """
    Percentile [2,98] scaling per-image for selected channels.
    x: HWC (float32), scaled in-place on those idxs.
    """
    p2  = np.percentile(x[..., idxs],  2, axis=(0,1))
    p98 = np.percentile(x[..., idxs], 98, axis=(0,1))
    denom = (p98 - p2 + 1e-6)
    for j, ch in enumerate(idxs):
        x[..., ch] = np.clip((x[..., ch] - p2[j]) / denom[j], 0, 1)
    return x

def _process_qa(x: np.ndarray):
    qa = x[..., IDX["qa"]]
    mx = qa.max()
    if mx > 0:
        qa = qa / mx if mx > 1.0 else qa
    x[..., IDX["qa"]] = np.clip(qa, 0, 1)

def _process_worldcover(x: np.ndarray):
    wc = x[..., IDX["worldcover"]]
    x[..., IDX["worldcover"]] = (wc == 80).astype(np.float32)

def _process_water_occ(x: np.ndarray):
    wocc = x[..., IDX["water_occ"]]
    if wocc.max() > 1.0:
        wocc = wocc / 100.0
    x[..., IDX["water_occ"]] = np.clip(wocc, 0, 1)

def read_stack_plus_mndwi(image_path: str) -> np.ndarray:
    """
    Reproduces your Dataset logic:
      - ensure HWC
      - append MNDWI as 13th band
      - percentile-scale spectral + DEM + MNDWI
      - normalize QA, binarize worldcover water=80, scale water_occ 0..1
      - return CHW (13,H,W), float32
    """
    arr = tiff.imread(image_path)
    hwc = _ensure_hwc(arr)

    # Append MNDWI
    mndwi = _compute_mndwi(hwc)[..., None]
    hwc13 = np.concatenate([hwc, mndwi], axis=-1)  # H,W,13

    # Percentile scaling for spectral + DEM + MNDWI
    spec_dem_mndwi_idx = [
        IDX["coastal"], IDX["blue"], IDX["green"], IDX["red"],
        IDX["nir"], IDX["swir1"], IDX["swir2"],
        IDX["merit_dem"], IDX["cop_dem"], MNDWI_POS
    ]
    hwc13 = _per_image_percentile_scale(hwc13, spec_dem_mndwi_idx)

    # QA/worldcover/water_occ processing
    _process_qa(hwc13)
    _process_worldcover(hwc13)
    _process_water_occ(hwc13)

    # To CHW
    chw = np.transpose(hwc13, (2, 0, 1)).astype(np.float32)  # (13,H,W)
    return chw
