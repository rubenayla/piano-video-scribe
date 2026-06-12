"""Per-key sampling backend with auto-selected acceleration.

Picks torch (MPS / CUDA / CPU) if available, otherwise vectorized NumPy.
Both backends share the same interface and produce numerically equivalent
output. Users without torch installed get the NumPy path automatically.
"""

import os
import numpy as np

_FORCE = os.environ.get("PVS_BACKEND", "").lower()  # "numpy" / "torch" / ""

if _FORCE == "numpy":
    _TORCH_OK = False
    _DEVICE = None
else:
    try:
        import torch
        _TORCH_OK = True
        if torch.backends.mps.is_available():
            _DEVICE = "mps"
        elif torch.cuda.is_available():
            _DEVICE = "cuda"
        else:
            _DEVICE = "cpu"
    except ImportError:
        _TORCH_OK = False
        _DEVICE = None


def get_backend_name() -> str:
    return f"torch-{_DEVICE}" if _TORCH_OK else "numpy"


class KeySampler:
    """Vectorized per-key reductions over an HSV frame.

    Pre-computes a flat (pixel_index, key_id) map at construction time.
    Each call to per_key_mean_sat reduces the whole frame in one
    bincount / scatter_add — no Python loop over the 88 keys.
    """

    def __init__(self, pitch_rects, frame_h: int, frame_w: int,
                 v_min: int = 30):
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.v_min = v_min

        pixel_idx_chunks = []
        key_id_chunks = []
        pitches = []
        rect_widths = []

        for k, (pitch, x1, x2, y1, y2) in enumerate(pitch_rects):
            pitches.append(pitch)
            x1c, y1c = max(0, x1), max(0, y1)
            x2c = min(frame_w, x2)
            y2c = min(frame_h, y2)
            if x2c <= x1c or y2c <= y1c:
                rect_widths.append(0)
                continue
            ys = np.arange(y1c, y2c, dtype=np.int64)
            xs = np.arange(x1c, x2c, dtype=np.int64)
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            flat = (yy * frame_w + xx).ravel()
            pixel_idx_chunks.append(flat)
            key_id_chunks.append(np.full(flat.size, k, dtype=np.int64))
            rect_widths.append(x2c - x1c)

        self.pitches = np.asarray(pitches, dtype=np.int32)
        self.K = len(pitches)
        self.rect_widths = np.asarray(rect_widths, dtype=np.int32)
        self._np_pixel_idx = (np.concatenate(pixel_idx_chunks)
                              if pixel_idx_chunks else np.zeros(0, dtype=np.int64))
        self._np_key_id = (np.concatenate(key_id_chunks)
                           if key_id_chunks else np.zeros(0, dtype=np.int64))

        if _TORCH_OK:
            self._t_pixel_idx = torch.from_numpy(self._np_pixel_idx).to(_DEVICE)
            self._t_key_id = torch.from_numpy(self._np_key_id).to(_DEVICE)

    def per_key_mean_sat(self, hsv: np.ndarray) -> np.ndarray:
        """Mean saturation per key, averaged over pixels with V >= v_min.

        Returns a (K,) float32 array indexed by self.pitches order.
        """
        if _TORCH_OK:
            return self._torch_mean_sat(hsv)
        return self._numpy_mean_sat(hsv)

    def _numpy_mean_sat(self, hsv: np.ndarray) -> np.ndarray:
        S = hsv[:, :, 1].ravel()
        V = hsv[:, :, 2].ravel()
        s_pix = S[self._np_pixel_idx].astype(np.float32)
        v_pix = V[self._np_pixel_idx]
        bright = (v_pix >= self.v_min).astype(np.float32)
        sums = np.bincount(self._np_key_id,
                           weights=s_pix * bright, minlength=self.K)
        cnts = np.bincount(self._np_key_id,
                           weights=bright, minlength=self.K)
        out = np.zeros(self.K, dtype=np.float32)
        nz = cnts > 0
        out[nz] = (sums[nz] / cnts[nz]).astype(np.float32)
        return out

    def _torch_mean_sat(self, hsv: np.ndarray) -> np.ndarray:
        S = torch.from_numpy(hsv[:, :, 1]).to(_DEVICE).flatten().to(torch.float32)
        V = torch.from_numpy(hsv[:, :, 2]).to(_DEVICE).flatten()
        s_pix = S[self._t_pixel_idx]
        v_pix = V[self._t_pixel_idx]
        bright = (v_pix >= self.v_min).to(torch.float32)
        sums = torch.zeros(self.K, dtype=torch.float32, device=_DEVICE)
        cnts = torch.zeros(self.K, dtype=torch.float32, device=_DEVICE)
        sums.scatter_add_(0, self._t_key_id, s_pix * bright)
        cnts.scatter_add_(0, self._t_key_id, bright)
        out = torch.where(cnts > 0, sums / cnts, torch.zeros_like(sums))
        return out.cpu().numpy()
