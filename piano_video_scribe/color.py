"""
color.py — Color sampling, classification, and detection utilities.

Functions for sampling HSV colors from video frames, classifying pixels
as right/left hand based on hue ranges, and auto-detecting the two hand
colors from falling-block waterfall regions.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Color sampling (from video frames)
# ---------------------------------------------------------------------------

def sample_color(cap, frame_idx, x_center, y_top, y_bot, half_w=10):
    """Sample the dominant saturated color at a note's position in a video frame.

    Finds the brightest, most-saturated pixel in the region, filtering out dark
    pixels (V < 80) that produce false high-saturation readings at borders.

    Returns:
        (h, s, v) floats, or (None, None, None) if no suitable pixel found.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None, None, None

    fh, fw = frame.shape[:2]
    x1, x2 = max(0, x_center - half_w), min(fw, x_center + half_w)
    y1, y2 = max(0, y_top), min(fh, y_bot)
    if y1 >= y2 or x1 >= x2:
        return None, None, None

    region = frame[y1:y2, x1:x2]
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h_ch = hsv_region[:, :, 0].flatten().astype(float)
    s_ch = hsv_region[:, :, 1].flatten().astype(float)
    v_ch = hsv_region[:, :, 2].flatten().astype(float)

    # Filter out dark pixels — they produce noise-saturation artifacts
    bright = v_ch > 80
    if not np.any(bright):
        return None, None, None

    s_bright = s_ch.copy()
    s_bright[~bright] = 0
    best = int(np.argmax(s_bright))
    return float(h_ch[best]), float(s_ch[best]), float(v_ch[best])


def sample_color_avg(hsv_frame, x_left, x_right, y_top, y_bot):
    """Sample color by averaging bright saturated pixels in a detector region.

    More robust than single-pixel max: resistant to sparkles and edge bleed.

    Args:
        hsv_frame: full frame already converted to HSV.
        x_left, x_right: horizontal bounds of the detector region.
        y_top, y_bot: vertical bounds of the sampling zone.

    Returns:
        (h, s, v) average of bright saturated pixels, or (None, None, None).
    """
    fh, fw = hsv_frame.shape[:2]
    x1, x2 = max(0, x_left), min(fw, x_right)
    y1, y2 = max(0, y_top), min(fh, y_bot)
    if y1 >= y2 or x1 >= x2:
        return None, None, None

    region = hsv_frame[y1:y2, x1:x2]
    h_ch = region[:, :, 0].flatten().astype(float)
    s_ch = region[:, :, 1].flatten().astype(float)
    v_ch = region[:, :, 2].flatten().astype(float)

    # Keep only bright AND saturated pixels (the actual colored key face)
    mask = (v_ch > 80) & (s_ch > 50)
    if np.sum(mask) < 3:  # need at least a few pixels to trust the average
        return None, None, None

    return float(np.mean(h_ch[mask])), float(np.mean(s_ch[mask])), float(np.mean(v_ch[mask]))


# ---------------------------------------------------------------------------
# Hand classification
# ---------------------------------------------------------------------------

def classify_hand(h, s, v, green_is_right=True, colors=None):
    """Classify a color sample as right hand (0), left hand (1), or unknown (None).

    Args:
        h, s, v: HSV values (OpenCV scale: H in [0,179]).
        green_is_right: if True, green = right (0), blue = left (1). Flip if needed.
            Ignored when colors dict contains 'right'/'left' keys.
        colors: dict with color keys, each containing h_min, h_max, s_min, v_min.
            Supports two formats:
            - Legacy: 'green' and 'blue' keys (uses green_is_right to map to hands)
            - Direct: 'right' and 'left' keys (maps directly, ignores green_is_right)

    Returns:
        0 (right), 1 (left), or None.
    """
    if h is None or s is None:
        return None

    if colors is None:
        colors = {
            'green': {'h_min': 40, 'h_max': 65, 's_min': 100, 'v_min': 80},
            'blue':  {'h_min': 100, 'h_max': 120, 's_min': 80, 'v_min': 80},
        }

    # Direct right/left config — no green_is_right logic needed
    if 'right' in colors and 'left' in colors:
        for hand_label, hand_id in [('right', 0), ('left', 1)]:
            c = colors[hand_label]
            if c['h_min'] <= h <= c['h_max'] and s > c['s_min'] and v > c['v_min']:
                return hand_id
        return None

    g = colors['green']
    b = colors['blue']
    is_green = g['h_min'] <= h <= g['h_max'] and s > g['s_min'] and v > g['v_min']
    is_blue = b['h_min'] <= h <= b['h_max'] and s > b['s_min'] and v > b['v_min']

    if is_green:
        return 0 if green_is_right else 1
    if is_blue:
        return 1 if green_is_right else 0
    return None


def fallback_hand(pitch, recent_right, recent_left):
    """Assign a hand by pitch proximity to recently seen notes of each hand.

    Falls back to: right if pitch >= 60 (middle C), left otherwise, when
    neither hand has any recent notes.
    """
    if not recent_right or not recent_left:
        return 0 if pitch >= 60 else 1
    d_right = abs(pitch - np.mean(list(recent_right)))
    d_left = abs(pitch - np.mean(list(recent_left)))
    return 0 if d_right <= d_left else 1


# ---------------------------------------------------------------------------
# Region-based hue/saturation helpers
# ---------------------------------------------------------------------------

def _region_avg_saturation(hsv_frame, x_left, x_right, y_top, y_bot):
    """Return the average saturation in a detector region."""
    fh, fw = hsv_frame.shape[:2]
    x1, x2 = max(0, x_left), min(fw, x_right)
    y1, y2 = max(0, y_top), min(fh, y_bot)
    if y1 >= y2 or x1 >= x2:
        return 0.0
    return float(np.mean(hsv_frame[y1:y2, x1:x2, 1]))


def _region_avg_hue(hsv_frame, x_left, x_right, y_top, y_bot, s_min=50):
    """Return the average hue of saturated pixels in a detector region."""
    fh, fw = hsv_frame.shape[:2]
    x1, x2 = max(0, x_left), min(fw, x_right)
    y1, y2 = max(0, y_top), min(fh, y_bot)
    if y1 >= y2 or x1 >= x2:
        return None
    region = hsv_frame[y1:y2, x1:x2]
    h_ch = region[:, :, 0].flatten().astype(float)
    s_ch = region[:, :, 1].flatten().astype(float)
    mask = s_ch > s_min
    if np.sum(mask) < 3:
        return None
    return float(np.mean(h_ch[mask]))


def _classify_hand_from_hue(hue, green_is_right=True, colors=None):
    """Classify hand from average hue using config color ranges."""
    if hue is None:
        return None
    if colors is None:
        g_min, g_max = 40, 65
        b_min, b_max = 85, 125
    elif 'right' in colors and 'left' in colors:
        # Direct right/left config
        for hand_label, hand_id in [('right', 0), ('left', 1)]:
            c = colors[hand_label]
            if c['h_min'] <= hue <= c['h_max']:
                return hand_id
        return None
    else:
        g = colors['green']
        b = colors['blue']
        g_min, g_max = g['h_min'], g['h_max']
        b_min, b_max = b['h_min'], b['h_max']
    if g_min <= hue <= g_max:
        return 0 if green_is_right else 1
    if b_min <= hue <= b_max:
        return 1 if green_is_right else 0
    return None


# ---------------------------------------------------------------------------
# Auto-detection of hand colors from falling blocks
# ---------------------------------------------------------------------------

def auto_detect_colors(cap, y_min, y_max, n_frames=20):
    """Auto-detect the two hand colors from block pixels.

    Samples multiple frames, collects hues of saturated pixels in the
    waterfall region, and finds the two dominant hue clusters.

    Returns list of dicts: [{'name': str, 'h_center': int, 'h_min': int, 'h_max': int}]
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_hues = []

    for fi in range(0, total, max(1, total // n_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        region = hsv[y_min:y_max, :, :]
        sat_mask = (region[:, :, 1] > 80) & (region[:, :, 2] > 80)
        hues = region[:, :, 0][sat_mask]
        all_hues.extend(int(h) for h in hues)

    if not all_hues:
        # Fallback: standard green/blue
        return [
            {'name': 'color1', 'h_center': 55, 'h_min': 35, 'h_max': 75},
            {'name': 'color2', 'h_center': 110, 'h_min': 90, 'h_max': 130},
        ]

    # Build hue histogram (0-180)
    hist = np.zeros(180, dtype=int)
    for h in all_hues:
        hist[h % 180] += 1

    # Smooth and find peaks
    from scipy.ndimage import uniform_filter1d
    try:
        smooth = uniform_filter1d(hist.astype(float), size=10, mode='wrap')
    except ImportError:
        # No scipy — manual smoothing
        smooth = np.convolve(hist.astype(float), np.ones(10) / 10, mode='same')

    # Find peaks: local maxima above 10% of max
    threshold = smooth.max() * 0.1
    peaks = []
    for i in range(180):
        if smooth[i] > threshold:
            prev = smooth[(i - 1) % 180]
            next_ = smooth[(i + 1) % 180]
            if smooth[i] >= prev and smooth[i] >= next_:
                peaks.append((i, int(smooth[i])))

    # Merge nearby peaks (within 15 hue units)
    merged_peaks = []
    for h, c in sorted(peaks, key=lambda x: -x[1]):
        if not any(abs(h - mh) < 15 or abs(h - mh) > 165 for mh, _ in merged_peaks):
            merged_peaks.append((h, c))
        if len(merged_peaks) >= 2:
            break

    colors = []
    for h_center, _ in merged_peaks:
        colors.append({
            'name': f'h{h_center}',
            'h_center': h_center,
            'h_min': (h_center - 15) % 180,
            'h_max': (h_center + 15) % 180,
        })

    return colors


def make_color_mask(hsv_region, color):
    """Create a boolean mask for a detected color range."""
    h_min = color['h_min']
    h_max = color['h_max']
    s_min = 80
    v_min = 80

    if h_min > h_max:  # wraps around 0/180 (red)
        return (((hsv_region[:, :, 0] >= h_min) | (hsv_region[:, :, 0] <= h_max)) &
                (hsv_region[:, :, 1] > s_min) & (hsv_region[:, :, 2] > v_min))
    else:
        return ((hsv_region[:, :, 0] >= h_min) & (hsv_region[:, :, 0] <= h_max) &
                (hsv_region[:, :, 1] > s_min) & (hsv_region[:, :, 2] > v_min))
