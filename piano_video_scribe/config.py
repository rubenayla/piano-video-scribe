"""CLI argument parsing, BPM detection, and JSON config loading."""

import argparse
import json
import os
import sys

import numpy as np

# Keys that are allowed in JSON files but aren't pipeline parameters
_METADATA_KEYS = {'name', 'notes'}

# Sections that can appear in both settings.json and standalone config files
CONFIG_SECTIONS = {'colors', 'sampling', 'keyboard'}


def _derive_settings_keys(parser):
    """Derive valid settings keys from argparse optional arguments."""
    skip = {'help', 'settings', 'dry_run', 'triplet'}
    keys = set()
    for action in parser._actions:
        if action.option_strings and action.dest not in skip:
            keys.add(action.dest)
    return keys


def _warn_unknown_keys(data, valid_keys, source):
    """Warn about unknown keys in a JSON file (catches typos)."""
    unknown = set(data) - valid_keys - _METADATA_KEYS - CONFIG_SECTIONS
    if unknown:
        print(f"WARNING: Unknown keys in {source}: {sorted(unknown)}", file=sys.stderr)


def _validate_hsv(color_dict, label):
    """Validate HSV threshold bounds for a single color entry."""
    errors = []
    h_min = color_dict.get('h_min', 0)
    h_max = color_dict.get('h_max', 179)
    s_min = color_dict.get('s_min', 0)
    v_min = color_dict.get('v_min', 0)
    if not (0 <= h_min <= 179):
        errors.append(f"{label}.h_min={h_min} out of range [0, 179]")
    if not (0 <= h_max <= 179):
        errors.append(f"{label}.h_max={h_max} out of range [0, 179]")
    if h_min > h_max:
        errors.append(f"{label}.h_min={h_min} > h_max={h_max}")
    if not (0 <= s_min <= 255):
        errors.append(f"{label}.s_min={s_min} out of range [0, 255]")
    if not (0 <= v_min <= 255):
        errors.append(f"{label}.v_min={v_min} out of range [0, 255]")
    return errors


def _validate_config(cfg):
    """Validate config values (HSV ranges, sampling params). Returns error list."""
    errors = []
    for color_name, color_dict in cfg.get('colors', {}).items():
        if isinstance(color_dict, dict):
            errors.extend(_validate_hsv(color_dict, f"colors.{color_name}"))
    sampling = cfg.get('sampling', {})
    if sampling.get('y_offset_top', 0) < 0:
        errors.append(f"sampling.y_offset_top={sampling['y_offset_top']} must be >= 0")
    if sampling.get('y_offset_bot', 0) < 0:
        errors.append(f"sampling.y_offset_bot={sampling['y_offset_bot']} must be >= 0")
    if sampling.get('half_w', 1) <= 0:
        errors.append(f"sampling.half_w={sampling['half_w']} must be > 0")
    return errors


def parse_args():
    p = argparse.ArgumentParser(
        description='Separate a Synthesia video into right/left hand MIDI tracks.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run
  python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm 120

  # Dry run: detect keyboard only, print stats, exit
  python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm 120 --dry-run

  # If hands are swapped, flip green interpretation
  python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm 120 --green-hand left

  # Clean up left hand notation (recommended for most songs)
  python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm 120 --left-hand no-overlap

  # Single melody line in right hand, clean bass in left
  python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm 120 --right-hand monophonic --left-hand no-overlap
""")
    p.add_argument('video', help='Path to Synthesia MP4 video')
    p.add_argument('midi', nargs='?', default=None,
                   help='Path to input MIDI (optional — if omitted, extracts notes '
                        'directly from the video)')
    p.add_argument('output', help='Path to output MIDI')
    # All settings-overridable args default to None so --settings can fill them in.
    # Hardcoded defaults are applied after settings are merged.
    p.add_argument('--bpm', type=float, default=None,
                   help='BPM of the song. Auto-detected from video audio if omitted.')
    p.add_argument('--frame', type=int, default=None,
                   help='Frame index for keyboard detection (default: 5, should be note-free)')
    p.add_argument('--green-hand', choices=['right', 'left'], default=None,
                   help='Which hand is shown in green in the video (default: right)')
    hand_choices = ['normal', 'no-overlap', 'monophonic']
    p.add_argument('--right-hand', choices=hand_choices, default=None,
                   help='Right hand processing: normal, '
                        'no-overlap (default, cut held notes at next onset, keep chords), '
                        'monophonic (single voice, highest note only)')
    p.add_argument('--left-hand', choices=hand_choices, default=None,
                   help='Left hand processing: normal, '
                        'no-overlap (default, cut held notes at next onset, keep chords), '
                        'monophonic (single voice, lowest note only)')
    p.add_argument('--key', type=str, default=None,
                   help='Key signature for MIDI metadata (e.g. "E", "Em"). '
                        'Auto-detected if omitted.')
    p.add_argument('--config', type=str, default=None,
                   help='Path to a JSON config file (colors, sampling zone, keyboard frame). '
                        'See configs/ directory for examples.')
    p.add_argument('--start-time', type=float, default=None,
                   help='Manual override: music start time in seconds (skips auto-detection)')
    p.add_argument('--end-time', type=float, default=None,
                   help='Manual override: music end time in seconds (skips auto-detection)')
    p.add_argument('--start-beat', type=float, default=None,
                   help='Beat position of the first note (default: auto-detect). '
                        'Use 1.5 for pickup on the "and" of beat 1, '
                        '4.5 for pickup on the "and" of beat 4, etc.')
    p.add_argument('--settings', type=str, default=None,
                   help='Path to a settings.json file with saved pipeline parameters. '
                        'CLI flags override values from the file.')
    p.add_argument('--time-sig', type=str, default=None, dest='time_sig',
                   help='Time signature (e.g. "6/8", "3/4"). Default: 4/4')
    p.add_argument('--transpose', type=int, default=0,
                   help='Transpose all notes by N semitones (e.g. -12 for one octave down)')
    p.add_argument('--triplet', action='store_true',
                   help='Use combined grid (12 per beat) that supports both straight 16ths '
                        'and triplet positions. Each note snaps to the nearest valid position.')
    p.add_argument('--detector', choices=['keys', 'falling-blocks'], default=None,
                   help='Note detection method: falling-blocks (default, detects notes from '
                        'falling colored blocks) or keys (key-lighting saturation fallback)')
    p.add_argument('--quantizer', choices=['simple', 'viterbi', 'adaptive', 'pll'], default=None,
                   help='Quantization method: viterbi (default, DP optimal grid '
                        'assignment at the drift-corrected effective BPM — keeps '
                        'identical repeating patterns identical), simple '
                        '(round-to-nearest with drift correction, snaps each '
                        'onset independently), adaptive (2-pass viterbi with '
                        'local BPM warping), or pll (online phase + BPM tracking '
                        'via EMA — use when video has significant tempo changes)')
    p.add_argument('--glow-margin', type=int, default=None,
                   help='Falling-blocks detector: vertical margin above keyboard for glow '
                        'filtering (pixels, default: 50)')
    p.add_argument('--s-min', type=int, default=None,
                   help='Falling-blocks detector: global saturation minimum override')
    p.add_argument('--s-min-per-color', type=json.loads, default=None,
                   help='Falling-blocks detector: per-color saturation minimums as JSON '
                        '(e.g. \'{"green": 120}\')')
    p.add_argument('--sample-step', type=int, default=None,
                   help='Falling-blocks detector: frame sampling interval (default: 2)')
    p.add_argument('--dry-run', action='store_true',
                   help='Detect keyboard and print stats only — do not write output MIDI')

    args = p.parse_args()

    # Derive valid keys from the parser itself — no separate list to maintain
    valid_settings_keys = _derive_settings_keys(p)

    # --- Merge settings file (if provided) — fill None values from JSON ---
    args._inline_config = {}
    if args.settings:
        with open(args.settings) as f:
            settings = json.load(f)

        # Extract config sections (colors, sampling, keyboard) for load_config
        for section in CONFIG_SECTIONS:
            if section in settings:
                args._inline_config[section] = settings[section]

        _warn_unknown_keys(settings, valid_settings_keys, args.settings)

        for key in valid_settings_keys:
            if key in settings:
                # Only override if the CLI didn't explicitly set it
                cli_val = getattr(args, key, None)
                if cli_val is None:
                    setattr(args, key, settings[key])

    # --- Apply hardcoded defaults for anything still None ---
    HARDCODED_DEFAULTS = {
        'frame': 5,
        'green_hand': 'right',
        'right_hand': 'no-overlap',
        'left_hand': 'no-overlap',
        'detector': 'falling-blocks',
    }
    for key, default in HARDCODED_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, default)

    # BPM: auto-detect from video audio if not provided
    args._had_explicit_bpm = args.bpm is not None
    if args.bpm is None:
        detected = detect_bpm_from_video(args.video)
        if detected is None:
            p.error('--bpm is required (auto-detection failed)')
        args.bpm = detected

    return args


def detect_bpm_from_video(video_path):
    """Extract audio from video and detect BPM using librosa.

    Returns the detected BPM as an integer, or None on failure.
    """
    import subprocess
    import tempfile

    # Extract audio to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
             '-ar', '44100', '-ac', '1', tmp_path, '-y'],
            capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print("WARNING: Could not extract audio from video for BPM detection.",
                  file=sys.stderr)
            return None

        import librosa
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)

        # Use beat_track for primary estimate
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        # tempo may be an array in some librosa versions
        primary_bpm = float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0])

        # Also get tempogram candidates for context
        oenv = librosa.onset.onset_strength(y=audio, sr=sr)
        tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
        ac = np.mean(tg, axis=1)
        bpms = librosa.tempo_frequencies(tg.shape[0], sr=sr)
        mask = (bpms >= 40) & (bpms <= 200)
        top_idx = np.argsort(ac[mask])[::-1][:3]
        candidates = [(bpms[mask][i], ac[mask][i]) for i in top_idx]

        print(f"Auto-detected BPM: {primary_bpm:.0f}")
        print(f"  Candidates: {', '.join(f'{b:.0f}' for b, _ in candidates)}")

        # Librosa often picks double-time; prefer half if it's in 80-140 range
        bpm = round(primary_bpm)
        if bpm > 140:
            half = bpm / 2
            if 60 <= half <= 140:
                print(f"  Halving {bpm} -> {half:.0f} (likely double-time)")
                bpm = round(half)

        print(f"  Using: {bpm} BPM")
        return bpm

    except Exception as e:
        print(f"WARNING: BPM auto-detection failed: {e}", file=sys.stderr)
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _deep_merge(base, override):
    """Deep-merge override into base (2 levels). Mutates base."""
    for section in override:
        if section not in base:
            base[section] = override[section]
        elif isinstance(base[section], dict) and isinstance(override[section], dict):
            for key in override[section]:
                if (isinstance(base[section].get(key), dict)
                        and isinstance(override[section][key], dict)):
                    base[section][key].update(override[section][key])
                else:
                    base[section][key] = override[section][key]
        else:
            base[section] = override[section]


# Valid top-level keys in a config/settings file's config sections
_VALID_CONFIG_SECTION_KEYS = {
    'colors': None,   # colors has dynamic keys (green/blue/right/left)
    'sampling': {'y_offset_top', 'y_offset_bot', 'half_w'},
    'keyboard': {'frame'},
}


def _warn_unknown_config_keys(cfg, source):
    """Warn about unknown keys within config sections."""
    unknown_sections = set(cfg) - CONFIG_SECTIONS - _METADATA_KEYS
    if unknown_sections:
        print(f"WARNING: Unknown config sections in {source}: "
              f"{sorted(unknown_sections)}", file=sys.stderr)
    for section, valid_keys in _VALID_CONFIG_SECTION_KEYS.items():
        if section in cfg and valid_keys is not None and isinstance(cfg[section], dict):
            bad = set(cfg[section]) - valid_keys
            if bad:
                print(f"WARNING: Unknown keys in {source} [{section}]: "
                      f"{sorted(bad)}", file=sys.stderr)


def load_config(config_path=None, inline_config=None):
    """Load config with defaults, optional file, and optional inline overrides.

    Precedence (lowest to highest): defaults < config_path < inline_config.

    Config sections: colors (HSV thresholds), sampling (y offsets, half_w),
    keyboard (detection frame).
    """
    defaults = {
        'colors': {
            'green': {'h_min': 40, 'h_max': 65, 's_min': 100, 'v_min': 80},
            'blue':  {'h_min': 100, 'h_max': 120, 's_min': 80, 'v_min': 80},
        },
        'sampling': {
            'y_offset_top': 90,
            'y_offset_bot': 0,
            'half_w': 10,
        },
        'keyboard': {
            'frame': 5,
        },
    }

    if config_path is not None:
        with open(config_path) as f:
            file_cfg = json.load(f)
        _warn_unknown_config_keys(file_cfg, config_path)
        _deep_merge(defaults, {k: v for k, v in file_cfg.items()
                               if k in CONFIG_SECTIONS})

    if inline_config:
        _deep_merge(defaults, {k: v for k, v in inline_config.items()
                               if k in CONFIG_SECTIONS})

    errors = _validate_config(defaults)
    if errors:
        print("ERROR: Invalid config values:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    return defaults
