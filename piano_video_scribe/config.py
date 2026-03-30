"""CLI argument parsing, BPM detection, and JSON config loading."""

import argparse
import json
import os
import sys

import numpy as np


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
    p.add_argument('--bpm', type=int, default=None,
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
    p.add_argument('--dry-run', action='store_true',
                   help='Detect keyboard and print stats only — do not write output MIDI')

    args = p.parse_args()

    # --- Merge settings file (if provided) — fill None values from JSON ---
    if args.settings:
        with open(args.settings) as f:
            settings = json.load(f)
        SETTINGS_KEYS = ['bpm', 'key', 'green_hand', 'frame', 'right_hand',
                         'left_hand', 'start_time', 'end_time', 'start_beat', 'config',
                         'time_sig', 'transpose', 'detector']
        for key in SETTINGS_KEYS:
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


def load_config(config_path):
    """Load a JSON config file and return its contents as a dict.

    Config files can override: colors (HSV thresholds for green/blue),
    sampling zone (y offsets, half_w), and keyboard detection frame.
    Missing keys fall back to defaults.
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
    if config_path is None:
        return defaults

    with open(config_path) as f:
        cfg = json.load(f)

    # Deep merge: config values override defaults (2 levels deep)
    for section in defaults:
        if section in cfg:
            if isinstance(defaults[section], dict):
                for key in cfg[section]:
                    if isinstance(defaults[section].get(key), dict) and isinstance(cfg[section][key], dict):
                        defaults[section][key].update(cfg[section][key])
                    else:
                        defaults[section][key] = cfg[section][key]
            else:
                defaults[section] = cfg[section]
    return defaults
