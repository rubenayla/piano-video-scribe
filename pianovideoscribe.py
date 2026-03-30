#!/usr/bin/env python3
"""Backward-compatible entry point. See piano_video_scribe/ for implementation."""

# Re-export public API for backward compatibility
from piano_video_scribe.keyboard import detect_keyboard, build_note_x_map, build_detector_regions
from piano_video_scribe.quantization import quantize_onsets_viterbi, quantize_onsets_adaptive
from piano_video_scribe.midi_output import build_track, make_monophonic, remove_overlaps
from piano_video_scribe.visualization import generate_summary_image
from piano_video_scribe.config import parse_args, load_config, detect_bpm_from_video
from piano_video_scribe.detectors.keys import extract_notes_from_video
from piano_video_scribe.pipeline import main

if __name__ == '__main__':
    main()
