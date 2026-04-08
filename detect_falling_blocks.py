#!/usr/bin/env python3
"""Backward-compatible entry point for falling-blocks detector.

All implementation is in piano_video_scribe.detectors.falling_blocks.
"""
import argparse

from piano_video_scribe.detectors.falling_blocks import (
    detect_falling_notes_pipeline,
    notes_to_midi,
    detect_blocks_in_frame,
    collect_and_cluster_blocks,
    fit_white_key_grid,
    identify_c_position,
    build_pitch_map,
    find_keyboard_y,
    measure_fall_speed,
    detect_bpm,
)


def main():
    parser = argparse.ArgumentParser(
        description='Detect notes from falling-block piano tutorial videos')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('output', help='Path for output MIDI file')
    parser.add_argument('--base-octave', type=int, default=4,
                        help='Octave of the first C (default: 4)')
    args = parser.parse_args()
    detect_falling_notes_pipeline(args.video, args.output, args.base_octave)


if __name__ == '__main__':
    main()
