"""PianoVideoScribe — extract hand-separated MIDI from Synthesia piano videos."""

from piano_video_scribe.constants import NOTE_NAMES, WHITE_SEMITONES, midi_to_name
from piano_video_scribe.keyboard import (
    detect_keyboard, build_note_x_map, build_detector_regions,
)
from piano_video_scribe.quantization import (
    quantize_onsets_viterbi, quantize_onsets_adaptive,
)
from piano_video_scribe.midi_output import build_track, make_monophonic, remove_overlaps
from piano_video_scribe.config import parse_args, load_config, detect_bpm_from_video
from piano_video_scribe.pipeline import main
