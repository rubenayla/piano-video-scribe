NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

WHITE_SEMITONES = [0, 2, 4, 5, 7, 9, 11]  # C D E F G A B


def midi_to_name(midi_num):
    octave = midi_num // 12 - 1
    name = NOTE_NAMES[midi_num % 12]
    return f"{name}{octave}"
