"""MIDI output helpers: overlap removal, monophonic reduction, track building."""

from collections import defaultdict

from mido import Message, MetaMessage, MidiTrack


def remove_overlaps(events):
    """Cut held notes when a new onset arrives, but keep chords intact.

    Notes starting at the same tick are treated as a chord and kept together.
    Notes held from a previous tick are cut short when any new note_on arrives.
    This produces clean notation without destroying chordal voicing.

    Args:
        events: list of (abs_tick, type, pitch, velocity) tuples.

    Returns:
        New event list with no cross-onset overlaps.
    """
    sorted_events = sorted(events, key=lambda e: (e[0], e[1] != 'note_off'))
    result = []
    active = {}  # pitch -> note_on tick

    for ev in sorted_events:
        abs_tick, ev_type, pitch, vel = ev
        if ev_type == 'note_on':
            # End all notes from *previous* ticks (not same-tick chord mates)
            to_end = [p for p, t in active.items() if t < abs_tick]
            for p in to_end:
                result.append((abs_tick, 'note_off', p, 0))
                del active[p]
            active[pitch] = abs_tick
            result.append(ev)
        elif ev_type == 'note_off':
            if pitch in active:
                result.append(ev)
                del active[pitch]

    return result


def make_monophonic(events, keep='highest'):
    """Force a list of note events to be monophonic (single voice).

    At each tick where multiple note_ons occur (chords), keep only one note
    (highest or lowest pitch). Then cut each note when the next one begins,
    so no two notes overlap in time.

    Args:
        events: list of (abs_tick, type, pitch, velocity) tuples.
        keep: 'highest' to keep the top note (melody), 'lowest' for bass.

    Returns:
        New list with exactly one note sounding at any time.
    """
    sorted_events = sorted(events, key=lambda e: (e[0], e[1] != 'note_off'))

    # Group note_ons by tick, pick one per tick
    tick_groups = defaultdict(list)
    for ev in sorted_events:
        if ev[1] == 'note_on':
            tick_groups[ev[0]].append(ev)

    kept_pitches = {}
    for tick, group in tick_groups.items():
        if len(group) == 1:
            kept_pitches[tick] = {group[0][2]}
        else:
            best = max(group, key=lambda e: e[2]) if keep == 'highest' else min(group, key=lambda e: e[2])
            kept_pitches[tick] = {best[2]}

    kept_notes = set()
    removed_notes = set()
    result = []
    active_pitch = None

    for ev in sorted_events:
        abs_tick, ev_type, pitch, vel = ev
        if ev_type == 'note_on':
            if pitch not in kept_pitches.get(abs_tick, set()):
                removed_notes.add(pitch)
                continue
            if active_pitch is not None:
                result.append((abs_tick, 'note_off', active_pitch, 0))
            active_pitch = pitch
            kept_notes.add(pitch)
            result.append(ev)
        elif ev_type == 'note_off':
            if pitch in removed_notes:
                removed_notes.discard(pitch)
                continue
            if pitch == active_pitch:
                result.append(ev)
                active_pitch = None

    return result


def build_track(events, name, tempo_us, channel=0):
    """Build a MidiTrack from a list of (abs_tick, type, pitch, velocity) events."""
    track = MidiTrack()
    track.name = name
    track.append(MetaMessage('set_tempo', tempo=tempo_us, time=0))
    events_sorted = sorted(events, key=lambda e: (e[0], e[1] != 'note_off'))
    prev = 0
    for abs_tick, ev_type, pitch, vel in events_sorted:
        delta = max(0, abs_tick - prev)
        prev = abs_tick
        if ev_type == 'note_on':
            track.append(Message('note_on', channel=channel, note=pitch, velocity=vel, time=delta))
        else:
            track.append(Message('note_off', channel=channel, note=pitch, velocity=0, time=delta))
    track.append(MetaMessage('end_of_track', time=0))
    return track
