"""Lead sheet generation: chord detection + single-staff MusicXML output.

Chord detection pipeline (ISMIR standard):
  1. Duration-weighted, hand-weighted chroma vectors
  2. Onset-boundary segmentation (not fixed windows)
  3. Cosine similarity against 84-template vocabulary (12 roots × 7 qualities)
  4. Viterbi / HMM smoothing to suppress spurious single-onset changes
"""

import os
from xml.sax.saxutils import escape as xml_escape

import numpy as np

from .midi_output import make_monophonic

# ─── chord template vocabulary ───────────────────────────────────────────────

_ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_ROOT_PC = {r: i for i, r in enumerate(_ROOTS)}

_QUALITY_INTERVALS = {
    'maj':  [0, 4, 7],
    'min':  [0, 3, 7],
    '7':    [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10],
    'dim':  [0, 3, 6],
    'aug':  [0, 4, 8],
}

def _build_templates():
    labels, vecs = [], []
    for root in _ROOTS:
        for qual, intervals in _QUALITY_INTERVALS.items():
            v = np.zeros(12)
            for i in intervals:
                v[(_ROOT_PC[root] + i) % 12] = 1.0
            v /= np.linalg.norm(v)
            labels.append(f'{root}:{qual}')
            vecs.append(v)
    return labels, np.array(vecs)  # (84, 12)

_CHORD_LABELS, _TEMPLATE_MATRIX = _build_templates()
_N_CHORDS = len(_CHORD_LABELS)

# ─── label → MusicXML display fields ─────────────────────────────────────────

_QUALITY_DISPLAY   = {'maj':'','min':'m','7':'7','maj7':'maj7','min7':'m7','dim':'dim','aug':'aug'}
_QUALITY_KIND_TEXT = {'maj':'','min':'m','7':'7','maj7':'maj7','min7':'m7','dim':'dim','aug':'aug'}
_QUALITY_KIND_VAL  = {
    'maj':'major','min':'minor','7':'dominant','maj7':'major-seventh',
    'min7':'minor-seventh','dim':'diminished','aug':'augmented',
}

def _label_to_display(label):
    """'A:min' → (display='Am', kind_text='m', kind_val='minor', step='A', alter=0)"""
    root, qual = label.split(':')
    display   = root + _QUALITY_DISPLAY.get(qual, qual)
    kind_text = _QUALITY_KIND_TEXT.get(qual, qual)
    kind_val  = _QUALITY_KIND_VAL.get(qual, 'major')
    step  = root[0]
    alter = 1 if len(root) > 1 and root[1] == '#' else (-1 if len(root) > 1 and root[1] == 'b' else 0)
    return display, kind_text, kind_val, step, alter

# ─── pitch → MusicXML step/alter/octave ──────────────────────────────────────

_PC_STEP = {
    0:('C',0),1:('C',1),2:('D',0),3:('D',1),4:('E',0),
    5:('F',0),6:('F',1),7:('G',0),8:('G',1),9:('A',0),10:('A',1),11:('B',0),
}

# ─── note pair extraction ─────────────────────────────────────────────────────

def _events_to_pairs(right_events, left_events):
    """Convert event lists → [(start_tick, end_tick, pitch, hand)] sorted by start."""
    pairs = []
    for events, hand in [(right_events, 'right'), (left_events, 'left')]:
        pending = {}
        for tick, ev_type, pitch, vel in sorted(events, key=lambda e: e[0]):
            if ev_type == 'note_on' and vel > 0:
                pending[pitch] = tick
            elif ev_type == 'note_off' or (ev_type == 'note_on' and vel == 0):
                if pitch in pending:
                    pairs.append((pending.pop(pitch), tick, pitch, hand))
        for pitch, start in pending.items():
            pairs.append((start, float('inf'), pitch, hand))
    pairs.sort()
    return pairs

# ─── duration-weighted chroma ─────────────────────────────────────────────────

def _chroma(pairs, start_tick, end_tick, lh_weight, rh_weight):
    """12-dim duration-weighted, hand-weighted chroma vector (unit-norm)."""
    v = np.zeros(12)
    for s, e, pitch, hand in pairs:
        e_eff = end_tick if e == float('inf') else e
        overlap = max(0.0, min(e_eff, end_tick) - max(s, start_tick))
        if overlap <= 0:
            continue
        v[pitch % 12] += overlap * (lh_weight if hand == 'left' else rh_weight)
    norm = v.sum()
    return v / norm if norm > 0 else v

# ─── per-segment chord scores ─────────────────────────────────────────────────

# Complexity prior: richer qualities (7ths, altered triads) must clear a small
# bar over plain triads, so a single melody passing-tone can't promote a triad to
# a 7th chord. Genuine 7ths (the 7th actually sounding) still clear it easily.
_DEFAULT_QUALITY_BIAS = {
    'maj': 0.0, 'min': 0.0,
    '7': -0.06, 'maj7': -0.06, 'min7': -0.06, 'dim': -0.06, 'aug': -0.06,
}

def _quality_bias_vec(quality_bias):
    qb = {**_DEFAULT_QUALITY_BIAS, **(quality_bias or {})}
    return np.array([qb.get(lbl.split(':')[1], 0.0) for lbl in _CHORD_LABELS])


def _score_segment(chroma_vec, bass_pc, bass_bonus, bias_vec=None):
    """
    Cosine similarity of chroma_vec against all 84 templates + bass bonus
    + per-quality complexity prior.
    Returns (84,) score array.
    """
    nrm = np.linalg.norm(chroma_vec)
    if nrm < 1e-8:
        return np.zeros(_N_CHORDS)
    sims = _TEMPLATE_MATRIX @ chroma_vec / nrm  # (84,) cosine similarities
    for i, label in enumerate(_CHORD_LABELS):
        if _ROOT_PC[label.split(':')[0]] == bass_pc:
            sims[i] += bass_bonus
    if bias_vec is not None:
        sims = sims + bias_vec
    return sims

# ─── onset-boundary segmentation ─────────────────────────────────────────────

def _segment(pairs, lh_weight, rh_weight, bass_bonus):
    """
    Collect all onset ticks from both hands, compute chroma + chord scores
    for every inter-onset interval.
    Returns list of (start_tick, end_tick, score_vector_84).
    """
    onsets = sorted({s for s, e, p, h in pairs if s != float('inf')})
    if not onsets:
        return []
    ends_finite = [e for _, e, _, _ in pairs if e != float('inf')]
    song_end = max(ends_finite) if ends_finite else onsets[-1] + 1
    boundaries = sorted(set(onsets) | {song_end})

    segments = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e <= s:
            continue
        chroma_vec = _chroma(pairs, s, e, lh_weight, rh_weight)
        active = [p for st, en, p, h in pairs if st <= s < (en if en != float('inf') else e + 1)]
        bass_pc = min(active) % 12 if active else 0
        scores  = _score_segment(chroma_vec, bass_pc, bass_bonus)
        segments.append((s, e, scores))
    return segments

# ─── metrical (measure-synchronous) chord estimation ──────────────────────────

def _metrical_chroma(pairs, s, e, tpb, lh_weight, rh_weight, onset_weight, sustain_cap):
    """
    Duration + onset weighted chroma for the metrical window [s, e).

    Each note contributes, scaled by hand weight:
      - its sounding duration inside the window (in beats), capped at `sustain_cap`
        beats so a long pedal note can't dominate;
      - plus `onset_weight` beat-equivalents if it ATTACKS inside the window.
    The onset term is what lets an articulated chord change beat a held-over bass.
    """
    cap = sustain_cap * tpb if sustain_cap else float('inf')
    v = np.zeros(12)
    for st, en, pitch, hand in pairs:
        en_eff = e if en == float('inf') else en
        overlap = max(0.0, min(en_eff, e) - max(st, s))
        attacks = s <= st < e
        if overlap <= 0 and not attacks:
            continue
        hw = lh_weight if hand == 'left' else rh_weight
        dur_beats = min(overlap, cap) / tpb
        onset_beats = onset_weight if attacks else 0.0
        v[pitch % 12] += hw * (dur_beats + onset_beats)
    norm = v.sum()
    return v / norm if norm > 0 else v


def _articulated_bass_pc(pairs, s, e):
    """Pitch class of the lowest note that ATTACKS in [s, e); None if the window
    only contains held-over notes (a pedal shouldn't claim the bass-bonus)."""
    attacked = [p for st, en, p, h in pairs if s <= st < e]
    return (min(attacked) % 12) if attacked else None


def _strike_tick(pairs, s, e):
    """Absolute tick to anchor the chord symbol inside window [s, e).

    Chords in real arrangements land on syncopated beats, not the barline — the
    left hand (accompaniment) attack marks the change. We then snap to the melody
    (right-hand) note struck nearest that attack, so the symbol sits exactly on
    the note it sounds with ("the C goes with that A") rather than drifting to the
    downbeat or floating between notes. Falls back to the left-hand attack, then
    any onset, then the bar start."""
    left = [st for st, en, p, h in pairs if h == 'left' and s <= st < e]
    rh = [st for st, en, p, h in pairs if h == 'right' and s <= st < e]
    if left:
        strike = min(left)
    elif rh:
        strike = min(rh)
    else:
        anyon = [st for st, en, p, h in pairs if s <= st < e and st != float('inf')]
        return min(anyon) if anyon else s
    if rh:
        return min(rh, key=lambda t: (abs(t - strike), t))  # nearest melody note
    return strike


def _detect_metrical(pairs, tpb, tpm, cfg):
    """One chord per metrical window: chroma → template argmax → smooth → merge.

    No Viterbi/onset boundaries — windows are fixed to the bar grid, so chords
    land exactly on downbeats (no lag) and can't over-segment within a bar.
    """
    seg = max(1, tpm // max(1, cfg['chords_per_measure']))
    onsets = [s for s, e, p, h in pairs if s != float('inf')]
    if not onsets:
        return []
    ends = [e for s, e, p, h in pairs if e != float('inf')]
    song_end = max(ends) if ends else max(onsets) + seg
    n = int((song_end + seg - 1) // seg)
    bias_vec = _quality_bias_vec(cfg.get('quality_bias'))

    # 1) per-bar chord score vector (None marks a silent bar)
    score_list = []
    silent = []
    for k in range(n):
        s, e = k * seg, (k + 1) * seg
        chroma = _metrical_chroma(pairs, s, e, tpb, cfg['lh_weight'], cfg['rh_weight'],
                                  cfg['onset_weight'], cfg['sustain_cap_beats'])
        if chroma.sum() <= 0:
            score_list.append(np.zeros(_N_CHORDS))
            silent.append(True)
            continue
        bass_pc = _articulated_bass_pc(pairs, s, e)
        score_list.append(_score_segment(chroma, bass_pc if bass_pc is not None else -1,
                                          cfg['bass_bonus'], bias_vec))
        silent.append(False)

    # 2) bar-level HMM smoothing (Viterbi). Resists spurious single-bar harmony
    #    flips from melodic passing-tones — a static Em pedal under a moving riff
    #    stays Em instead of flickering Em/E7 — while a real change (bar's chord
    #    clearly wins) still flips. One label per bar, anchored on the downbeat.
    idx = _viterbi(score_list, cfg['bar_change_penalty'])
    labels = [_CHORD_LABELS[i] for i in idx]

    # 3) silent bars carry the previous chord (held harmony, not a new one)
    last = None
    for i in range(n):
        if silent[i]:
            labels[i] = last
        else:
            last = labels[i]

    # 4) merge consecutive identical windows into runs; place each NEW chord at
    #    the tick where it is actually struck (not the barline), so the symbol
    #    sits on the melody note it sounds with.
    runs = []
    for k, label in enumerate(labels):
        if label is None:
            continue
        s, e = k * seg, (k + 1) * seg
        if runs and runs[-1][2] == label:
            runs[-1] = (runs[-1][0], e, label)
        else:
            runs.append((_strike_tick(pairs, s, e), e, label))
    return runs

# ─── Viterbi smoothing ────────────────────────────────────────────────────────

def _viterbi(segment_scores, chord_change_penalty):
    """
    HMM Viterbi over chord sequence.

    Emissions are log-softmax of raw cosine scores (temperature=10), so the
    best chord gets ~0 and others get large negatives. Transitions are:
      - self-loop:  0 (no penalty)
      - chord change: -chord_change_penalty (tunable; higher = fewer changes)

    segment_scores: list of (84,) cosine score arrays.
    Returns list of chord indices (one per segment).
    """
    T = len(segment_scores)
    C = _N_CHORDS
    TEMP = 10.0  # softmax temperature — sharpens the distribution

    # Log-softmax normalises raw scores into proper log-probs
    raw = np.array(segment_scores) * TEMP  # (T, C)
    log_emit = raw - np.log(np.exp(raw - raw.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True) + 1e-30) \
                   - raw.max(axis=1, keepdims=True) + raw.max(axis=1, keepdims=True)
    # Simpler: use scipy if available, else manual
    try:
        from scipy.special import log_softmax as _ls
        log_emit = np.array([_ls(r) for r in raw])
    except ImportError:
        shifted = raw - raw.max(axis=1, keepdims=True)
        log_emit = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True) + 1e-30)

    V = log_emit[0].copy()  # (C,)
    back = np.zeros((T, C), dtype=np.int32)

    penalty = -abs(chord_change_penalty)

    for t in range(1, T):
        # best previous for a chord-change transition
        best_other_prev = int(np.argmax(V))
        best_other_val  = V[best_other_prev] + penalty

        new_V = np.empty(C)
        for c in range(C):
            self_val = V[c]           # self-loop: no penalty
            if self_val >= best_other_val:
                back[t, c] = c
                new_V[c] = self_val + log_emit[t, c]
            else:
                back[t, c] = best_other_prev
                new_V[c] = best_other_val + log_emit[t, c]
        V = new_V

    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(V))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path.tolist()

# ─── public: detect chords ────────────────────────────────────────────────────

_DEFAULT_CONFIG = {
    'segmentation':         'metrical',  # 'metrical' (one chord per bar) or 'onset' (legacy)
    'chords_per_measure':   1,    # metrical granularity (1 = one chord per bar, 2 = per half-bar)
    'lh_weight':            1.0,  # weight for left-hand notes in chroma
    'rh_weight':            0.3,  # weight for right-hand notes (melody pollutes less)
    'onset_weight':         1.5,  # extra weight per note ATTACK in the segment (beat-equivalents).
                                  # De-emphasises held-over pedal tones so the articulated
                                  # harmony wins — a sustained bass note shouldn't outvote a
                                  # freshly-struck chord change above it.
    'sustain_cap_beats':    1.0,  # cap a note's duration contribution at N beats (pedal damping)
    'bass_bonus':           0.1,  # cosine score bonus when template root == articulated bass
    'quality_bias':         None, # per-quality score prior (None → _DEFAULT_QUALITY_BIAS:
                                  # 7ths/dim/aug get -0.06 so they only win when really present)
    'bar_change_penalty':   1.5,  # bar-level Viterbi penalty per chord change (higher = stickier,
                                  # resists single-bar flips from melodic passing-tones)
    # ── legacy onset-segmentation knobs (only used when segmentation == 'onset') ──
    'chord_change_penalty': 1.5,  # Viterbi penalty per chord change (higher = fewer changes)
    'chord_snap_beats':     2,    # snap chord boundaries to nearest N-beat grid (2 = half-measure)
}

def detect_chords(right_events, left_events, tpb, config=None, tpm=None):
    """
    Full pipeline: events → [(start_tick, end_tick, mir_eval_label)]

    label format follows mir_eval: 'Root:quality'  (e.g. 'A:min', 'G:7', 'C:maj')

    tpm: ticks per measure (for metrical segmentation). Defaults to 4 beats (4/4).
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    pairs = _events_to_pairs(right_events, left_events)

    # Measure-synchronous estimation: one chord per bar, anchored on the downbeat.
    if cfg.get('segmentation', 'metrical') == 'metrical':
        return _detect_metrical(pairs, tpb, tpm or (4 * tpb), cfg)

    # ── legacy onset-boundary + Viterbi path ──
    segments = _segment(pairs, cfg['lh_weight'], cfg['rh_weight'], cfg['bass_bonus'])
    if not segments:
        return []

    scores   = [s[2] for s in segments]
    indices  = _viterbi(scores, cfg['chord_change_penalty'])

    # Build raw sequence and merge consecutive same-chord runs
    result = []
    for (start, end, _), idx in zip(segments, indices):
        label = _CHORD_LABELS[idx]
        if result and result[-1][2] == label:
            result[-1] = (result[-1][0], end, label)
        else:
            result.append([start, end, label])

    # Snap chord boundaries to the nearest half-measure so they align with
    # measure/half-measure marks in the score. Without this, chord-change
    # onsets (which fall on actual note attacks) land mid-measure and the
    # bar lines visually disagree with where the harmony actually shifts.
    snap_unit = tpb * cfg['chord_snap_beats']
    if snap_unit > 0:
        snapped = []
        for s, e, label in result:
            s_snap = round(s / snap_unit) * snap_unit
            if snapped and snapped[-1][2] == label:
                snapped[-1] = (snapped[-1][0], e, label)
            elif snapped and s_snap <= snapped[-1][0]:
                # Snapped onto a previous chord — overwrite (latest wins)
                snapped[-1] = (snapped[-1][0], e, label)
            else:
                snapped.append((s_snap, e, label))
        result = snapped

    return [(s, e, l) for s, e, l in result]

# ─── chord list for MusicXML ──────────────────────────────────────────────────

def group_chords(right_events, left_events, tpb, config=None, tpm=None):
    """
    Wrapper around detect_chords() returning format expected by build_musicxml:
    [(abs_tick, display_name, kind_text, kind_val, step, alter)]
    """
    detected = detect_chords(right_events, left_events, tpb, config, tpm=tpm)
    result = []
    for start_tick, _, label in detected:
        display, kt, kv, step, alter = _label_to_display(label)
        result.append((start_tick, display, kt, kv, step, alter))
    return result

# ─── MusicXML generation ──────────────────────────────────────────────────────

def _clean_values(tpb):
    """Representable note durations (ticks, type, dotted), largest first."""
    return [
        (tpb * 4,       'whole',   False),
        (tpb * 3,       'half',    True),
        (tpb * 2,       'half',    False),
        (tpb * 3 // 2,  'quarter', True),
        (tpb,           'quarter', False),
        (tpb * 3 // 4,  'eighth',  True),
        (tpb // 2,      'eighth',  False),
        (tpb * 3 // 8,  '16th',    True),
        (tpb // 4,      '16th',    False),
        (tpb * 3 // 16, '32nd',    True),
        (tpb // 8,      '32nd',    False),
    ]


def _nearest_type(ticks, tpb):
    return min(_clean_values(tpb), key=lambda c: abs(c[0] - ticks))


def _decompose_duration(duration, tpb):
    """Split an arbitrary tick duration into a list of (ticks, type, dotted)
    components, each an exactly representable note value, summing to `duration`.

    A single note whose `<duration>` doesn't equal the value implied by its
    `<type>` (e.g. a 2.5-beat span written as one dotted half = 3 beats) is what
    MuseScore 4 flags as a corrupt file. A 2.5-beat span must be a half tied to
    an eighth. Greedy largest-first gives a valid (if not always beat-optimal)
    decomposition; the caller ties pitched components together.
    """
    vals = _clean_values(tpb)
    smallest = vals[-1][0]
    out = []
    remaining = duration
    while remaining >= smallest:
        for ticks, ntype, dotted in vals:
            if ticks <= remaining:
                out.append((ticks, ntype, dotted))
                remaining -= ticks
                break
        else:
            break
    if not out:  # sub-32nd remainder — emit the smallest clean value
        out.append(vals[-1])
    return out


def _pitch_xml(midi_note):
    step, alter = _PC_STEP[midi_note % 12]
    octave = (midi_note // 12) - 1
    alter_tag = f'<alter>{alter}</alter>' if alter else ''
    return f'<pitch><step>{step}</step>{alter_tag}<octave>{octave}</octave></pitch>'


def _note_xml(duration, tpb, midi_note=None, tie_start=False, tie_stop=False):
    """One or more MusicXML <note> elements for `duration` ticks.

    Durations that aren't a single representable note value are split into tied
    components (pitched) or consecutive notes (rests), so every emitted note has
    <type> consistent with <duration> — MuseScore 4 rejects mismatches as corrupt.
    Element order matters: pitch → duration → tie → type → dot → notations.
    """
    parts = _decompose_duration(duration, tpb)

    if midi_note is None:
        # Rests can't be tied — just emit consecutive rests.
        out = ''
        for ticks, ntype, dotted in parts:
            dot_tag = '<dot/>' if dotted else ''
            out += f'<note><rest/><duration>{ticks}</duration><type>{ntype}</type>{dot_tag}</note>\n'
        return out

    n = len(parts)
    out = ''
    for i, (ticks, ntype, dotted) in enumerate(parts):
        dot_tag = '<dot/>' if dotted else ''
        # Chain ties across components; carry the outer tie flags on the ends.
        this_stop = tie_stop if i == 0 else True
        this_start = tie_start if i == n - 1 else True
        tie_tags = ''
        notation_inner = ''
        if this_stop:
            tie_tags += '<tie type="stop"/>'
            notation_inner += '<tied type="stop"/>'
        if this_start:
            tie_tags += '<tie type="start"/>'
            notation_inner += '<tied type="start"/>'
        notations = f'<notations>{notation_inner}</notations>' if notation_inner else ''
        out += (
            f'<note>{_pitch_xml(midi_note)}<duration>{ticks}</duration>'
            f'{tie_tags}<type>{ntype}</type>{dot_tag}{notations}</note>\n'
        )
    return out


def _harmony_xml(kind_text, kind_val, step, alter):
    alter_tag = f'<root-alter>{alter}</root-alter>' if alter else ''
    return (
        f'<harmony><root><root-step>{step}</root-step>{alter_tag}</root>'
        f'<kind text="{xml_escape(kind_text)}">{kind_val}</kind></harmony>\n'
    )


def build_musicxml(right_events, left_events, tempo_us, tpb, time_sig=(4, 4), config=None):
    """
    Build a MusicXML lead sheet string:
    - Melody (monophonic right hand) on one treble staff
    - Chord symbols above from full detection pipeline
    """
    beats_per_measure, beat_type = time_sig
    tpm = beats_per_measure * tpb  # ticks per measure

    chord_list = group_chords(right_events, left_events, tpb, config, tpm=tpm)

    # Monophonic melody from right hand
    mono = make_monophonic(right_events)
    active = {}
    segments = []
    for abs_tick, ev_type, pitch, vel in sorted(mono, key=lambda e: e[0]):
        if ev_type == 'note_on':
            active[pitch] = abs_tick
        elif ev_type == 'note_off' and pitch in active:
            segments.append((active.pop(pitch), abs_tick, pitch))
    segments.sort()

    if not segments:
        return None

    # Re-strike the melody at chord-change downbeats. MuseScore won't anchor a
    # chord symbol to a note that is *tied over* the barline (the symbol drifts to
    # the next onset, ~a 16th late). Beat It's riff is syncopated, so melody notes
    # routinely tie across the bar — exactly where chords change. Splitting the
    # note at the chord tick (no tie) gives that downbeat a fresh onset for the
    # chord to sit on. Only notes that actually span a change are touched.
    chord_ticks = sorted({t for t, *_ in chord_list})
    if chord_ticks:
        resplit = []
        for start, end, pitch in segments:
            cuts = [t for t in chord_ticks if start < t < end]
            prev = start
            for t in cuts:
                resplit.append((prev, t, pitch))
                prev = t
            resplit.append((prev, end, pitch))
        segments = resplit

    song_end = max(e for _, e, _ in segments)
    n_measures = max(1, (song_end + tpm - 1) // tpm)

    # Distribute melody into measures, tagging cross-measure notes for ties
    measures = [[] for _ in range(n_measures)]
    for start, end, pitch in segments:
        m_start = start // tpm
        m_end_idx = (end - 1) // tpm
        for m in range(m_start, min(m_end_idx + 1, n_measures)):
            s = max(start, m * tpm) - m * tpm
            e = min(end, (m + 1) * tpm) - m * tpm
            if e > s:
                tie_stop  = (m > m_start)        # continues from previous measure
                tie_start = (m < m_end_idx)      # continues into next measure
                measures[m].append((s, e, pitch, tie_start, tie_stop))

    def fill(segs):
        """Insert rests in gaps. Rest entries have tie flags False."""
        out, cursor = [], 0
        for s, e, p, ts, tp in sorted(segs):
            if s > cursor:
                out.append((cursor, s, None, False, False))
            out.append((s, e, p, ts, tp))
            cursor = e
        if cursor < tpm:
            out.append((cursor, tpm, None, False, False))
        return out

    bpm = int(60_000_000 / tempo_us)
    pending = list(chord_list)
    pending_idx = 0
    measures_xml = []

    for m_idx in range(n_measures):
        base = m_idx * tpm
        m_xml = ''
        if m_idx == 0:
            m_xml += (
                f'<attributes>'
                f'<divisions>{tpb}</divisions>'
                f'<key><fifths>0</fifths></key>'
                f'<time><beats>{beats_per_measure}</beats><beat-type>{beat_type}</beat-type></time>'
                f'<clef><sign>G</sign><line>2</line></clef>'
                f'</attributes>'
                f'<direction placement="above"><direction-type>'
                f'<metronome><beat-unit>quarter</beat-unit><per-minute>{bpm}</per-minute></metronome>'
                f'</direction-type></direction>\n'
            )
        for start_in_m, end_in_m, pitch, tie_start, tie_stop in fill(measures[m_idx]):
            abs_start = base + start_in_m
            while pending_idx < len(pending) and pending[pending_idx][0] <= abs_start:
                _, _n, kt, kv, step, alter = pending[pending_idx]
                m_xml += _harmony_xml(kt, kv, step, alter)
                pending_idx += 1
            m_xml += _note_xml(end_in_m - start_in_m, tpb, pitch,
                               tie_start=tie_start, tie_stop=tie_stop)
        measures_xml.append(f'<measure number="{m_idx + 1}">{m_xml}</measure>\n')

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN"'
        ' "http://www.musicxml.org/dtds/partwise.dtd">\n'
        '<score-partwise version="3.1">\n'
        '<part-list><score-part id="P1"><part-name>Melody</part-name></score-part></part-list>\n'
        '<part id="P1">\n'
        + ''.join(measures_xml)
        + '</part>\n</score-partwise>\n'
    )


def save_lead_sheet(midi_output_path, right_events, left_events, tempo_us, tpb,
                    time_sig=(4, 4), config=None):
    """Generate and save a -lead-sheet.musicxml file next to the MIDI output."""
    xml = build_musicxml(right_events, left_events, tempo_us, tpb, time_sig, config)
    if xml is None:
        print('Lead sheet: no melody notes, skipping.')
        return None
    out_path = os.path.splitext(midi_output_path)[0] + '-lead-sheet.musicxml'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(xml)
    print(f'Lead sheet: {out_path}')
    return out_path
