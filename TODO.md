# TODO

- [ ] Auto-detect BPM from audio using `librosa.beat.beat_track()` instead of asking the user.
      Could also detect time signature (4/4, 3/4, etc.).
      Reference: https://librosa.org/doc/latest/generated/librosa.beat.beat_track.html
- [ ] (Low priority) Adaptive tempo tracking for drifting tempos. Currently the BPM is fixed
      for the whole piece. For Synthesia videos this is fine (computer-generated, perfectly
      constant tempo — tested: ±1 frame jitter only, no drift). Would matter for human
      performance recordings or variable-speed videos. Approach: sliding window over note
      onsets, estimate local BPM, adjust quantization grid per section.
