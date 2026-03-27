# TODO
any room to simplify, make the code more elegant while passing tests good?

- [x] "Keyboard detection failed — only found 1 white key." — Fixed: auto-detect keyboard
      frame by scanning forward for a frame with regularly-spaced white keys + black keys.
      Cause: videos with intros have blank/title frames at the default frame 5.
      Also falls back to auto-scan if a config-specified frame doesn't show a keyboard.

- [ ] I want to use Karpathy's autoresearch algorithm to improve this software, but for that i need a clear metric of improvement. I dont know if i can trust you to manually check sheets or we can do other checks, like using videos with known melodies and count number of notes... maybe download more videos for those known melodies, or you can think of other better ways to check that the software is good. I'd like to try lots of different detections methods, resistant to glow sparkly stuff etc. And as generic as possible, so it works for many keyboards with just a few adjustments. As elegant as possible.
- [x] Auto-detect BPM from audio using `librosa.beat.beat_track()` — done, `--bpm` is now optional.
      Could also detect time signature (4/4, 3/4, etc.).
- [ ] Robustness against screen-recorded videos. Screen recordings from phones/tablets
      include navigation bars, status bars, and other UI overlays that sit on or near the
      keyboard region. Current fix: reject saturated pixels whose hue doesn't match any
      configured hand color. Further improvements:
      - Auto-detect the keyboard bounding box and clip detector regions to it (ignore
        pixels outside the actual keyboard area).
      - Detect persistent saturation (>5s at same position) as non-musical artifacts.
      - Use the clean reference frame to build a mask of "non-keyboard" regions.
- [ ] **Falling blocks: auto color detection via spatial statistics.**
      For the region above the keyboard (falling blobs area only):
      1. Sample random frames scattered across the video
      2. Apply high-saturation filter (S>80) to remove background
      3. Find the 2-3 most common remaining hue clusters — these are the blob colors
      4. Split each frame into left half and right half
      5. Count each color's pixel frequency in left vs right across all sampled frames
      6. The color dominant on the right side = right hand, dominant on left = left hand
      This works because RH notes are higher pitch (right side of keyboard = right side of frame)
      and LH notes are lower pitch (left side). Pure traditional programming, no ML needed.
      Could be improved further by Karpathy-style autoresearch — let the AI find even better
      heuristics for color/hand assignment.

- [ ] **Autoresearch: autonomous improvement loop (Karpathy-style).**
      Use our high-fidelity synthetic tests (midi2video rendered, 100% correct GT) as the
      scoring function. The loop: read score.py output → analyze failures → hypothesize fix →
      implement → re-score → keep if improved, revert if not. Target: 100% on test11/test12
      using the falling blocks detector. Then expand to real videos.

- [ ] Auto-calibrate saturation thresholds per video. Use delta detection on a few clear
      onsets to measure the actual saturation values when keys are pressed/released, then
      set SAT_ON/SAT_OFF thresholds automatically instead of using fixed values (70/40).
- [ ] (Low priority) Adaptive tempo tracking for drifting tempos. Currently the BPM is fixed
      for the whole piece. For Synthesia videos this is fine (computer-generated, perfectly
      constant tempo — tested: ±1 frame jitter only, no drift). Would matter for human
      performance recordings or variable-speed videos. Approach: sliding window over note
      onsets, estimate local BPM, adjust quantization grid per section.
