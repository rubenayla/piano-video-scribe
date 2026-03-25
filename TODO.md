# TODO

"Keyboard detection failed — only found 1 white key. The default frame likely doesn't show the keyboard clearly. Let me check frame 0 vs a later frame." what caused it to fail?

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
- [ ] Auto-calibrate saturation thresholds per video. Use delta detection on a few clear
      onsets to measure the actual saturation values when keys are pressed/released, then
      set SAT_ON/SAT_OFF thresholds automatically instead of using fixed values (70/40).
- [ ] (Low priority) Adaptive tempo tracking for drifting tempos. Currently the BPM is fixed
      for the whole piece. For Synthesia videos this is fine (computer-generated, perfectly
      constant tempo — tested: ±1 frame jitter only, no drift). Would matter for human
      performance recordings or variable-speed videos. Approach: sliding window over note
      onsets, estimate local BPM, adjust quantization grid per section.
