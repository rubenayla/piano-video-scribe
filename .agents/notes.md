# Notes

## MuseScore MIDI import: channels vs tracks (2026-03-11)

Tested all combinations to understand how MuseScore 4 interprets MIDI:

| Setup | Result |
|---|---|
| 2 tracks, 1 channel (ch0) | Grand staff with treble + bass clef |
| 3 tracks (conductor + 2), 1 channel | Same as above |
| 2 tracks, 2 channels (ch0 + ch1) | 2 separate instruments, no grand staff |
| 1 track, 2 channels (ch0 + ch1) | 2 separate instruments, no grand staff |

**Rule: channel = instrument in MuseScore.** Same channel across tracks → one instrument with multiple staves (grand staff). Different channels → separate instruments, even within one track.

For piano hand separation: use 2 tracks (Right Hand, Left Hand), both on channel 0. MuseScore assigns treble/bass clef automatically based on pitch range. No conductor track needed.
