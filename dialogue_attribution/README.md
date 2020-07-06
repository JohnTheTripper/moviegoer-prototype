# Dialogue Attribution
## Attributing Dialogue by Analyzing Frames, Audio, and Subtitles
With scenes identified, we can begin to analyze what's in those scenes. The biggest task is dialogue attribution, or parsing what each character is saying. Since the subtitle file has actual transcription, we don't need to conduct any sort of voice-to-text transcription.

## Current Scope: Attributing Dialogue to Characters (Current Effort)
To establish the characters, we cluster face encodings and voice encodings and then use various features in the three streams of data (frames, audio, and subtitles) to tie faces to voices. Then we can attribute dialogue from the subtitles to these characters. There are two challenges: the subtitles do not indicate which character is speaking, and each stream of data uses a separate timestamp system.

## Repository Files
The repository contains the following files. The Attribution files, when read in order, begin with an exploration of the three streams of data: (visual) frames, audio, and subtitles. Then in *speaker_identification*, the frames and audio are linked by matching the onscreen speakers with the dialogue audio. Finally, in *subtitle_attribution*, we also link the written subtitle dialogue to each character.

### Attribution Files
1. *attribution_visual.ipynb* - explores frame data to determine extractable features
2. *attribution_audio.ipynb* - explores audio data to determine extractable features
3. *attribution_subtitle.ipynb* - explores subtitle data to determine extractable features
4. *speaker_identification.ipynb* - uses features from the three streams of data to tie together onscreen speakers and dialogue audio
5. **subtitle_attribution.ipynb** - ties subtitles to characters, achieving dialogue attribution **(Current Effort)**

### Support Files
- **dialogue_attribution_io.py** - functions for extracting features from frames, audio, and subtitles, as well as automation of dialogue attribution **(Current Effort)**
