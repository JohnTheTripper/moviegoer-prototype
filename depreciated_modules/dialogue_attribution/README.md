# Dialogue Attribution
## Attributing Dialogue by Analyzing Frames, Audio, and Subtitles
With scenes identified, we can begin to analyze what's in those scenes. The biggest task is dialogue attribution, or parsing what each character is saying. Since the subtitle file has actual transcription, we don't need to conduct any sort of voice-to-text transcription.

## Current Scope: Attributing Dialogue to Characters
To establish the characters, we can independently cluster face encodings and also cluster voice encodings. But we don't have a way to automatically tie the two totgether. We must use various features in the three tracks of data (frames, audio, and subtitles) to links faces to voices. Only then we can attribute dialogue from the subtitles to these characters. There are two challenges: the subtitles do not indicate which character is speaking, and each stream of data uses a separate timestamp system.

## Repository Files
The repository contains the following files. The Attribution files, when read in order, begin with an exploration of the three trakcs of data: (visual) frames, audio, and subtitles. Then in *speaker_identification*, the frames and audio are linked by matching the onscreen speakers with the dialogue audio. Finally, in *subtitle_attribution*, we also link the written subtitle dialogue to each character.

### Attribution Files
1. *attribution_visual.ipynb* - explores frame data to determine extractable features
2. *attribution_audio.ipynb* - explores audio data to determine extractable features
3. *attribution_subtitle.ipynb* - explores subtitle data to determine extractable features
4. *speaker_identification.ipynb* - uses features from the three streams of data to tie together onscreen speakers and dialogue audio
5. *subtitle_attribution.ipynb* - ties subtitles to characters, achieving dialogue attribution

### Support Files
- *dialogue_attribution_io.py* - functions for extracting features from frames, audio, and subtitles, as well as automation of dialogue attribution

## Face and Voice Clustering
We need to identify the two characters in the scene by their faces and voices.

For facial clustering, we use the *face_recognition* library to create the face encodings, then *AgglomerativeClustering* from the *Keras* library to cluster them. We separate them into faces A and B.
For voices, we use the *pyAudioAnalysis* library to perform speaker diarization. Diarization breaks the audio track into a list of who-spoke-when (or who-spoke-last). These voices are separated into voices M and N.

## Visual, Audio, and Subtitle Attribution Flags
Though we have faces onscreen and we have voices on the audio track, we don't know how to tie them together. We've assembled a series of flags that serve as clues to identify who's speaking:
- Visual, Mouth Open: When a primary character is onscreen, we can reasonably assume they're speaking if their mouth is open.
- Audio, Audible Sound: A character can only be speaking if there isn't a period of silence.
- Subtitle, Subtitle Onscreen: A character can only be speaking if there are subtitles onscreen.

## Speaker Identification
We need to tie the faces and voices together, either as (Face A to Voice M and Face B to Voice N) OR (Face A to Voice N and Face B to voice M). We count the frames that support either scenario, when all the attribution flags are true.

## Subtitle Attribution
After tying together the faces and voices, we can attribute the written dialogue from the subtitles file to either character. The challenge here lies in the fact that the subtitle file doesn't designate speaker - it's up to the audience to infer who's speaking. We can use what we learned during the speaker diarization to figure out which character is speaking when, and then attribute each line to each character. This requires a number of functions related to time, because of the difference in frame/subtitle time systems, and the need for a time offset, in case the subtitles and audio track don't perfectly align.
