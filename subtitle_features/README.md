# Subtitle Features
## Extracting subtitle features
With a film's subtitles, we can extract the ground-truth transcription of the spoken dialogue. We'll eventually be able to use this to analyze the film's entire plot. But for now, we can look for localized information, such as sentiment analysis conducted on word choice. We can even count up the most common spoken names, to try and determine character names. And since subtitles contain descriptions of non-dialogue audio, like sound effects, we may be able to tie this to onscreen action or the audio track.

## Current Scope: Local Features
Right now, we'll see what local features we can unlock, such as analyzing speech and descriptions of sound effects. In the future, we'll see how we can use the subtitle text to interpret the film's plot.

## Repository Files
The directory contains the following files:

1. *subtitle_cleaning.ipynb* - parses and cleans the subtitle files into an NLP-friendly format
2. *subtitle_cleaning_io.py* - functions for automatically cleaning subtitles
3. *subtitle_nlp.ipynb* - NLP-based analysis of subtitle text
