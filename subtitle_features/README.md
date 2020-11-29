# Subtitle Features
## Extracting subtitle features
With a film's subtitles, we can extract the ground-truth transcription of the spoken dialogue. We'll eventually be able to use this to analyze the film's entire plot. But for now, we can look for localized information, such as sentiment analysis conducted on word choice. We can even count up the most common spoken names, to try and determine character names. And since subtitles contain descriptions of non-dialogue audio, like sound effects or laughter, we may be able to tie this to onscreen action or the audio track.

![subtitle with both dialogue and non-dialogue descriptions](/readme_images/subtitle_laugh.png "subtitle with both dialogue and non-dialogue descriptions")

Subtitles describe both dialogue and non-dialogue utterances from characters.

## Repository Files
The directory contains the following files:

1. *subtitle_cleaning.ipynb* - parses and cleans the subtitle files into an NLP-friendly format
2. *subtitle_nlp.ipynb* - NLP-based analysis of subtitle text
3. *subtitle_dataframes.ipynb* - creating three types of dataframes of subtitle features
4. *character_identification.ipynb* - identifying characters based on dialogue (e.g. "Nice to meet you, I'm Jim.")
5. *word_importance.ipynb* - identifying the words most important to scenes
6. *subtitle_auxiliary.ipynb* - other subtitle-related tricks for finding film information
7. *subtitle_cleaning_io.py* - functions for automatically cleaning subtitles
8. *subtitle_dataframes_io.py* - functions for creating subtitle-related dataframes
9. *subtitle_auxiliary_io.py* - functions for film-level subtitle information
10. *phrases_io.py* - functions for identifying specific words and phrases

## Subtitle Format
Subtitle files can be extracted from a movie in the form of a .srt file. These are basically just text files, but with very strict formatting. Each subtitle has a unique ID, a start and end time (indicating that this should be displayed at HH:MM:SS:MIL and end at HH:MM:SS:MIL), and one or two lines for the actual subtitle text to be displayed.

![subtitle example](/readme_images/subtitles.png "subtitle example")

## Subtitle Cleaning
### Line Separation and Concatenation
First, we’ll properly separate subtitles into individual lines. Remember that subtitle text is either one or two lines. This leads to three cases:
- One line: This is a single-line piece of dialogue or auditory description. This doesn’t require any cleaning.
- Two lines, one speaker: This is a piece of dialogue spoken by a single character that spans both lines. These should be concatenated.
- Two lines, two speakers: This is two separate speakers, each speaking a small piece of dialogue. These should be separated into two lines.

![two lines, two speakers, with someone speaking from offscreen](/readme_images/subtitle_offscreen.png "two lines, two speakers, with someone speaking from offscreen")

This single subtitle has two characters speaking (and a speaker name to indicate the offscreen speaker).


### Text Cleaning for NLP
Since subtitles are created for the hearing-impaired, they also convey non-dialogue information important to the film, such as a character laughing, or the identify of a character speaking from off-screen. We’ll want to parse each line and watch out for these – we’ll remove them for the purposes of data cleaning for NLP input, but make note of them for later analyses. Here’s a few examples of what we’ll be cleaning:
- Italics: An entire line is italics, denoted by the HTML tags “<i>” and “</i>. Italics are often used to designate narration, or someone speaking offscreen over the phone. We’ll discard the HTML tags but keep the rest of the text.
- Music: Song lyrics begin and end with a music note, regardless of whether they’re diegetic (like characters signing karaoke) or non-diegetic (like music overlaid on a montage). We’ll discard all of these and not use them in NLP analysis.
- Parenthetical: Full-line parentheticals describe both sound effects, and non-dialogue sounds from characters like “(grunting)”. We’ll remove these lines from the NLP input.
- Laughter: A character’s laughter is often included in the subtitles, as something like “(laughter)”. Fortunately, there are few enough laughter strings to look for, and we can just create a list containing phrases like “(chuckles)”, “(laughing)”, and “(laughter)”. We’ll remove these from the subtitle text but keep all the other dialogue.
- Speaker: When a character is speaking from off-screen, their name will be displayed with their subtitle text. When watching a film, we can recognize the voice of an off-screen speaker, but the hearing-impaired don’t have that luxury. We’ll remove the off-screen character’s name from the text, but we can save this name for later, since it directly tells us who’s speaking.

![subtitle only containing a parenthetical '(SPEAKING JAPANESE)'](/readme_images/subtitle_parenthetical.png "subtitle only containing a parenthetical '(SPEAKING JAPANESE)'")

This subtitle contains no language information, and will be removed before NLP analysis.
