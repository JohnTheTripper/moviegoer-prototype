# Audio Features
## Extracting audio features
A film’s audio mix is made of three components: the score, the diegetic sound effects, and the dialogue. We can extract features from each, using a mix of audio processing libraries and pre-trained models. These features will also be useful in the feature, perhaps if we incorporate a neural network or other deep learning analyses.

## Current Scope: Score, Sound Effects, and Dialogue Tone
We can extract basic, low-hanging fruit features for now. In the future, with a better understanding of film score, diegetic sound effects, and audio engineering, we can draw more meaningful conclusions from each.

## Repository Files
The directory contains the following files:

1. *voice_tone.ipynb* - basic visualizations and voice emotional analysis using a pre-trained model
2. *score.ipynb* - tempo, scale/chord identification, and diegesis testing

## Voice — Visualization and Tone Analysis
Audio analysis is heavily rooted in signal processing – these concepts should be familiar to electrical engineers. To process audio, we’ll have to convert audio files into usable representations in both the time domain and frequency domain. Luckily, we can use the librosa library. Created by LabROSA at Columbia University, this library contains all sorts of tools for audio processing.

librosa makes it easy to create two fundamental audio visualizations: the waveform plot and the spectrogram.

### Waveform Plot
The waveform plots amplitude vs. time. We can think of the amplitude as power, or volume. Here’s a waveform of a female character in Booksmart speaking a sentence for eight seconds. Since this is a stereo track, the left channel amplitude is above the axis and the right channel is below. The loudest portions of the sentence are denoted by the peaks.

![a waveform plot for a female character](/readme_images/waveplot.png "a waveform plot for a female character")

### Spectrogram
Next, we have the spectrogram, which represents frequency over time. This plot tells us the strongest frequencies of the audio signal at each timeframe. The frequency defines the profile of the audio signal. As an example, we can look at two spectrograms for a male voice and a female voice. Male and female voices have different frequencies: men between 80-180 Hz, and women between 1670-250 Hz.

![a spectogram for a female character](/readme_images/spectrogram_female.png "a spectogram for a female character")
![a spectogram for a male character](/readme_images/spectrogram_male.png "a spectogram for a male character")

### Voice Tone
Since the eventual goal of the project is to quantify emotion, we’ll want to measure emotion in voice tone. Since voice emotional analysis is a very popular subject, there are plenty of pre-trained models available. We used a model from GitHub user MITESHPUTHRANNEU. This model was trained on the famous RAVDESS dataset, which has male and female actors deliver lines with various emotions.

We were able to use the model on a plug-and-play basis using Keras’ models.load_model(). By loading a 2.5 second clip, it was able to identify gender, and one of five emotions. Of course, we can train our own models using the RAVDESS dataset, but it’s been done satisfactorily many times already, and we can gratefully use this work to further progress in the Moviegoer project.

## Score
A film's score adds emotional impact to the action onscreen. King Kong (1933) was the first film to incorporate a full orchestra score, adding weight to Kong's rampage. The famous shower scene in Pyscho (1960) wouldn't be nearly as frightening without the violin screeches. And the two notes of the shark's theme in Jaws (1975) were a big source of suspense for the POV underwater stalking scenes. A film score is a directorial decision, made to influence our emotional response. We can use the librosa library to conduct various analyses.

### Tempo and Chromagram
Lost in Translation (2003), a film about loneliness and isolation, benefits from a score with heavy use of shoegaze: slow, ethereal, and introspective. "Alone in Kyoto", by the electronic group Air, is a slow, minimalistic song that accompanies Scarlett Johansson's character as she wanders through Japanese temples alone, taking in the foreign sights and customs.
We can estimate the tempo, a relatively slow, calm 89 BPM. The chroma features can be extracted to estimate with notes/pitch classes are present at each time window. The chromagram provides a visualization — it looks like the D note is the most prevalent.

![chromagram for 'Alone in Kyoto'](/readme_images/chromagram.png "chromagram for 'Alone in Kyoto'")

### Scales and Chords – Major and Minor
At its most simplified level, the score music is happy or sad. As a general, broad rule of thumb, music composed in major scales are happy, and music composed in minor scales are sad. Scales are composed of seven of the 12 pitch classes. Using chroma_stft() gives us the mean intensity of all 12, making a best effort to group audio data into the 12 pitch classes. By looking at the list of top seven, we may be able to map these to major and minor scales.

We may also want to identify major and minor chords, with the help of the pychord library. Again, major chords are happy, and minor chords sad. Chords are made of three or four individual notes to create (what sounds like) a single tone. Here, we can find a minor chord, the diminished triad for the root note C (Cdim) in a French-inspired scene transition in The Hustle (2019). For each time window, we can look for each pitch above a certain intensity. In the below example, we found three semitones in this window, index numbers 0, 3, 6, corresponding to the notes C, E-flat, and G-flat.

![chord lookup](/readme_images/chord_lookup.png "chord lookup")

### Diegetic vs. Non-Diegetic Music
We’ll also want to differentiate between diegetic and non-diegetic music. Non-diegetic music has been overlaid on top of the film’s soundtrack, with the implication that it isn’t part of the in-movie story. Diegetic music is in-universe, and may be a song on a car radio, or a character singing karaoke. 
We may be able to tell the difference by looking at the song’s frequencies, specifically the spectrograms and frequency roll-offs.

We can also compare the frequency roll-offs, or the frequencies that contain a certain percentage of overall intensity. Typically when shaping audio to make it sound like it’s coming out of speakers onscreen, the high and low frequencies are reduced or rejected entirely with band-reject filters. Although further research is required to tell the difference, we can use these principles as a start.
