# Audio Features
## Extracting audio features
A film’s audio mix is made of three components: the score, the diegetic sound effects, and the dialogue. We can extract features from each, using a mix of audio processing libraries and pre-trained models. These features will also be useful in the feature, perhaps if we incorporate a neural network or other deep learning analyses.

## Current Scope: Score, Sound Effects, and Dialogue Tone
We can extract basic, low-hanging fruit features for now. In the future, with a better understanding of film score, diegetic sound effects, and audio engineering, we can draw more meaningful conclusions from each.

## Repository Files
The directory contains the following files:

1. *voice_tone.ipynb* - basic visualizations and voice emotional analysis using a pre-trained model

## Vision Feature Categories
A single movie frame can be a powerful medium for conveying emotion. While they communicate important plot information like onscreen characters and location, they can also influence the impact of onscreen action through subtle choices in light and color. Dramas may have scenes with very bright and very dark areas, a high contrast that doesn’t exist in well-lit comedies. Scenes that skew blue are “cooler”, and may depict locations that are unfeeling or clinical. These aspects don’t happen by chance – they are conscious directorial decisions.

## Audio Visualization
Audio analysis is heavily rooted in signal processing – these concepts should be familiar to electrical engineers. To process audio, we’ll have to convert audio files into usable representations in both the time domain and frequency domain. Luckily, we can use the librosa library. Created by LabROSA at Columbia University, this library contains all sorts of tools for audio processing.

librosa makes it easy to create two fundamental audio visualizations: the waveform plot and the spectrogram.

### Waveform Plot
The waveform plots amplitude vs. time. We can think of the amplitude as power, or volume. Here’s a waveform of a female character in Booksmart speaking a sentence for eight seconds. Since this is a stereo track, the left channel amplitude is above the axis and the right channel is below. The loudest portions of the sentence are denoted by the peaks.

![a waveform plot for a female character](/readme_images/waveplot.png "a waveform plot for a female character")

### Spectrogram
Next, we have the spectrogram, which represents frequency over time. This plot tells us the strongest frequencies of the audio signal at each timeframe. The frequency defines the profile of the audio signal. As an example, we can look at two spectrograms for a male voice and a female voice. Male and female voices have different frequencies: men between 80-180 Hz, and women between 1670-250 Hz.

![a spectogram for a female character](/readme_images/spectrogram_female.png "a spectogram for a female character")
![a spectogram for a male character](/readme_images/spectrogram_male.png "a spectogram for a male character")

## Voice Tone
Since the eventual goal of the project is to quantify emotion, we’ll want to measure emotion in voice tone. Since voice emotional analysis is a very popular subject, there are plenty of pre-trained models available. We used a model from GitHub user MITESHPUTHRANNEU. This model was trained on the famous RAVDESS dataset, which has male and female actors deliver lines with various emotions.

We were able to use the model on a plug-and-play basis using Keras’ models.load_model(). By loading a 2.5 second clip, it was able to identify gender, and one of five emotions. Of course, we can train our own models using the RAVDESS dataset, but it’s been done satisfactorily many times already, and we can gratefully use this work to further progress in the Moviegoer project.
