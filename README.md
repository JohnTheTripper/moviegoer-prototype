# Update
This repository is no longer maintained, though progress on Moviegoer continues! The latest project updates can be found at [moviegoer.ai]

# Project Overview and Scope

![logo](/readme_images/logo.png "logo")

## Can a Machine Watch a Movie?
If we wanted to teach robots to understand human behavior, we would have them watch movies. Cinema is the medium with the closest approximation to reality. At their simplest, films portray mimicry of basic human behavior. At their deepest, films convey emotional reactions to psychological experiences. But movies are incredibly difficult for a machine to interpret. Whether we realize it or not, there are many filmmaking conventions which we take for granted (e.g. passage of time between scenes, dramatic music over a conversation, montages). We humans can understand how these affect a film, but can a robot?

**The *Moviegoer* project has the lofty goal of unlocking the enormous wealth of emotional data within cinema by turning films into structured data.** This structured data can then be used to train emotional AI models. Cinema has the potential to be the world's largest dataset of emotional knowledge.

*Moviegoer* will be pieced together iteratively, with various co-reinforcing modules based in filmmaking domain knowledge. There's been much research in the fields of emotional facial recognition, speech analysis, etc, and there are plenty of pre-trained models freely available. But none of these can't be applied to cinema until we can decode a movie's near-infinite possibilities into structured data.

Though portions of this project will be heavily reliant on pre-trained models and OTS Python libraries (see Tech Stack section), I’ll be developing original algorithms, rooted in my expertise in filmmaking, cinematography, and film editing. (I’ve produced, directed, shot, and edited a feature-length movie, with another on the way.) Throughout this project, I’ll be providing visual explanations or examples from famous movies – I hope to enlighten the reader on various aspects of filmmaking as I justify my design decisions.

![credits](/readme_images/credits.png "credits")

## Why Movies?
Movies are chronologically linear, with clear cause-and-effect. Characters are established, emotional experiences happen to them, and they change. Consider a small-scale example: we see a character smiling, and two seconds later he’s frowning. What happened in those two seconds? Someone has said to him “I hate you”. We recognize this specific piece of dialogue is the antecedent, or stimulus to his emotional change.

Some additional information on why movies are the ultimate dataset of emotional knowledge can be found at [moviegoer.ai](https://moviegoer.ai).

# Four Categories of Comprehension
Continued progress has helped clarify the overall goals of the project — we’ve identified four broad categories of knowledge that Moviegoer must identify and recognize. These categories aren’t tied to any specific aspect of the tech stack, and advances in one category may support another. Much like a human viewer, a machine must be able to parse four categories of comprehension to “watch a movie”: film structure; characters; plot and events; and style features.

### Structure
Though taken completely for granted by an audience, a movie is comprised of granular, self-contained units called “scenes”, which usually take place at a fixed location and involve one or more characters. A machine must be able to recognize the beginnings and endings of individual scenes. This type of analysis can be extended to other units, from the three acts that form the entire film, all the way down to the individual shot.

As a starting point, *Moviegoer* identifies two-character dialogue scene, the fundamental building block of nearly every film. From an information theory perspective, two-character dialogue scenes are very dense. No distractions, just two characters speaking and advancing the plot.

![Lost in Translation Scene Identification](/readme_images/pd_scene_lit_1a.png "Lost in Translation Scene Identification")

### Plot
A film consists of many different events and happenings. A robot must understand how the emotional significance of specific events (e.g. a love interest gets a significant other, a rival gets a job promotion, a daughter becomes injured) and how they impact character goals.

![Plus One (2019) Scene Emotional Analysis](/readme_images/pd_scene_po_17.png "Plus One (2019) Scene Emotional Analysis")

### Characters
Each character has a unique face and voice, which can be tracked throughout the film. These help with predicting character demographics. In order to understand character motivations, spoken dialogue must be attributed to individual characters to identify what’s important to them.

![Plus One (2019) Character Information](/readme_images/pd_character_demographics_emotion.png "Plus One (2019) Character Information")

### Style
These are artistic choices, such as shot length or color scheme, used to elicit specific emotions in the audience. Music score is the most prominent example — although we understand this music doesn’t actually exist within the scene, it’s been layered on top of it to make the audience feel sad, excited, tense, or a multitude of other emotions.

This category of comprehension is simultaneously the most powerful as well as the most debatable. Recognizing these clues (and coding them into the project) relies heavily on domain knowledge in filmmaking. These empirical rules have evolved from over a century’s worth of advances in filmmaking, and require a strong understanding of the craft. At the same time, some directors will consciously flout these rules as an artistic choice, and *Moviegoer* must be ready to accept these scenarios. But, if a style rule helps us interpret emotion in 99% of films, across all genres, it’ll greatly help in interpreting films.

![Booksmart (2019) Color Shots](/readme_images/pd_color_shots.png "Booksmart (2019) Color Shots")

# Repository Files
This project contains several directories, each focused on a separated task. **Each directory has its own Readme file**, going into further detail on contents and design decisions.

For the purposes of the prototype, functions are contained within each individual module directory. This is for the purposes of small-scale development and demonstration of self-contained functionality. They will be eventually be centralized in a single directory with a proper init.py file.

## Data Exploration and Function Creation
Since a movie can be broken into three streams of data, there is one directory each for Vision, Audio, and Subtitle. These directories contain exploratory Jupyter notebooks, essentially playing with the data to see what features can be unearthed. After this EDA, function files are created, for later reference.

### Vision Features – Color, Composition, and Other Computer Vision Analyses
We can use computer vision to extract visual features from frames using computer vision. These features can be populated into the frame-level DataFrame. These features may deal with things like brightness or strong presence of specific colors. We can also use cinematography axioms, like the rule of thirds to define points-of-interest in frames.

### Audio Features – Speech Tone, Score, and Sound Effects
We're looking for three categories of features when analyzing a film's audio track: speech tone (*how* words are said), score (music), and specific sound effects. Speech tone will help with measuring emotion, while score can help determine the intended mood of the scene. Sound effects may be able to help with scene location identification — waves crashing implies the scene takes place at, or near the sea. Identifying specific sounds can also help with plot interpretation. Gunshots, car honking, and cash registers all have distinct auditory profiles, and are relatively unambigious in what they mean for the onscreen action.

### Subtitle Features – Audio Transcriptions and Descriptions
Subtitles contain the ground-truth dialogue - no audio-based transcription required. Using NLP, we can analyze the word choice and get a feel for sentiment and emotion. NER (named-entity recognition) can offer various clues like character names, and any locations mentioned by characters. Subtitles also contain lots of non-dialogue information, like the presence of sound effects or laughter.

## Data Processing
With functions created in the three separate streams, we begin to link these different streams together. We also build a pipeline to process and serialize film data. With this pipeline in place, we can put together a demonstration of *Moviegoer's* capabilities.

### Unifying Features – Tying It All Together
After taking a deep dive into the individual data streams of visuals, audio, and subtitle, we can use individual features from each to accomplish larger goals of the project. Using them in tandem will require unifying them on the same time reference, as well as some basic input/output to pass data to one another.

### Data Serialization – Processing Movie Data
These five files, when run in sequence, process a movie and turn it into the dataframes that power the analytical functions.

### Prototype Demo - Moviegoer Capabilities
This is a visual demonstration of how structured data from films can be used. The demo covers applications in each of the four categories of comprehension.

# Additional Information
### Movie Copyright
A list of films used in this project is available in the movies_cited.md file.

### Tech Stack
This project uses a number of Python libraries. As of the creation of the prototype, these are in use:
- Common
  - Python 3.7 (PyCharm 2020.2)
  - Jupyter Notebook 6.1 (Anaconda 2020.02)
  - TensorFlow 2.3 (Docker Community 19.03)
  - Pandas 1.1
  - NumPy 1.19
  - Matplotlib 3.2
  - Scikit-Learn 0.22
- Vision
  - face_recognition 1.3
  - deepface 0.0.26
  - OpenCV 4.2
  - PyTesseract 0.3
- Subtitle
  - SpaCy 2.3
  - pysrt 1.1
- Audio
  - pyAudioAnalysis 0.3
  - Librosa 0.7

### Contact
[moviegoer.ai](https://moviegoer.ai)

[contact@moviegoer.ai](mailto:contact@moviegoer.ai)

This project is strictly for non-commercial educational and research purposes.
