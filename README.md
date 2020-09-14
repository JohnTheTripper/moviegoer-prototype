# Project Overview and Scope
## Can a Machine Watch a Movie?
If we wanted to teach robots to understand human behavior, we would have them watch movies. Cinema is the medium with the closest approximation to reality. At their simplest, films portray mimicry of basic human behavior. At their deepest, films convey emotional reactions to psychological experiences. But movies are incredibly difficult for a machine to interpret. Whether we realize it or not, there are many filmmaking conventions which we take for granted (e.g. passage of time between scenes, dramatic music over a conversation, montages). We humans can understand how these affect a film, but can a robot?

**The *Moviegoer* project has the lofty goal of unlocking the enormous wealth of emotional data within cinema by turning films into structured data.** Rather than create a single, all-encompassing model, *Moviegoer* will be pieced together iteratively, with various co-reinforcing modules based in transfer learning or filmmaking domain knowledge. There's been much research in the fields of emotional facial recognition, speech analysis, etc, and there are plenty of pre-trained models freely available. But none of these can't be applied to cinema until we can decode a movie's near-infinite possibilities into structured data.

Though portions of this project will be heavily reliant on pre-trained models and OTS libraries, I’ll be training original models and developing original algorithms, rooted in my expertise in filmmaking, cinematography, and film editing. (I’ve produced, directed, shot, and edited a feature-length movie, with another on the way.) Throughout this project, I’ll be providing visual explanations or examples from famous movies – I hope to enlighten the reader on various aspects of filmmaking as I justify my design decisions.

![credits](/readme_images/credits.png "credits")

## Why Movies?
Movies are chronologically linear, with clear cause-and-effect. Characters are established, emotional experiences happen to them, and they change. Consider a small-scale example: we see a character smiling, and two seconds later he’s frowning. What happened in those two seconds? Someone has said to him “I hate you”. We recognize this specific piece of dialogue is the antecedent, or stimulus to his emotional change. 

Though some films are freeform, open-ended, and experimental, many follow a specific recipe (think Acts 1, 2, and 3), and can be interpreted according to this formula. Interpreting a film as a holistic piece of work is a future goal, but for now, we’ll start smaller.

As a starting point, the project scope is limited to the two-character dialogue scene, the fundamental building block of nearly every film. From an information theory perspective, two-character dialogue scenes are very dense. No distractions, just two characters speaking and advancing the plot. In the future, advancements made in certain modules will pave the way for widening the scope beyond two-character dialogue scenes. 

# Repository Files
This project has five main modules, each focused on a specific task in turning films into structured data. **Each module is separated into a different directory, each with its own Readme file**, going into further detail on design decisions.

### Shot Recognition – CNN Image Recognition
There are a handful of very common cinematography (photography) shots used in most movies. This type of recognition can aid in identifying types of scenes, or certain cause-and-effect beats. The first type of shot recognized was the medium close-up, a shot commonly used in two-character dialogue scenes. We’ve trained a CNN (using an original, hand-labeled dataset of 11,000+ frames) to recognize these types of images. 

### Scene Boundary Identification – Shot Clustering and Original Algorithm
Movies can be broken down into individual scenes, self-contained units of dialogue and action. In the current scope, we’re looking to identify two-character dialogue scenes. These scenes have a distinct pattern: we see character A speak, then character B speak, back to A, then B, etc. Using the VGG16 image recognition model and HAC clustering, we’ve grouped frames into shots, and then created an original algorithm to look for these ABAB patterns of shots (assisted by the CNN image classifier above).

### Dialogue Attribution – Voice Clustering, Facial Analysis, and Subtitle Parsing
With scene boundaries identified, we can analyze individual scenes. The biggest task is dialogue attribution: determining which character is speaking.  A scene contains three streams of data: visual, audio, and subtitles. We need to be able to tie the onscreen characters in the frames, with the voices in the audio, with the written dialogue in the subtitles. We’ll glean clues from each of the three data streams on how to attribute dialogue.

### Vision Features – Color, Composition, and Other Computer Vision Analyses
We can use computer vision to extract visual features from frames using computer vision. These features can be populated into the frame-level DataFrame. These features may deal with things like brightness or strong presence of specific colors. We can also use cinematography axioms, like the rule of thirds to define points-of-interest in frames.

### Audio Features – Speech tone, Score, and Sound Effects
We're looking for three categories of features when analyzing a film's audio track: speech tone (*how* words are said), score (music), and specific sound effects. Speech tone will help with measuring emotion, while score can help determine the intended mood of the scene. Sound effects may be able to help with scene location identification — waves crashing implies the scene takes place at, or near the sea. Identifying specific sounds can also help with plot interpretation. Gunshots, car honking, and cash registers all have distinct auditory profiles, and are relatively unambigious in what they mean for the onscreen action.

### Subtitle Features – Speech tone, Score, and Sound Effects *(Current Effort)*
Subtitles contain the ground-truth dialogue - no audio-based transcription required. Using NLP, we can analyze the word choice and get a feel for sentiment and emotion. NER (named-entity recognition) can offer various clues like character names, and any locations mentioned by characters. Subtitles also contain lots of non-dialogue information, like the presence of sound effects or laughter.

### Other Files
These additional files are in the repository root:
- *extract.py* - generates screenshots from movie files
- *movies_cited.md* - contains a list of movies used in this project

# Current Progress
We've identified the start and end points of two-character dialogue scenes. We've also come up with a makeshift way of attributing spoken dialogue and written subtitles to each character.

Two qualities make it possible for us to, given a set of input frames, **identify where two-character dialogue scenes begin and end**:
- They're visually easy to identify. They are often shot using the the **Medium Close-Up shot**, a very recognizable cinematography shot. We've built a **CNN image classifier** from scratch to determine if frames/shots are Medium Close-Ups.
- They're comprised of predictable patterns of shots. In a two-character dialogue, the shots are usually presented as a pattern of **speaker A, speaker B, speaker A, speaker B**. Using **Keras' VGG16 image model and HAC clustering**, we've created an algorithm to group individual frames into shots, and look for this A/B/A/B pattern.

Below is an example of a series of frames, grouped into shots, which form the A/B/A/B pattern. Each shot is an example of a Medium Close-Up.

![abab pattern and mcu example](/readme_images/abab.png "abab pattern and mcu example")

With these scenes identified, we cluster face and voice encodings (independently of each other). We tie them together using clues from the visual, audio, and subtitle channels. Once faces and voices are tied together, we can look into the subtitle data and build lists of dialogue for each character.

# Four Categories of Comprehension
Continued progress has helped clarify the overall goals of the project — we’ve identified four broad categories of knowledge that Moviegoer must identify and recognize. These categories aren’t tied to any specific aspect of the tech stack, and advances in one category may support another. Much like a human viewer, a machine must be able to parse four categories of comprehension to “watch a movie”: film structure; characters; plot and events; and emotional and style features.

### Structure
An individual scene is a granular, self-contained component of every film. It has a fixed location, a set number of characters, and conveys one or more story beats. A scene can be analyzed individually, or compared against other scenes. Earlier in the project, we created an algorithm to identify a specific type of scene: the two-character dialogue scene. But we’ll need to be able to divide the entire film into its individual scenes.

We’ll also want to divide the film into its eight sequences. Many films follow the eight-sequence-approach, which can be thought of a more detailed breakdown of the three-act structure. These eight sequences, each lasting roughly 15 minutes in a two-hour film, denote (broadly) when major plot points are supposed to be unfold and when new characters might be introduced. Each of the eight sequences ends in a climax — this could be an important clue when identifying major plot points.

### Characters
We’ll need to persistently track characters throughout the entire film, to track their events and emotional changes. We can look for the vectorizations of their face and voice throughout the entire film, locating in which scenes they appear. We’ll also need to attribute dialogue to each character, using NLP on the subtitles to understand what they’re saying.

Films elicit responses through their characters’ emotions, and we’ll also need to monitor their emotions throughout the film. We can track their ups and downs through analysis of their facial expression, voice tone, and word choice, and see what antecedents triggered those emotional changes.

### Plot and Events
A plot consists of many different events and happenings. We’ll need to use context to understand where a scene is taking place, and what’s happening. Maybe an outdoor scene on a boat can be identified by the sound of waves crashing. Dialogue with a previously-unknown character about appetizers or entrees may hint a character is ordering with the waiter at a restaurant.

This particular category might be the most difficult to populate, and its conclusions might be filled with qualifiers and “best guesses”.

### Emotional and Style Features
Emotional and style features are somewhat “intangible”, and subject to interpretation. These are directorial choices used to elicit specific emotions in the audience. Music score is the most prominent example — although we understand this music doesn’t actually exist within the scene, it’s been layered on top of it to make the audience feel sad, excited, tense, or a multitude of other emotions.

Color and brightness can easily be quantified with computer vision. Dark scenes might be moody or uneasy. A scene tinted blue is “cool”, and the location or situation could be inhospitable or foreign.

The cinematography, or shot choice, can also be scrutinized. A character’s face may fill the frame to emphasize a facial reaction, or we may see his entire body from afar to emphasize loneliness or emotional distance. A shot might be looking down at a character to make her seem powerless, or looking up at her for the opposite effect.

This category of comprehension is simultaneously the most powerful as well as the most debatable. Recognizing these clues (and coding them into the project) relies heavily on domain knowledge in filmmaking. These empirical rules have evolved from over a century’s worth of advances in filmmaking, and require a strong understanding of the craft. At the same time, some directors will consciously flout these rules as an artistic choice, and Moviegoer must be ready to accept these scenarios. But, if a style rule helps us interpret emotion in 99% of films, across all genres, it’ll greatly help in interpreting films.

![transfer learning](/readme_images/transfer.png "transfer learning")

# Additional Information
### Frames, not Videos
This project will strictly be using frames (screenshots), as opposed to video snippets (multiple frames), as input data. This has a number of benefits: reducing computational complexity, removing the need for recurrent elements of neural networks, and more granular data. This, of course, requires some sort of external timestamping system to track where frames occur in the film.

### Movie Copyright
40 films were used to train the CNN model, and additional films may be used for further training and development. Because these films are copyrighted, neither the model nor the dataset will be publicly released. A list of films used in this project is available in the movies_cited.md file.

This project is strictly for non-commercial educational and research purposes.
