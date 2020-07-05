# Project Overview and Scope
## Can a Machine Watch a Movie?
If we wanted to teach robots to understand human behavior, we would have them watch movies. Cinema is the medium with the closest approximation to reality. At their simplest, films portray mimicry of basic human behavior. At their deepest, films convey emotional reactions to psychological experiences. But movies are incredibly difficult for a machine to interpret. Whether we realize it or not, there are many filmmaking conventions which we take for granted (e.g. passage of time between scenes, dramatic music over a conversation, montages). We humans can understand how these affect a film, but can a robot?

The Moviegoer project has the lofty goal of unlocking the enormous wealth of emotional data within cinema, by turning films into structured data. Rather than create a single, all-encompassing model, Moviegoer will be pieced together iteratively, with various co-reinforcing modules based in transfer learning. There's been much research in the fields of emotional facial recognition, speech analysis, etc, and there are plenty of pre-trained models freely available. But none of these can't be applied to cinema until we can decode a movie's near-infinite possibilities into structured data.

Though portions of this project will be heavily reliant on pre-trained models and OTS libraries, I’ll be training original models and developing original algorithms, rooted in my expertise in filmmaking, cinematography, and film editing. (I’ve produced, directed, shot, and edited a feature-length movie, with another on the way.) Throughout this project, I’ll be providing visual explanations or examples from famous movies – I hope to enlighten the reader on various aspects of filmmaking as I justify my design decisions.

![credits](/readme_images/credits.png "credits")

## Why Movies?
Movies are chronologically linear, with clear cause-and-effect. Characters are established, emotional experiences happen to them, and they change. Consider a small-scale example: we see a character smiling, and two seconds later he’s frowning. What happened in those two seconds? Someone has said to him “I hate you”. We recognize this specific piece of dialogue is the antecedent, or stimulus to his emotional change. 

Though some films are freeform, open-ended, and experimental, many follow a specific recipe (think Acts 1, 2, and 3), and can be interpreted according to this formula. Interpreting a film as a holistic piece of work is a future goal, but for now, we’ll start smaller.

As a starting point, the project scope is limited to the two-character dialogue scene, the fundamental building block of nearly every film. From an information theory perspective, two-character dialogue scenes are very dense. No distractions, just two characters speaking and advancing the plot. In the future, advancements made in certain modules will pave the way for widening the scope beyond two-character dialogue scenes. 

# Repository Files
This project has three main modules, each focused on a specific task in turning films into structured data. **Each module is separated into a different directory, each with its own Readme file**, going into further detail on design decisions.

### Shot Recognition – CNN Image Recognition
There are a handful of very common cinematography (photography) shots used in most movies. This type of recognition can aid in identifying types of scenes, or certain cause-and-effect beats. The first type of shot recognized was the medium close-up, a shot commonly used in two-character dialogue scenes. We’ve trained a CNN (using an original, hand-labeled dataset of 11,000+ frames) to recognize these types of images. 

### Scene Boundary Identification – Shot Clustering and Original Algorithm
Movies can be broken down into individual scenes, self-contained units of dialogue and action. In the current scope, we’re looking to identify two-character dialogue scenes. These scenes have a distinct pattern: we see character A speak, then character B speak, back to A, then B, etc. Using the VGG16 image recognition model and HAC clustering, we’ve grouped frames into shots, and then created an original algorithm to look for these ABAB patterns of shots (assisted by the CNN image classifier above).

### Dialogue Attribution – Voice Clustering, Facial Analysis, and Subtitle Parsing
With scene boundaries identified, we can analyze individual scenes. The biggest task is dialogue attribution: determining which character is speaking.  A scene contains three streams of data: visual, audio, and subtitles. We need to be able to tie the onscreen characters in the frames, with the voices in the audio, with the written dialogue in the subtitles. We’ll glean clues from each of the three data streams on how to attribute dialogue.

### Other Files
These additional files are in the repository root:
- *extract.py* - generates screenshots from movie files
- *movies_cited.md* - contains a list of movies used in this project

# Current Effort: Dialogue Attribution in Two-Character Dialogue Scenes
We've identified the start and end points of two-character dialogue scenes, and the next step is to identify which character is speaking, and what they're saying. To do this, we'll be using clues from the three streams of data: visual, audio, and subtitles.

Previous effort allowed us to identify these types of scenes. Two qualities make it possible for us to, given a set of input frames, **identify where they begin and end**:
- They're visually easy to identify. They are often shot using the the **Medium Close-Up shot**, a very recognizable cinematography shot. We've built a **CNN image classifier** from scratch to determine if frames/shots are Medium Close-Ups.
- They're comprised of predictable patterns of shots. In a two-character dialogue, the shots are usually presented as a pattern of **speaker A, speaker B, speaker A, speaker B**. Using **Keras' VGG16 image model and HAC clustering**, we've created an algorithm to group individual frames into shots, and look for this A/B/A/B pattern.

Below is an example of a series of frames, grouped into shots, which form the A/B/A/B pattern. Each shot is an example of a Medium Close-Up.

![abab pattern and mcu example](/readme_images/abab.png "abab pattern and mcu example")

# Future Modules
There are many directions to take this project, and individual ideas will be explored on an exploratory basis. Proof-of-concept Python files/Notebooks may be posted to the repository, knowing that may not completely function without other portions of the project, for example subtitle NLP analysis will not be useful until a reliable way is found to attribute speaking characters to those subtitles. Still, this sort of POC exploration is necessary to decide which parts of the project (scene identification, persistent character identification, dialogue attribution, etc.) should be developed to advance the overall project. These are the ideas currently being explored:

### Definining Scene Boundaries
To supplement the technique of discovering A/B/A/B patterns through clustering and MCU classification, we can also use pre-built **facial recognition models to identify A/B/A/B patterns of characters**. This will solve the problem of scenes suddenly switching from Medium Close-Ups to Close-Ups mid-scene. The shots (clusters) would change (potentially breaking an A/B/A/B pattern) but the characters would persist and appear in an A/B/A/B pattern.

### Facial Emotion Analysis
Many **pre-built models are available for analyzing facial expressions** and assigning one of Darwin's six emotional states. As a bonus, some models can also approximate demographic information, such as age, gender, and race.

### Subtitle Analysis
Subtitles can be extracted and analyzed with various **NLP** techniques. Named Entity Recognition can be used to identify character names. We can also search for emotionally-charged words, or specifically look for phrases and words that accompany a sudden change in emotion.

![transfer learning](/readme_images/transfer.png "transfer learning")

# Additional Information
### Frames, not Videos
This project will strictly be using frames (screenshots), as opposed to video snippets (multiple frames), as input data. This has a number of benefits: reducing computational complexity, removing the need for recurrent elements of neural networks, and more granular data. This, of course, requires some sort of external timestamping system to track where frames occur in the film.

### Movie Copyright
40 films were used to train the CNN model, and additional films may be used for further training and development. Because these films are copyrighted, neither the model nor the dataset will be publicly released. A list of films used in this project is available in the movies_cited.md file.

This project is strictly for non-commercial educational and research purposes.
