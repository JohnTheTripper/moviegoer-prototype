# Vision Features
## Extracting visual features
We can extract many features to populate in the frame-level DataFrame. These can be used for other modules' purposes, such as scene boundary partitioning (for example, by detecting all-black frames between scenes) or persistent character identification. These features will also be useful in the feature, perhaps if we incorporate a neural network or other deep learning analyses.

## Current Scope: Frame-level features
Right now, only frame-level features are being detected. These are features that can be identified from a single frame, such as brightness or aspect ratio. This is in contrast to shot-level features, which rely on the analysis of multiple frames, like detecting movement (difference from frame to frame). This will avoid the need for a recurrent aspect of a neural network.

## Repository Files
The directory contains the following files:

1. *dimensions.ipynb* - explores features related to the image size, as well as identifies specific points-of-interest, like rule-of-thirds intersections
2. *chroma_luma.ipynb* - explores features related to color intensity (chroma) and overall brightness (luma)
3. *onscreen_text* - uses optical character recognition to detect and store onscreen text
4. *faces* - identifies number, size, and location of faces in frame
5. *vision_features_io.py* - contains all functions

## Vision Feature Categories
A single movie frame can be a powerful medium for conveying emotion. While they communicate important plot information like onscreen characters and location, they can also influence the impact of onscreen action through subtle choices in light and color. Dramas may have scenes with very bright and very dark areas, a high contrast that doesn’t exist in well-lit comedies. Scenes that skew blue are “cooler”, and may depict locations that are unfeeling or clinical. These aspects don’t happen by chance – they are conscious directorial decisions.

### Basic Dimensions
We can extract the height and width of each frame, giving us the aspect ratio. We can also search for artificial aspect ratios. Sometimes a film will show a flashback or a “film-within-a-film”, changing the aspect ratio to give it a vintage look.

![a scene from 'Vault'](/readme_images/vault_frame205.jpg "a scene from 'Vault'")

*The opening montage of “Vault” is a vintage montage from earlier days, using an artificial aspect ratio.*

With the basic frame dimensions, we can also calculate the center-point and rule-of-thirds points of the frame. Typically, cinematographers will position characters at the intersection of two-thirds from the bottom and one-third from the left or right edge. This is especially true in the staple shot of two-character dialogue scenes, the Medium Close-Up. We’ll be using these reference points later.

### Chroma and Luma
Cinematographers and colorists use light and color to shape the atmosphere of the scene. We can calculate a frame’s overall brightness as well as the contrast, a measure of brightness variation. This can differentiate if a scene is moody, emphasizing a few points of interest illuminated in light, or just dark.
We can also determine if there’s a dominant color in the frame. These frames are analyzed within the BGR (RGB) color space, an additive color space that uses the blue, green, and red primary colors to represent the entire spectrum of colors. These can be used to create the secondary colors yellow, magenta, and cyan. For example, yellow is created by adding green and red, but it can also be thought of as the absence of blue. Typically, scenes are balanced with a roughly equal amount of blue, green, and red, but we can calculate if a frame is composed of mostly a single primary or secondary color. A mostly-blue scene might be emotionally cold (or maybe it takes place underwater).

![a scene from 'Booksmart'](/readme_images/booksmart_frame3913.jpg "a scene from 'Booksmart'")

*This highly-stylized scene in “Booksmart” uses bright, colored lights to backlight its characters. We can detect a red dominant color.*

### Onscreen Text
Using optical character recognition (OCR), we can detect text printed onscreen. Unfortunately, we’ve gotten poor results, most likely because the 480p frame size results in very small text. We’ll get better results by using movies with a larger resolution.

### Faces
We can also detect the number and location of faces. We can also identify the “primary” character of a scene, if a physical face size is larger than other faces (in the background). With the location of faces, we can determine if they’re in the prototypical rule-of-thirds points (defined in “Basic Dimensions”). If they are, and they’re at least a certain size, we may be able to conclude the shot is a Medium Close-Up, a giveaway that this is part of a two-character dialogue scene.

![a scene from 'Parasite'](/readme_images/parasite_frame2673.jpg "a scene from 'Parasite'")

*In “Parasite”, Mr. Kim is the primary character, with the physically largest face, and lies atop a rules-of-thirds point.*
