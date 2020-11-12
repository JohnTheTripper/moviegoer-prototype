# Scene Boundary Partitioning
## Identifying Scene Boundaries with an Original Algorithm
Films are divided into individual scenes, a self-contained series of shots which may contain dialogue, visual action, and more. Being able to programmatically identify specific scenes is key to turning a film into structured data. We attempt to identify the start and end frames for individual scenes by using Keras’ VGG16 image model to group similar frames (images) into clusters known as shots. Then an original algorithm, rooted in film editing expertise, is applied to partition individual scenes.

## Current Scope: Identifying Two-Character Dialogue Scenes
As a starting point, we'll try and identify the basic building block of nearly every film: **the two-character dialogue scene**. These types of scenes are simply two characters sharing a conversation, but they are the primary driver of plot advancement: just two characters speaking, with no distractions. These scenes are also very information-dense. Two qualities make it possible for us to, given a set of input frames, **identify where they begin and end**:
- They're visually easy to identify. They are often shot using the the **Medium Close-Up shot**, a very recognizable cinematography shot. We've previously built a **CNN image classifier** from scratch to determine if frames/shots are Medium Close-Ups.
- They are comprised of predictable patterns of shots. In a two-character dialogue, the shots are usually presented as a pattern of **speaker A, speaker B, speaker A, speaker B**. Using **Keras' VGG16 image model and HAC clustering**, we will group individual frames into shots, and look for this A/B/A/B pattern.

Below is an example of a series of frames, grouped into shots, which form the A/B/A/B pattern. Each shot is an example of a Medium Close-Up.

![abab pattern and mcu example](/readme_images/abab.png "abab pattern and mcu example")

## Project Links
- [Presentation for Scene Boundary Partitioning](https://docs.google.com/presentation/d/1bUYIPmKFG0cvVYVDsd3v0mIRydsCCLgZe7pZsSwOBSI/edit?usp=sharing)

## Directory Files
The directory contains the following files:

### Dialogue Scene Partioning Files
1. *dialogue_scene_boundary.ipynb* - illustrates a start-to-finish walkthrough of the five-step algorithm of generating scene boundaries for two-character dialogue scenes
2. *dialogue_scene_boundary_aggregation.ipynb* - applies the algorithm to find two-character dialogue scenes across an entire film

### Support Files
- *scene_partition_io.py* - contains functions for the scene partitioning process

# Identifying Scene Boundaries for Two-Character Dialogue Scenes
## Overview
Our goal is to, given a set of input frames, identify the start frame and end frame for individual scenes. (This is completely unsupervised, but for the purposes of explanation, I'll comment on our progress, as well as provide visualization.) In this example, 400 frames, one taken every second from *The Hustle* (2019) are being fed into the algorithm. Keras' VGG16 image model is used to vectorize these images, and then unsupervised HAC clustering is applied to group similar frames into clusters. Frames with equal cluster values are similar, so a set of three consecutive frames with the same cluster value could represent a three-second shot of a character.

Here is the vectorization of our sample 400 frames from *The Hustle*.

![all frames](/readme_images/allframes.png "all frames")

In this example, we have two partial scenes and two complete scenes. Our goal is to identify the scene boundaries of each scene; in this example, we'll try and identify the boundaries of the blue scene. Again, I've colored in this visualization manually, to illustrate our "target".

![target clusters](/readme_images/clusters.png "target clusters")

## Five-Step Algorithm for Designating Scene Boundaries
### Step 1: Finding the A/B/A/B Shot Pattern
Among all 400 frames, we look for any pairs of shots that form an A/B/A/B pattern.
![abab pattern and mcu example](/readme_images/abab.png "abab pattern and mcu example")

### Step 2: Checking for MCUs
Finding four A/B/A/B patterns, we run each shot through the MCU image classifier we've previously trained. Two of the patterns were rejected because they contain a shot that doesn't pass the MCU check. In the below image, the top shot-pair represents our example scene.
![abab pattern and mcu example](/readme_images/mcu_check.png "abab pattern and mcu example")

### Step 3: Designating a Preliminary Scene Boundary: Anchor Start/End
Once we've confirmed that we're looking at Medium Close-Up shots, we can reasonably believe that we're looking at a two-character dialogue scene. We look for the first and last appearances of either shot (regardless of A or B). These frames define the Anchor Start and Anchor End Frames, a preliminary scene boundary.
![anchor boundaries](/readme_images/anchors.png "anchor boundaries")

### Step 4: Identify Cutaways
In between the Anchor Start and Anchor End are many other shots known as Cutaways. These may represent any of the following:
- POV shots, showing what characters are looking at offscreen
- Inserts, different shots of Speaker A or B, such as a one-off close-up
- Other characters, both silent and speaking

After we identify these Cutaways, we may be able to expand the scene's start frame backward, and the end frame forward. If we see these Cutaways again, but before the Anchor start or after the Anchor end, they must still be part of the scene.
![cutaways](/readme_images/cutaways.png "cutaways")

### Step 5a: Extending the Scene End
After the Anchor End are three frames with a familiar shot (cluster). Since we saw this cluster earlier, as a Cutaway, we incorporate these three frames into our scene. The following frames are unfamiliar, and are indeed not part of this scene.
![extending the scene end](/readme_images/extension_end.png "extending the scene end")

### Step 5b: Extending the Scene Start
We apply this same tecnique to the scene's beginning, in the opposite direction. We find many Cutaways, so we keep progressing earlier and earlier until no more Cutaways are found.
![extending the scene start](/readme_images/extension_start.png "extending the scene start")

## Evaluation
Below is a visualization of the total frames in the scene, with the blue highlighted frames included in our prediction, and the orange highlighted frames not included in our prediction. This algorithm managed to label most frames of the scene. Although some frames were missed at the scene's beginning, these are non-speaking introductory frames. The scene takes some time to get started, and we've indeed captured all frames containing dialogue, the most important criteria.

![evaluation](/readme_images/evaluation.png "evaluation")

# Future Improvements
### Two-Character Scene Partitioning
The clustering performed very well for certain scenes, but failed to identify others. Additional shot patterns (not just A/B/A/B) can be identified through further analysis. We can improve our existing scene boundary definitions, perhaps by looking for clusters that are similar (but not quite the same). This may help identify establishing (scene-starting) shots, which the current algorithm has never included.
