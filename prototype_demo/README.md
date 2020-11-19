# Prototype Demo
This directory serves as a technical demonstration of *Moviegoer's* capabilities. The Jupyter notebooks offer a view of how the code is used, but since this project is inherently visual, it's best to follow along with the images in this Readme.

The examples below were taken straight from the prototype, with no pre-processing cleaning required of the input movie data. Instructions for using *Moviegoer* to replicate these results are located in the data_serialization/ directory. 

## Repository Files
The directory contains the following files:

1. *structure_plot.ipynb* - identifying individual scene and conducting NLP analyses on scene dialogue
2. *characters.ipynb* - tracking character emotion and predicting demographic information
3. *style.ipynb* - searching for various style features

The following sections correspond to the four categories of comprehension, detailed in the main repo Readme.

# Structure
Without any structure, a film is just a collection of a few thousand frames of images and a very long audio track. Conversations bleed into one another, characters appear and disappear without reason, and we teleport from one location to the next. We can begin to organize a film by dividing it into individual scenes.

## Scene Identification
We have an algorithm to identify scenes, partitioning them by identifying their first and last frames. Currently, we're focused on a specific type of scene: the two-character dialogue scene. These scenes are the basic building blocks of cinema: two characters speaking to each other, with no distractions, purely advancing the plot.
Let's take a look at *Lost in Translation* (2003), a famously quiet film, light on dialogue. This is the first scene we've identified, which is also the first time that our characters Bob and Charlotte have a conversation. It doesn't occur until 31 minutes into the film - again, the film is sparse on dialogue.

![Lost in Translation Scene Identification](/readme_images/pd_scene_lit_1a.png "Lost in Translation Scene Identification")

In modern filmmaking, two-character dialogue scenes follow a very distinct pattern. Character A speaks, then Character B, then back to A, then to B, etc. The film cuts back and forth between these two shots, the Anchor shots. There's a little more magic to the algorithm, including the identification of Cutaway shots, such as the two-shot where they're both sitting at the bar. We've now discovered a handful of scenes, which we'll be using for Plot and Character analyses.

# Plot
To understand a film, a machine needs to comprehend the significance of the various happenings of a film. This can most effectively be accomplished by analyzing the dialogue and identifying key points of emotional expression.

## Key Dialogue
With the scene boundaries of the above identified scene, we can isolate dialogue and analyze them as a single conversation. Below, we've identified some important pieces of dialogue and events, and mapped them back to their frames.

This is visualized to convey the scene's scope, but it's important to note that this image is mostly just for our interpretation ; Moviegoer doesn't actually need to "see" these frames.

![Lost in Translation (2003) Scene Visualization](/readme_images/pd_scene_lit_1b.png "Lost in Translation (2003) Scene Visualization")

We conduct an NLP analysis on the dialogue to try and understand what's happening. In particular, we isolate every Directed Question, questions that address the other person as "you". These types of questions usually elicit a personal response from the other character. Below is a mapping of every frame of interest:
- First and Last Frames
- Icebreaker and Kicker (First, Last Three Lines of Dialogue)
- Directed Questions and Responses
- Laughter

![Lost in Translation (2003) Scene Key Dialogue](/readme_images/pd_scene_lit_1c.png "Lost in Translation (2003) Scene Key Dialogue")

This scene had 6 Directed Questions. Since this scene was the first time the characters spoke to one another, they were getting to know each other by asking them personal questions.

## Emotional Analysis, at the Scene Level
With individual scenes identified, we can analyze the emotional content within each scene. In this scene, Bob and Charlotte reconcile after a fight. It's a quiet scene, even for a quiet film. It has a very slow conversation cadence of 8 sentences per minute (vs. the film's baseline of 15 sentences per minute). The emotional impact comes not from the dialogue, but from the characters' facial features as they look at each other in silence.

![Lost in Translation (2003) Scene Emotional Analysis](/readme_images/pd_scene_lit_4.png "Lost in Translation (2003) Scene Emotional Analysis")

We can calculate their Primary Emotion by measuring their facial emotion in each scene they appear, and then picking the most common emotion. Charlotte, sad about their impending separation, has a Sad face in almost 40% of her frames. Bob, played by the notoriously deadpan Bill Murray, has a Neutral look for the majority of the scene.

## Finding Drama
Next, we take a look at a scene from Plus One (2019), a romantic comedy. Two-character dialogue scenes full of sharp dialogue are a staple of rom-coms. We've identified 18 of these ;  let's take a look at Scene 17.

![Plus One (2019) Scene Emotional Analysis](/readme_images/pd_scene_po_17.png "Plus One (2019) Scene Emotional Analysis")

It has twice as much profanity as the film's average, indicating that it might be a dramatic scene. Profanity is an example of a measure of drama, and we can compare these indicators against the film's baseline, to find the most dramatic scenes.

Also of note are the First-Person Declarations we identified. These are sentences where a character declares something, with one's self as the subject. (It's easier understood by looking at the examples above.)

# Characters
A film conveys its emotional responses through its characters. Since we'll eventually want to determine what causes characters' emotions to change, we need to track characters and their emotions throughout the film.

## Finding Characters' Scenes
Since we've previously identified scenes, as well as the facial identities of their participants, we can search for all scenes in which a character appears. We discovered 18 scenes in Plus One, and Alice was found as a participant in 13 of them.

![Plus One (2019) Finding Characters' Scenes](/readme_images/pd_character_scenes.png "Plus One (2019) Finding Characters' Scenes")

## Character Information

We can guess Ben's demographic information with facial recognition models. We were able to guess that he is white, male and 32 years old. The actor playing Ben, Jack Quaid, is 28 years old, so this was a pretty good guess.

We also want to plot Ben's emotions through the film. We count up the times he appears "Sad" and "Angry", and group those into "Upset". We can then plot these Upset emotions across the film. This plot roughly tracks with the traditional three-act structure - lots of drama at the film's climax, before culminating in a happy ending.

![Plus One (2019) Character Information](/readme_images/pd_character_demographics_emotion.png "Plus One (2019) Character Information")

# Style

A film is more than just dialogue. There are many style features meant to influence the emotional impact of a particular scene. Below are three types of features for which we can look. Though they don't quite have a definitive meaning, we can still infer information from each.

## Color Shots

Every movie frame can be broken down to its RGB values, or additive color components. Each pixel in a frame has a red, green, and blue value, and they can be averaged into three values representing the entire image. These three value tend to be relatively balanced, but we can look for frames where they aren't: images that skew toward one of the primary additive colors red, green, or blue; or images that lack one of the primary colors, skewing toward the secondary colors yellow, cyan, or magenta.

These color images may be the result of creative lighting, or just the context of the scene (e.g. underwater or containing fire). The three most prominent examples from the high school comedy Booksmart (2019), are all from dialogue-free set pieces: a dream sequence dance with a crush, a karaoke party, and an underwater chase.

![Booksmart (2019) Color Shots](/readme_images/pd_color_shots.png "Booksmart (2019) Color Shots")

## Non-Conforming Aspect Ratios

Certain shots of a film might be displayed in an aspect ratio different than the rest of the film. For example, a more widescreen aspect ratio might be used to show a "film within a film", or a more square ratio for an "old-timey" flashback. In Booksmart, all the frames with non-conforming aspect ratios are used to display footage seen on characters' phones.

![Booksmart (2019) Non-Conforming Aspect Ratios](/readme_images/pd_aspect_ratio.png "Booksmart (2019) Non-Conforming Aspect Ratios")

## Long Takes
Long takes are shots that are held for a period of time - think the action sequences from Children of Men (2006). A long shot builds tension and suspense, and they're not just for action scenes: they're effective for dialogue as well. 

Ford v Ferrari (2019), a motorsport drama, is filled with racing scenes that use short, fast shots to emphasize speed and white-knuckle action. But it also uses long takes effectively. Here are three examples: a monologue about the challenges of endurance racing, a driver's conversation with his son about the mythical perfect lap, and a pre-credits ride (drive) into the sunset. Long takes were used to emphasize the importance of the monologue and conversation; we infer that the dialogue content is of particular importance to the characters.

![Ford v Ferrari (2019) Long Takes](/readme_images/pd_long_takes.png "Ford v Ferrari (2019) Long Takes")
