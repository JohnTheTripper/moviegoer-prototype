# moviegoer

## Data Understanding and Labeling
### Data Extraction
Frames (akin to screenshots) were extracted from 480p files from 40 films.

### Data Classification
Frames were labeled by hand, into the two categories **MCU** and **non-MCU**. Each frames was visually inspected and categorized. Though it seems like a daunting task, an MCU is immediately recognizable to any experienced cinematographer, especially if it's part of a two-character dialogue scene, or as an over-the-shoulder shot.

However, it is understood that hand-labeling is more of an art than a science. Leeway was skewed toward labeling as frames as MCUs, such as during non-dialogue scenes but the shot happens to be an MCU, or when it's a toss-up between MCU and non-MCU but during a dialogue scene. False Positives are tolerable to increase the MCU Recall rate.

### Frame Variety
Exposing the model to as many films as possible will prepare it to "watch" future, unseen films. Every film has a unique look, starting with the medium on which it was shot (graininess of 35mm film vs. the pristine look of digital CMOS sensors). Film selection also attempted to maximize variety in scene lighting (well-lit comedies vs. moody dramas), actors (non-white and child actors), and production value (slick blockbusters vs. naturalistic indies).

## Data Preparation
### Class Imbalance, and Abstaining from Data Augmentation
There was a strong class imbalance. After labeling, MCU frames only made **45%** of frames. This caused serious issues during preliminary modeling, where every Test frame was predicted to be a non-MCU. The *Data Management* notebook 
