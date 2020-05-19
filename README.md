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
### Populating Test and Train
The *Data Management* notebook contains functions allowing for the automated population of the Test and Train folders, for use in modeling. There are two data configurations:
1. Mixed Frames - This shuffles frames from all 40 films into a random split of 80% Train and 20% Test. This is the primary configuration used for preliminary training, as it allows for the most variety in frames for training. 
2. Unwatched Frames - This randomly splits the films into 32 Train and 8 Test movies; each film's frames will only be in Train OR Test. This simulates the testing of the model on films it hasn't watched before, but at this early stage of training, the 25% increase in frame variety is more important.

During modeling, `ImageDataGenerator()` provided innate support for creating a Validation set of 20% of the Train data.

### Class Imbalance, and Abstaining from Data Augmentation
There was a strong class imbalance. After labeling, MCU frames only made **45%** of frames. This caused serious issues during preliminary modeling, where every Test frame was predicted to be a non-MCU. The Test/Train population functions of *Data Management* notebook allow for a parameter *imbalance_removal*, which allows a certain percentage of non-MCU frames to be removed from the Training set. Various percentages (20%, 40%, 60%) were tested in the *Baseline Creation* notebook; 60% was ultimately chosen as it provided the best results (and provided an almost-equal distribution of mcu vs. non-MCU frames).

Keras' `ImageDataGenerator()` provides excellent options for data augmentation, to increase the variety of data, especially for an underrepresented class. After careful consideration, I declined to use any of these.
* width_shift_range, height_shift_range
* shear_range
* zoom_range
* horizontal_flip
* vertical_flip
