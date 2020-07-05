# Shot Modeling
## Recognizing Cinematography Shots Through CNN Modeling
Though the possibilities for cinematography shots seem endless, there are a handful of shots that are commonly used for specific purposes. Seeing a certain shot onscreen may give us a hint about the type of scene or other narrative information. We can train convolutional neural networks (CNN) to recognize specific shots by hand-labeling individual frames from films and then conducting the modeling process. The CNN models in this project are completely original, trained on hand-labeled frame images, and no sort of transfer learning or tuned pre-trained models. Over 11,000 frame images were hand-labeled to train the first model, detailed below.

## Current Scope: Medium Close-Up Shots
Since the overall Moviegoer project is focused on identifying and analyzing two-character dialogue scene, the current scope of Shot Modeling is limited to a specific type of shot, the **Medium Close-Up**. Medium Close-Ups are the standard cinematography shot of the classically-shot two-character dialogue scene. They are typically a shot of the character from the torso-up, with little-to-no headroom (space between the top of their head and the frame). The character is framed to the left- or right-of-center; usually a character takes one side (e.g. left-) and the other character's shot is mirrored (e.g. right-).

![mcu samples](/readme_images/mcu.png "mcu samples")

## Project Links
- [Presentation for CNN Image Classifier](https://docs.google.com/presentation/d/1JytHUAu_NN734GOuSn8xJDm-2z6jJcKcaj_O8DvddRw/edit?usp=sharing)

## Repository Files
The repository contains the following files. The Modeling files, when read in order, provide a start-to-finish view of modeling for Medium Close-Ups, from data preparation all the way to model layer visualization.

### Modeling Files
1. *mcu_data_management.ipynb* - contains functions for controlling the Test/Train splits, with light EDA/visualizations
2. *mcu_baseline_creation.ipynb* - evaluates 3 CNN model designs and 3 data configurations to decide the basic baseline model
3. *mcu_basic_tuning.ipynb* - trains and evaluates variations of the baseline model, with a single parameter or attribute tuned, to see how that affects model performance
4. *mcu_modeling.ipynb* - using the lessons learned in *basic_tuning*, iteratively modifies and trains sucessive models until no more improvements are possible
5. *mcu_classification_evaluation.ipynb* - contains commentary and allows for visualizations of False Positives, False Negatives, etc., as well as individual activation layers

### Support Files
- *modeling_metrics_io.py* - functions for generating accuracy/loss visualizations and other metrics for evaluating each model

# MCU Modeling
## Data Understanding and Labeling
### Data Extraction
Frames (akin to screenshots) were extracted from 40 films: one frame every 8 seconds. This produced 675-900 frames per film, including frames from the beginning vanity logos to the ending credits. Each frame file is actually quite small, ranging from 50-80 kB, with an image size around 860x360. Though most films fit this cinema standard 2.39:1 ratio, some were wider or narrower. Still, no film deviated from this widescreen format enough to warrant resizing image files.

### Data Classification
11,432 frames were labeled by hand, into the two categories **MCU** (3,023) and **non-MCU** (8,409). Each frame was visually inspected and categorized. Though it seems like a daunting task, an MCU is immediately recognizable to any experienced cinematographer, especially if it's part of a two-character dialogue scene, and/or as an over-the-shoulder shot.

However, it needs to be emphasized that hand-labeling is more of an art than a science. Leeway was skewed toward labeling as frames as MCUs, such as during non-dialogue scenes but the shot happens to be an MCU, or when it's a toss-up between MCU and non-MCU but during a dialogue scene. False Positives are tolerable, in the interest of increasing the MCU Recall rate (more on Recall later).

![class imbalance](/readme_images/imbalance.png "class imbalance")

### Frame Variety
Only 200-400 frames were labeled per film, before moving onto the next film. Exposing the model to as many films as possible will prepare it to "watch" future, unseen films. Every film has a unique look, starting with the medium on which it was shot (graininess of 35mm film vs. the pristine look of digital CMOS sensors). Film selection also attempted to maximize variety in scene lighting (well-lit comedies vs. moody dramas), actors (non-white and child actors), and production value (slick blockbusters vs. naturalistic indies).

![frames per movie](/readme_images/frames.png "frames per movie")

![percentage of frames per movie](/readme_images/framepercentages.png "percentage of mcu frames per movie")

## Data Preparation
### Populating Test and Train
The *Data Management* notebook contains functions allowing for the automated population of the Test and Train folders, for use in modeling. There are two data configurations:
1. Mixed Frames - This shuffles frames from all 40 films into a random split of 80% Train and 20% Test. This is the primary configuration used for preliminary training, as it allows for the most variety in frames for training. 
2. Unwatched Frames - This randomly splits the films into 32 Train and 8 Test movies; each film's frames will only be in Train OR Test. This simulates the testing of the model on films it hasn't watched before, but at this early stage of training, the 25% increase in frame variety is more important.

During modeling, `ImageDataGenerator()` provided innate support for creating a Validation set of 20% of the Train data.

### Frame Input
`ImageDataGenerator.flow_from_directory()` was used to vectorize image data. For computational resource purposes, frames were resized to 128x128, and converted to grayscale. 128x128 is actually the minimum size necessary for a 6x6 CNN layer.

### Class Imbalance, and Abstaining from Data Augmentation
There was a strong class imbalance: after labeling, MCU frames only made **26%** of frames. This caused serious issues during preliminary modeling, where nearly every Test frame was predicted to be a non-MCU. The custom Test/Train population functions of *Data Management* notebook allow for a parameter `imbalance_removal`, which allows a certain percentage of non-MCU frames to be removed from the Training set. Various percentages (20%, 40%, 60%) were tested in the *Baseline Creation* notebook; 60% was ultimately chosen as it provided the best results (and provided an almost-equal distribution of mcu vs. non-MCU frames).

Keras' `ImageDataGenerator()` provides excellent options for data augmentation, to increase the variety of data, especially for an underrepresented class. After careful consideration, I declined to use any of these. A parameter-by-parameter rejection can be found in *baseline_creation*.

## Modeling
There were three phases of modeling, each documented in a separate notebook:
1. *mcu_baseline_creation* - This is where 9 models were evaluated to choose a basic model. 3 CNN designs (6x6, 5x5, 4x4) were evaluated against 3 datasets (60%, 40%, or 20% of non-MCUs removed to mitigate class imbalance). Ultimately, a 6x6 CNN trained on the dataset with 60% of non-MCUs removed was chosen as the final configuration.
2. *mcu_basic_tuning* - With the baseline model chosen, many new models were trained, each a copy of the baseline model, but with a single parameter tuned. By evaluating each parameter's impact on the MCU Recall and False Positive rate, we can decide which parameters were worth incorporating during final tuning. These parameters included number of units/filters per layer, or the addition of regularization/normalization layers.
3. *mcu_modeling* - A baseline model was chosen, and iteratively tuned. The baseline model was successively tuned using the lessons learned from *basic_tuning*.

This is the final design and summary of the final model.

![model](/readme_images/model.png "model")

![summary](/readme_images/summary.png "summary")

## Metrics and Evaluation
Two caveats need to be mentioned about metrics:
1. Since the MCU classifier is just one "confirmation" of the overall co-reinforcing dialogue scene identification goal, we are prioritizing Recall, allowing for a few False Positives to slip through. This is because both "confirmations" need to be positive to identify a dialogue scene. So Recall was the most important metric, with a trade-off of the False Positive Rate (non-MCUs predicted to be MCUs).
2. Because of the large class imbalance in the Test set, accuracy is heavily skewed in terms of non-MCU predictions.

The final model has a Recall of 81% and False Positive Rate of 30%. This is acceptable, for now.

![results](/readme_images/results.png "results")

## Model Classification Visualization
With a saved tuned model, we can generate predictions on the Test set, and display samples of True Positives, True Negatives, False Positives, and False Negatives. This type of visualization helps to provide clarity to the decisions the model is making, and also reminds us of the context of this project: we're working with images not just arrays of 1s and 0s. And since data is hand-labeled, these False Positive/Negative "disagreements" help provide clarity for future labeling. 

### True Positives
![true positives](/readme_images/truepos.png "true positives")
### True Negatives
![true negatives](/readme_images/trueneg.png "true negatives")
### False Positives
![false positives](/readme_images/falsepos.png "false positives")
### False Negatives
![false negatives](/readme_images/falseneg.png "fales negatives")

### First Activation Layer
We can also see how the various activation layers visualize an example frame. At the input layer, 64 filters visualize the 128x128 image in various ways. We can clearly see the outline of the character.

![first activation layer](/readme_images/firstact.png "first activation layer")

### Last Activation Layer
At the lowest activation layer (and not its pooling counterpart), 256 filters are looking at a 9x9 image. These images don't mean anything to us humans.

![last activation layer](/readme_images/lastact.png "last activation layer")

# Future Improvements
### MCU Classifier
The MCU classifier could use improvement, both in terms of more data and perhaps a more stringent hand-labelling process. False Positives and False Negatives could be scrutinized to see what went wrong. Perhaps if the data were more consistent, (Dropout) Regularization and (Batch) Normalization would have a more positive impact on the model; neither of these were included in the model design because of poor performance.

### Two Shots
We can also train a model to recognize another type of shot, the *two shot*. These are shots with both characters in frame, usually to establish their spatial presence (sitting together on a bench, standing face-to-face, etc.). These types of shots usually precede two-character dialogue scenes, and this identification could help improve the scene partitioning algorithm.
