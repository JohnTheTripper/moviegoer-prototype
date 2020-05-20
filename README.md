# moviegoer

## Project Overview and MCU Scope
(Overview text)

(MCU text)

## Important Links
(Links)


## Repository Files
The repository contains the following files. The Modeling files, when read in order, provide a start-to-finish view of modeling, from data preparation all the way to model layer visualization.

### Modeling Files
1. *mcu_data_management.ipynb* - contains functions for controlling the Test/Train splits, with light EDA/visualizations
2. *mcu_baseline_creation.ipynb* - evaluates 3 CNN model designs and 3 data configurations to decide the basic baseline model
3. *mcu_basic_tuning.ipynb* - trains and evaluates variations of the baseline model, with a single parameter or attribute tuned, to see how that affects model performance
4. *mcu_modeling.ipynb* - using the lessons learned in *basic_tuning*, iteratively modifies and trains sucessive models until no more improvements are possible
5. *mcu_classification_evaluation.ipynb* - contains commentary and allows for visualizations of False Positives, False Negatives, etc., as well individual activation layers

### Support Files
- *metric_functions.py* - generates accuracy/loss visualizations and various metrics for evaluating each model
- *extract.py* - generates screenshots from movie files

## More about MCUs
(MCUs)

(Accuracy and False Positives)


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
There was a strong class imbalance. After labeling, MCU frames only made **26%** of frames. This caused serious issues during preliminary modeling, where every Test frame was predicted to be a non-MCU. The Test/Train population functions of *Data Management* notebook allow for a parameter *imbalance_removal*, which allows a certain percentage of non-MCU frames to be removed from the Training set. Various percentages (20%, 40%, 60%) were tested in the *Baseline Creation* notebook; 60% was ultimately chosen as it provided the best results (and provided an almost-equal distribution of mcu vs. non-MCU frames).

Keras' `ImageDataGenerator()` provides excellent options for data augmentation, to increase the variety of data, especially for an underrepresented class. After careful consideration, I declined to use any of these. A parameter-by-parameter rejection can be found in *baseline_creation*.

## Modeling
There were three phases of modeling, each documented in a separate notebook:
1. *baseline_creation* - This is where 9 models were evaluated to choose a basic model. 3 CNN designs (6x6, 5x5, 4x4) were evaluated against 3 datasets (60%, 40%, or 20% of non-MCUs removed to mitigate class imbalance). Ultimately, a 6x6 CNN trained on the dataset with 60% of non-MCUs removed was chosen as the final configuration.
2. *basic_tuning* - With the baseline model chosen, many new models were trained, each a copy of the baseline model, but with a single parameter tuned. By evaluating each parameter's impact on the MCU Recall and False Positive rate, we can decide which parameters were worth incorporating during final tuning. These parameters included number of units/filters per layer, or the addition of regularization/normalization layers.
3. *modeling* - A baseline model was chosen, and iteratively tuned. The baseline model was successively tuned using the lessons learned from *basic_tuning*.

This is the final design and summary of the final model.

(IMAGES)

## Model Evaluation and Visualization
(ADD NOTE ABOUT RECALL AND FALSE POSITIVES)

With a saved tuned model, we can generate predictions on the Test set, and display samples of True Positives, True Negatives, False Positives, and False Negatives. This type of visualization helps to provide clarity to the decisions the model is making, and also reminds us of the context of this project: we're working with images not just arrays of 1s and 0s. And since data is hand-labeled, these False Positive/Negative "disagreements" help provide clarity for future labeling. 

We can also see how the various activation layers visualize an example frame. At the input layer, 64 filters visualize the 128x128 image in various ways. We can clearly see the outline of the character.

(IMAGES)

At the lowest activation layer (and not its pooling counterpart), 256 filters are looking at a 9x9 image. These images don't mean anything to us humans.

(IMAGES)
