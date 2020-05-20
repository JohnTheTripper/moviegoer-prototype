## Project Overview and Scope
### Moviegoer Project
The Moviegoer project has the lofty goal of unlocking the enormous wealth of emotional data within cinema, by turning films into structured data. Rather than create a single, all-encompassing model, the Moviegoer will be cobbled together iteratively, with various co-refinforcing modules based in transfer learning. There's been much research in the fields of emotional facial recognition, speech analysis, etc, and there are plenty of pre-trained models freely available. But all of these can't be applied to cinema until we can decode a movie's near-infinite possibilities into structured data.

### Current Scope: CNN Identification of Medium Close-Ups
Currently, the scope of this project is limited to identifying specific cinematography frames (akin to screenshots) as **MCUs, or medium close-ups**. (As the scope of the project expands, this Readme, and the Repository layout will change accordingly.) A convolutional neural network will be designed to (binary) classify film frames as MCUs, or non-MCUs.

The project's first large goal is the automatic identification of a film's two-character dialogue scenes, and this CNN image classification plays a key part in that identification. The two-character dialogue scene is the basic building block of most every film, allowing for plot advancement; in information theory parlance, this is where the "densest" information lies. These types of scenes are typically filmed using the medium close-up shot. More details on the MCU can be found below, but this is what our CNN is looking for:

![mcu samples](/images/mcu.png "mcu samples")

### Frames, not Videos
One final note about frames: this project will strictly be using frames (screenshots), as opposed to video snippets (multiple frames), as input data. This has a number of benefits: reducing computational complexity, removing the need for recurrent elements of neural networks, and more granular data. This, of course, requires some sort of external timestamping system to track where frames occur in the film.

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


## Data Understanding and Labeling
### A Stronger Definition of Medium Close-Ups
Medium close-ups are the standard cinematography shot of the classicly-shot two-character dialogue scene. They are typically a shot of the character from the torso-up, with little-to-no headroom (space between the top of their head and the frame). The character is framed to the left- or right-of-center; usually a character takes one side (e.g. left-) and the other character's shot is mirrored (e.g. right-).

### Data Extraction
Frames (akin to screenshots) were extracted from 40 films: one frame every 8 seconds. This produced 675-900 frames per film, including frames from the beginning vanity logos to the ending credits. Each frame file is actually quite small, ranging from 50-80 kB, with an image size around 860x360. Though most films fit this cinema standard 2.39:1 ratio, some were wider or narrower. Still, no film deviated from this widescreen format enough to warrant resizing image files.

### Data Classification
11,432 frames were labeled by hand, into the two categories **MCU** (3,023) and **non-MCU** (8,409). Each frame was visually inspected and categorized. Though it seems like a daunting task, an MCU is immediately recognizable to any experienced cinematographer, especially if it's part of a two-character dialogue scene, and/or as an over-the-shoulder shot.

However, it needs to be emphasized that hand-labeling is more of an art than a science. Leeway was skewed toward labeling as frames as MCUs, such as during non-dialogue scenes but the shot happens to be an MCU, or when it's a toss-up between MCU and non-MCU but during a dialogue scene. False Positives are tolerable, in the interest of increasing the MCU Recall rate (more on Recall later).

![class imbalance](/images/imbalance.png "class imbalance")

### Frame Variety
Only 200-400 frames were labeled per film, before moving on. Exposing the model to as many films as possible will prepare it to "watch" future, unseen films. Every film has a unique look, starting with the medium on which it was shot (graininess of 35mm film vs. the pristine look of digital CMOS sensors). Film selection also attempted to maximize variety in scene lighting (well-lit comedies vs. moody dramas), actors (non-white and child actors), and production value (slick blockbusters vs. naturalistic indies).

![frames per movie](/images/frames.png "frames per movie")

![frames per movie](/images/framepercentages.png "percentage of mcu frames per movie")

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
1. *baseline_creation* - This is where 9 models were evaluated to choose a basic model. 3 CNN designs (6x6, 5x5, 4x4) were evaluated against 3 datasets (60%, 40%, or 20% of non-MCUs removed to mitigate class imbalance). Ultimately, a 6x6 CNN trained on the dataset with 60% of non-MCUs removed was chosen as the final configuration.
2. *basic_tuning* - With the baseline model chosen, many new models were trained, each a copy of the baseline model, but with a single parameter tuned. By evaluating each parameter's impact on the MCU Recall and False Positive rate, we can decide which parameters were worth incorporating during final tuning. These parameters included number of units/filters per layer, or the addition of regularization/normalization layers.
3. *modeling* - A baseline model was chosen, and iteratively tuned. The baseline model was successively tuned using the lessons learned from *basic_tuning*.

This is the final design and summary of the final model.

![model](/images/model.png "model")

![summary](/images/summary.png "summary")

## Metrics and Evaluation
Two caveats need to be mentioned about metrics:
1. Since the MCU classifier is just one "confirmation" of the overall co-reinforcing dialogue scene identification goal, I am prioritizing Recall, allowing for a few False Positives to slip through. This is because both "confirmations" need to be positive to identify a dialogue scene. So Recall was the most important metric, with a trade-off of the False Positive Rate (non-MCUs predicted to be MCUs).
2. Because of the large class imbalance in the Test set, accuracy is heavily skewed in terms of non-MCU predictions.

The final model has a Recall of 81% and False Positive Rate of 30%. This is acceptable, for now.

![results](/images/results.png "results")

## Model Classification Visualization
With a saved tuned model, we can generate predictions on the Test set, and display samples of True Positives, True Negatives, False Positives, and False Negatives. This type of visualization helps to provide clarity to the decisions the model is making, and also reminds us of the context of this project: we're working with images not just arrays of 1s and 0s. And since data is hand-labeled, these False Positive/Negative "disagreements" help provide clarity for future labeling. 

### True Positives
![true positives](/images/truepos.png "true positives")
### True Negatives
![true negatives](/images/trueneg.png "true negatives")
### False Positives
![false positives](/images/falsepos.png "false positives")
### False Negatives
![false negatives](/images/falseneg.png "fales negatives")

### First Activation Layer
We can also see how the various activation layers visualize an example frame. At the input layer, 64 filters visualize the 128x128 image in various ways. We can clearly see the outline of the character.

![first activation layer](/images/firstact.png "first activation layer")

### Last Activation Layer
At the lowest activation layer (and not its pooling counterpart), 256 filters are looking at a 9x9 image. These images don't mean anything to us humans.

![last activation layer](/images/lastact.png "last activation layer")

## Future Improvements to the MCU classifier and Continuation of the Moviegoer Project
The MCU classifier could use improvement, both in terms of more data, and perhaps a more stringent hand-labelling process. False Positives and False Negatives could be scrutinized to see what went wrong. Perhaps if the data were more consistent, (Dropout) Regularization and (Batch) Normalization would have a more positive impact on the model; neither of these were included in the model design because of poor performance.

The next phase of the Moviegoer Project will the other half of the two-character dialogue scene identification: automated HAC clustering. These process will attempt to automatically identify two characters having a dialogue, and then using the CNN classifer to determine if the shot is an MCU. Confirmations from both halves will result in a positive identification of the two-character dialogue scene.
