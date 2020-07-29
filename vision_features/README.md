# Vision Features
## Extracting visual features
We can extract many features to populate in the frame-level DataFrame. These can be used for other modules' purposes, such as scene boundary partitioning (for example, by detecting all-black frames between scenes) or persistent character identification. These features will also be useful in the feature, perhaps if we incorporate a neural network or other deep learning

## Current Scope: Frame-level features
Right now, only frame-level features are being detected. These are features that can be identified from a single frame, such as brightness or aspect ratio. This is in contrast to shot-level features, which rely on the analysis of multiple frames, like detecting movement (difference from frame to frame). This will avoid the need for a recurrent aspect of a neural network.

## Repository Files
The directory contains the following files.

### Feature Files
1. *chroma_luma.ipynb* - explores features related to color intensity (chroma) and overall brightness (luma)
2. *dimensions.ipynb* - explores features related to the image size, as well as identifies specific points-of-interest, like rule-of-thirds intersections
