# Data Serialization
## Serializing Film Data
These files contain the end-to-end workflow of creating dataframes from the original movie file. In essence, this is where a movie is being "watched". Dataframes are saved as pickle files for later use.

## Repository Files
The directory contains the following files. The serialization files must be run in order.
### Serialization Workflow
1. *frame_extract.py* - extracts individual frames from the movie file, creating one jpg frame per second of film
2. *base_dataframes.py* - creates base dataframes
3. *shot_recogntion.py* - uses the VGG16 image model to create an array of features from each frame for the purposes of identifying "shots', essentially clusters of similar frames; this is run in a Docker container for GPU processing
4. *vision_cluster.py* - clusters face encoding data, as well as the above shots
5. *emotion_recogntion.py* - identifies facial emotion of frames with primary characters
### Support Files
- *serialization_preprocessing_io.py* - functions for the above files, as well as for loading dataframes from pickle files
