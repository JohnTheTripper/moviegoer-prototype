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

## Usage
Processing movies is a multi-step process. The input is a movie file and its subtitle file, the output is five dataframes (stored as .pkl pickle objects).

### Preparation
See the main repo Readme file for the Tech Stack; required Python libraries are listed here.

Movie files (.mkv) should be placed in a folder called input_videos/ and their subtitle files (.srt) should be placed in a folder called subtitles/. Both of these directories are in the project root.

### Processing
Each of the processing files are run in sequence; each of the steps takes the previous step's as output as input. These intermediary files aren't deleted.
1. *frame_extract.py* - manually enter the film name; this creates the directory frame_per_second/FILM_NAME/ and populates image frames
2. *base_dataframes.py* - manually enter the film name and number of frames; this creates the directory serialized_objects/FILM_NAME/ and populates it with dataframe pickle objects
3. *shot_recogntion.py* - manually enter the film name and number of frames; this creates a (large) NumPy file with image vectorizations
4. *vision_cluster.py* - manually enter the film name; this adds clustering for faces and frames into their respective dataframes
5. *emotion_recogntion.py* - manually enter the film name; this adds emotion to the face dataframe

### Loading the Pickle Files
The dataframes can be loaded with the following code:

    from serialization_preprocessing_io import *
    film = 'booksmart_2019'
    srt_df, subtitle_df, sentence_df, vision_df, face_df = read_pickle(film)
