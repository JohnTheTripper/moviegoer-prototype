import sys
sys.path.append('../subtitle_features')
from subtitle_dataframes_io import *
from subtitle_auxiliary_io import *
sys.path.append('../vision_features')
from vision_dataframes_io import *
sys.path.append('../audio_features')
from audio_dataframes_io import *
from time_reference_io import *


def get_primary_chars(sentences, subtitle_df, nlp):
    chars_sub_mentions = character_subtitle_mentions(sentences, nlp)
    chars_offscreen_speakers = character_offscreen_speakers(subtitle_df)

    characters = []

    for character in chars_sub_mentions:
        if character[1] >= 10:
            characters.append(character[0].lower())

    for character in chars_offscreen_speakers:
        if character[1] >= 5:
            characters.append(character[0].lower())

    characters = list(set(characters))

    return characters


def self_intro_average_encoding(movie_choice, char_name, sentence_df, subtitle_df):
    # find subtitle indices
    intro_indices = sentence_df[
        sentence_df.self_intro.str.contains(char_name, na=False, case=False)].subtitle_indices.values
    flattened_indices = np.concatenate(intro_indices).ravel()

    # calculate frames from start and end times
    mid_time_frames = []
    for sub_index in flattened_indices:
        mid_time = subtitle_mid_time(subtitle_df.iloc[sub_index].start_time, subtitle_df.iloc[sub_index].end_time)
        mid_time_frames.append(time_to_frame(mid_time))

    mid_time_frames

    # check each frame for faces
    char_encodings = []

    for frame_number in mid_time_frames:
        frame = load_frame(movie_choice, frame_number)

        locations = face_recognition.face_locations(frame, number_of_times_to_upsample=2)
        encodings = face_recognition.face_encodings(frame, locations)
        if encodings:
            char_encodings.append(encodings)

    # flatten list of face encodings
    char_flat_scratch = char_encodings.copy()
    char_encodings_flat = []
    for x in char_flat_scratch:
        for y in x:
            char_encodings_flat.append(y)

    # get majority similarities
    char_matches = []
    x = 0
    face_candidates = len(char_encodings_flat)

    while x < face_candidates:
        list_compare = char_encodings_flat.copy()
        char_compare = list_compare[x]
        del list_compare[x]
        if sum(face_recognition.compare_faces(list_compare, char_compare, tolerance=1)) >= (face_candidates - 1) / 2:
            char_matches.append(char_compare)
        x += 1

    char_matches = np.array(char_matches)

    # create average encoding
    if len(char_matches) != 0:
        average_encoding = np.average(char_matches, axis=0)
    else:
        average_encoding = None

    return average_encoding