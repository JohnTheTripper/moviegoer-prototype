import sys
sys.path.append('../subtitle_features')
from subtitle_dataframes_io import *
from subtitle_auxiliary_io import *
sys.path.append('../vision_features')
from vision_dataframes_io import *
#sys.path.append('../audio_features')
#from audio_dataframes_io import *
from time_reference_io import *
from collections import Counter


def get_primary_chars(sentence_df, subtitle_df, nlp):
    """
    returns a potential list of characters in the film
    looks for names that are mentioned in subtitle dialogue, as well as names of offscreen speakers
    """
    sentences = sentence_df['sentence'].tolist()
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


''' may be depreciated
def self_intro_average_encoding(movie_choice, char_name, sentence_df, subtitle_df):
    """
    returns a list of facial encodings, given a character name
    looks for self-introduction statements to find character names
    """
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
'''


def generate_character_clusters(scene_dictionaries, common_count=5):
    """
    returns a list of list of facial encodings, grouped by similar characters
    character faces are grouped together based in appearances in the same scene, as well as different scenes
    """
    # search all scenes for anchor face clusters, flatten, and then generate list of most common
    anchor_face_cluster_lists = []

    x = 1

    while x <= len(scene_dictionaries):
        scene_left_list = []

        scene_left_list.append(scene_dictionaries[x]['left_anchor_face_cluster'])

        for matching_left_face in scene_dictionaries[x]['matching_left_face_clusters']:
            scene_left_list.append(matching_left_face)

        scene_right_list = []

        scene_right_list.append(scene_dictionaries[x]['right_anchor_face_cluster'])

        for matching_right_face in scene_dictionaries[x]['matching_right_face_clusters']:
            scene_right_list.append(matching_right_face)

        anchor_face_cluster_lists.append(scene_left_list)
        anchor_face_cluster_lists.append(scene_right_list)

        x += 1

    flattened_anchor_list = []
    for anchor_list in anchor_face_cluster_lists:
        for anchor in anchor_list:
            flattened_anchor_list.append(anchor)

    anchor_character_counter = Counter(flattened_anchor_list)
    most_common_anchors = []

    for anchor in anchor_character_counter.most_common(common_count):
        most_common_anchors.append(anchor[0])

    # input most common anchors, output character cluster list
    character_cluster_list = []

    for character in most_common_anchors:
        matching_clusters = [character]
        search_clusters = []
        search_continues = 1

        # search face clusters of all scenes to find matching clusters, then repeat search until no more found
        while search_continues == 1:
            if search_clusters:
                matching_clusters = search_clusters.copy()
            for anchor_face_clusters in anchor_face_cluster_lists:
                for matching_cluster in matching_clusters:
                    if matching_cluster in anchor_face_clusters:
                        for anchor_face in anchor_face_clusters:
                            search_clusters.append(anchor_face)
            search_clusters = list(set(search_clusters))

            if sorted(matching_clusters) == sorted(search_clusters):
                search_continues = 0
                character_cluster_list.append(sorted(matching_clusters))

    # remove duplicates
    unique_character_clusters = []
    for character_clusters in character_cluster_list:
        if character_clusters not in unique_character_clusters:
            unique_character_clusters.append(character_clusters)

    return unique_character_clusters


def generate_characters(scene_dictionaries):
    """
    returns a dictionary of characters, based on scene_dictionaries
    scene_dictionaries is used to identify scenes in the film, which are then used to cluster/group facial encodings
    """
    character_clusters = generate_character_clusters(scene_dictionaries)
    character_dictionary_list = []

    y = 1
    for single_character_face_clusters in character_clusters:

        character_scenes = []

        x = 1
        while x <= len(scene_dictionaries):
            for cluster in single_character_face_clusters:
                if cluster == scene_dictionaries[x]['left_anchor_face_cluster'] or cluster in scene_dictionaries[x][
                    'matching_left_face_clusters'] or cluster == scene_dictionaries[x][
                    'right_anchor_face_cluster'] or cluster in scene_dictionaries[x]['matching_right_face_clusters']:
                    character_scenes.append(x)
            x += 1

        character_scenes = sorted(list(set(character_scenes)))

        character_dict = {'character_id': y,
                          'face_clusters': single_character_face_clusters,
                          'scenes_present': character_scenes}

        character_dictionary_list.append(character_dict)
        y += 1

    character_dictionaries = {}

    z = 1
    for character_dict in character_dictionary_list:
        character_dictionaries[z] = character_dict
        z += 1

    return character_dictionaries
