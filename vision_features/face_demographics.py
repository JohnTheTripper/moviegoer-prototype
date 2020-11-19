import sys
sys.path.append('../unifying_features')
sys.path.append('../data_serialization')
from serialization_preprocessing_io import read_pickle
from scene_identification_io import generate_scenes
from character_identification_io import generate_characters
import os
from deepface import DeepFace

"""This exists in a separate .py file because it may not run in a Jupyter notebook"""


def get_biggest_face_frame(character_face_clusters, face_df):
    """
    returns the frame with the biggest face size of a given character
    useful for finding a frame to look up facial demographics
    """
    x = 0
    biggest_face_size = 0
    biggest_face_list_index = 0
    for face_size in face_df[face_df['p_face_cluster'].isin(character_face_clusters)].face_sizes:
        if face_size[0] > biggest_face_size:
            biggest_face_size = face_size[0]
            biggest_face_list_index = x
        x += 1

    biggest_face_frame = face_df[face_df['p_face_cluster'].isin(character_face_clusters)].iloc[
        biggest_face_list_index].name

    return biggest_face_frame


def display_character_demographics(film, chosen_character_index):
    """
    prints predicted age, race, and emotion for a given character
    """
    srt_df, subtitle_df, sentence_df, vision_df, face_df = read_pickle(film)
    scene_dictionaries = generate_scenes(vision_df, face_df, substantial_minimum=4, anchor_search=8)
    character_dictionaries = generate_characters(scene_dictionaries)

    chosen_character = character_dictionaries[chosen_character_index]
    character_face_clusters = chosen_character['face_clusters']

    frame_number = get_biggest_face_frame(character_face_clusters, face_df)
    frame_folder = os.path.join('../frame_per_second', film)
    img_path = frame_folder + '/' + film + '_frame_' + str(frame_number) + '.jpg'

    obj = DeepFace.analyze(img_path, actions=['age', 'gender', 'race'])
    print(round(obj['age']), 'years old')
    print(obj['dominant_race'])
    print(obj['gender'])


film = 'plus_one_2019'
chosen_character_index = 1
display_character_demographics(film, chosen_character_index)
