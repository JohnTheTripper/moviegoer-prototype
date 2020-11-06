import os
import sys
sys.path.append('../venv/lib/python3.6/site-packages')
sys.path.append('../vision_features')
from vision_dataframes_io import *
from deepface import DeepFace
import pandas as pd


def get_primary_char_emotion(film, frame_number):
    frame_folder = os.path.join('../frame_per_second', film)
    img_path = frame_folder + '/' + film + '_frame_' + str(frame_number) + '.jpg'

    obj = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=False)
    return obj["dominant_emotion"]


film = 'plus_one_2019'
serialized_object_directory = '../serialized_objects/'
film_directory = os.path.join(serialized_object_directory, film)

face_df = pd.read_pickle(os.path.join(film_directory, 'face_df_pre_emotion.pkl'))

prim_char_flags = face_df.prim_char_flag.tolist()
frame_list = face_df.index.tolist()

p_emotions_visual = []
x = 0

for flag in prim_char_flags:
    if flag == 1:
        p_emotions_visual.append(get_primary_char_emotion(film, frame_list[x]))
    elif flag == 0:
        p_emotions_visual.append(None)
    x += 1

face_df['p_emotion'] = p_emotions_visual
face_df.to_pickle(os.path.join(film_directory, 'face_df.pkl'))


