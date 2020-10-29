import os
import sys
sys.path.append('../venv/lib/python3.6/site-packages')
sys.path.append('../vision_features')
from vision_dataframes_io import *
from faces_io import get_primary_char_emotion
from data_preprocessing_io import read_pickle
import pandas as pd

film = 'second_act_2018'
srt_df, subtitle_df, sentence_df, vision_df, face_df = read_pickle(film)

#face_df = face_df[100:110] # test

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

emotion_df = pd.DataFrame(list(zip(frame_list, p_emotions_visual)), columns=['frame', 'p_visual_emotion'])
emotion_df = emotion_df.set_index('frame')

df_object_directory = '../dataframe_objects/'
film_directory = os.path.join(df_object_directory, film)

emotion_df.to_pickle(os.path.join(film_directory, 'emotion_df.pkl'))
