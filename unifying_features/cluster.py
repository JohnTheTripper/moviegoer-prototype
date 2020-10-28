import os
from data_preprocessing_io import *

film = 'once_upon_a_time_in_hollywood_2019'

srt_df, subtitle_df, sentence_df, vision_df, face_df = read_pickle(film)

face_df_clustered = cluster_primary_faces(face_df)
vision_df_clustered = cluster_shots(film, vision_df)

df_object_directory = '../dataframe_objects/'
film_directory = os.path.join(df_object_directory, film)

face_df_clustered.to_pickle(os.path.join(film_directory, 'face_df_clustered.pkl'))
vision_df_clustered.to_pickle(os.path.join(film_directory, 'vision_df_clustered.pkl'))
