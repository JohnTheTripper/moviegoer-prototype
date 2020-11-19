import os
import sys
from serialization_preprocessing_io import *
import pandas as pd
sys.path.append('../subtitle_features')
from subtitle_dataframes_io import *

"""
performs clustering of frames (from saved numpy file of VGG16 vectorizations) and faces (from pickled face_df dataframe)
to use, change film
"""

film = 'black_and_blue_2019'

'''change above parameters'''

serialized_object_directory = '../serialized_objects/'
film_directory = os.path.join(serialized_object_directory, film)
vision_df = pd.read_pickle(os.path.join(film_directory, 'vision_df_pre_cluster.pkl'))
face_df = pd.read_pickle(os.path.join(film_directory, 'face_df_pre_cluster.pkl'))

face_df_clustered = cluster_primary_faces(face_df)
vision_df_clustered = cluster_shots(film, vision_df)

face_df_clustered.to_pickle(os.path.join(film_directory, 'face_df_pre_emotion.pkl'))
vision_df_clustered.to_pickle(os.path.join(film_directory, 'vision_df.pkl'))

sentence_df = pd.read_pickle(os.path.join(film_directory, 'sentence_df_pre_cluster.pkl'))
subtitle_df = pd.read_pickle(os.path.join(film_directory, 'subtitle_df.pkl'))
sentence_df = generate_speakers(sentence_df, subtitle_df, face_df)
sentence_df.to_pickle(os.path.join(film_directory, 'sentence_df.pkl'))
