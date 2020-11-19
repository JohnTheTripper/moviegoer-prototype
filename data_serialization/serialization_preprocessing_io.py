import os
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def read_pickle(film):
    """
    returns dataframes, read from serialized pickle file
    """
    df_object_directory = '../serialized_objects/'
    film_directory = os.path.join(df_object_directory, film)
    srt_df = pd.read_pickle(os.path.join(film_directory, 'srt_df.pkl'))
    subtitle_df = pd.read_pickle(os.path.join(film_directory, 'subtitle_df.pkl'))
    sentence_df = pd.read_pickle(os.path.join(film_directory, 'sentence_df.pkl'))
    vision_df = pd.read_pickle(os.path.join(film_directory, 'vision_df.pkl'))
    face_df = pd.read_pickle(os.path.join(film_directory, 'face_df.pkl'))

    return srt_df, subtitle_df, sentence_df, vision_df, face_df


def cluster_primary_faces(face_df):
    """
    returns face_df with primary faces clustered
    """
    prim_char_flags = face_df.prim_char_flag.tolist()
    face_encodings = face_df.face_encodings.tolist()

    prim_face_encodings = []
    x = 0
    for encodings in face_encodings:
        if prim_char_flags[x] == 1:
            prim_face_encodings.append(encodings[0])
        x += 1

    prim_face_encodings_np = np.array(prim_face_encodings)

    hac = AgglomerativeClustering(n_clusters=None, distance_threshold=2).fit(prim_face_encodings_np)
    hac_labels = hac.labels_

    primary_face_clusters = []

    x = 0

    for flag in prim_char_flags:
        if flag == 1:
            primary_face_clusters.append(hac_labels[x])
            x += 1
        elif flag == 0:
            primary_face_clusters.append(None)

    face_df['p_face_cluster'] = primary_face_clusters

    return face_df


# shot cluster
def cluster_shots(film, vision_df):
    """
    returns vision_df with shots clustered, and each assigned an incremental shot_id
    """
    df_object_directory = '../serialized_objects/'
    film_directory = os.path.join(df_object_directory, film)
    vgg16_feature_list_np = np.load(os.path.join(film_directory, 'vgg16_features.npy'))

    hac = AgglomerativeClustering(n_clusters=None, distance_threshold=2750).fit(vgg16_feature_list_np)
    hac_labels = hac.labels_

    vision_df['shot_cluster'] = hac_labels

    shot_id = 0
    shot_id_list = []
    prev_frame = 1000

    for cluster in hac_labels:
        if cluster != prev_frame and prev_frame != 1000:
            shot_id += 1
        shot_id_list.append(shot_id)
        prev_frame = cluster

    vision_df['shot_id'] = shot_id_list
    vision_df = vision_df.apply(pd.to_numeric, errors='ignore', downcast='integer')

    return vision_df
