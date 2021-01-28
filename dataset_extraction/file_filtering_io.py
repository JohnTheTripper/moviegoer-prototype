import os
from shutil import copyfile
import pandas as pd
import sys
sys.path.append('../data_serialization')
from serialization_preprocessing_io import read_pickle


def filter_primary_character(film):
    srt_df, subtitle_df, sentence_df, vision_df, face_df = read_pickle(film)
    prim_char_indices = face_df[face_df['prim_char_flag'] == 1].index.tolist()
    prim_large_indices = []

    for index in prim_char_indices:
        if 20 > face_df.loc[index].face_sizes[0] > 4:
            prim_large_indices.append(index)

    negative_folder = '../filtered_data/primary_character/negative/'
    positive_folder = '../filtered_data/primary_character/positive/'

    for prim_char_frame in prim_large_indices:
        source_img_path = '../frame_per_second/' + film + '/' + film + '_frame_' + str(prim_char_frame) + '.jpg'
        dest_img_path = os.path.join(positive_folder + film + '_frame_' + str(prim_char_frame) + '.jpg')
        copyfile(source_img_path, dest_img_path)

    neg_prim_large_indices = []
    for frame in range(1, len(face_df) + 1):
        if frame not in prim_large_indices:
            neg_prim_large_indices.append(frame)

    for neg_prim_char_frame in neg_prim_large_indices:
        source_img_path = '../frame_per_second/' + film + '/' + film + '_frame_' + str(neg_prim_char_frame) + '.jpg'
        dest_img_path = os.path.join(negative_folder + film + '_frame_' + str(neg_prim_char_frame) + '.jpg')
        copyfile(source_img_path, dest_img_path)

    print('Moved', len(prim_large_indices), 'Positive frames and', len(neg_prim_large_indices), 'Negative frames.')