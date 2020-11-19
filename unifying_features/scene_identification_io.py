import face_recognition
import numpy as np
import pandas as pd
from time_reference_io import *


def find_alternating_clusters(vision_df):
    """
    returns two lists: alternating pairs of shot clusters, and the corresponding start/end shot_ids
    the heart of the scene identification algorithm, finds shots which follow an A/B/A/B pattern
    these shots are known as "anchors"
    useful for finding dialogue scenes, which are overwhelmingly filmed in the shot/reverse-shot style
    """
    shot_id_list = vision_df.shot_id.tolist()
    shot_clusters = vision_df.shot_cluster.tolist()
    frame_choice = range(1, (len(vision_df) + 1))

    # to check for an A/B/A/B pattern, we must store the previous three clusters in memory
    prev_clust_1 = 1001
    prev_clust_2 = 1002
    prev_clust_3 = 1003
    prev_shot_id = -1
    alternate_a_list = []
    alternate_b_list = []
    pair_shot_ids = []
    pair_found = 0

    # zip our various lists into a usable data structure
    for frame_file, cluster, shot_id in zip(frame_choice, shot_clusters, shot_id_list):

        # we use prev_shot_id to identify when there's a new shot (when the cluster value changes)
        # when iterating through each frame, look for an A/B/A/B pattern, and save the clusters of any patterns
        if shot_id != prev_shot_id:
            if cluster == prev_clust_2 and prev_clust_1 == prev_clust_3:
                if pair_found == 0:
                    alternate_a_list.append(
                        min(cluster, prev_clust_1))  # min and max are used to avoid duplicates of (1, 2), (2, 1)
                    alternate_b_list.append(max(cluster, prev_clust_1))
                    beginning_shot = shot_id - 3
                pair_found = 1
            else:
                if pair_found == 1:
                    ending_shot = shot_id - 1
                    pair_shot_ids.append([beginning_shot, ending_shot])
                pair_found = 0

            # every time there's a new shot, we update the cluster memory
            prev_shot_id = shot_id
            prev_clust_3 = prev_clust_2
            prev_clust_2 = prev_clust_1
            prev_clust_1 = cluster

        # the below print can be used for troubleshooting and visualizing the memory state at each frame
        # print(frame_file, '\t', mcu_flag, '\t', cluster,'\t', shot_id, '\t', prev_shot_id, '\t', prev_clust_1, '\t', prev_clust_2, '\t', prev_clust_3, '\tend')

    # save non-unique alternating pairs, because these must line up with pair_shot_ids
    alternating_pairs = []

    for a, b, in zip(alternate_a_list, alternate_b_list):
        alternating_pairs.append([int(a), int(b)])

    return alternating_pairs, pair_shot_ids


def filter_substantial_shot_pairs(alternating_pairs, pair_shot_ids, threshold=6):
    """
    returns an updated list of alternating shot pairs, by checking for a minimum threshold of A/B/A/B altneration
    """
    substantial_pair_shot_ids = []
    substantial_anchor_shot_clusters = []

    for anchor_pair, shot_id_pair in zip(alternating_pairs, pair_shot_ids):
        if shot_id_pair[1] - shot_id_pair[0] > threshold:
            substantial_pair_shot_ids.append(shot_id_pair)
            substantial_anchor_shot_clusters.append(anchor_pair)
    return substantial_anchor_shot_clusters, substantial_pair_shot_ids


def left_face_clusters(alternation_face_df):
    """
    returns the primary face cluster for the left character in a scene, and a list of additional matching face clusters
    primary face cluster is the most prevalent, and matching clusters are the rest
    """
    matching_left_clusters = []

    left_value_counts = alternation_face_df[(alternation_face_df['p_center_alignment'] == 'left') & (alternation_face_df['faces_found'] == 1)].p_face_cluster.value_counts(normalize=True)

    if len(alternation_face_df[(alternation_face_df['p_center_alignment'] == 'left') & (alternation_face_df['faces_found'] == 1)]) > 2:
        if left_value_counts.values[0] >= .5:
            left_anchor_face_cluster = left_value_counts.index.values[0]
            left_anchor_face_encoding = np.average(alternation_face_df.loc[(alternation_face_df['p_center_alignment'] == 'left') & (alternation_face_df['p_face_cluster'] == left_anchor_face_cluster) & (alternation_face_df['faces_found'] == 1)].face_encodings.tolist(), axis=0)[0]
            for candidate in left_value_counts.index.values[1:]:
                left_cluster_candidate = np.average(alternation_face_df.loc[(alternation_face_df['p_center_alignment'] == 'left') & (alternation_face_df['p_face_cluster'] == candidate) & (alternation_face_df['faces_found'] == 1)].face_encodings.tolist(), axis=0)[0]
                if face_recognition.compare_faces([left_anchor_face_encoding], left_cluster_candidate)[0] == True:
                    matching_left_clusters.append(candidate)
            return left_anchor_face_cluster, matching_left_clusters
        else:
            return None, None
    else:
        return None, None


def right_face_clusters(alternation_face_df):
    """
    returns the primary face cluster for the right character in a scene, and a list of additional matching face clusters
    primary face cluster is the most prevalent, and matching clusters are the rest
    """
    matching_right_clusters = []

    right_value_counts = alternation_face_df[(alternation_face_df['p_center_alignment'] == 'right') & (
                alternation_face_df['faces_found'] == 1)].p_face_cluster.value_counts(normalize=True)

    if len(alternation_face_df[
               (alternation_face_df['p_center_alignment'] == 'right') & (alternation_face_df['faces_found'] == 1)]) > 2:
        if right_value_counts.values[0] >= .5:
            right_anchor_face_cluster = right_value_counts.index.values[0]
            right_anchor_face_encoding = np.average(alternation_face_df.loc[
                                                        (alternation_face_df['p_center_alignment'] == 'right') & (
                                                                    alternation_face_df[
                                                                        'p_face_cluster'] == right_anchor_face_cluster) & (
                                                                    alternation_face_df[
                                                                        'faces_found'] == 1)].face_encodings.tolist(),
                                                    axis=0)[0]
            for candidate in right_value_counts.index.values[1:]:
                right_cluster_candidate = np.average(alternation_face_df.loc[
                                                         (alternation_face_df['p_center_alignment'] == 'right') & (
                                                                     alternation_face_df[
                                                                         'p_face_cluster'] == candidate) & (
                                                                     alternation_face_df[
                                                                         'faces_found'] == 1)].face_encodings.tolist(),
                                                     axis=0)[0]
                if face_recognition.compare_faces([right_anchor_face_encoding], right_cluster_candidate)[0] == True:
                    matching_right_clusters.append(candidate)
            return right_anchor_face_cluster, matching_right_clusters
        else:
            return None, None
    else:
        return None, None


def find_alternating_scenes(substantial_anchor_shot_clusters, substantial_pair_shot_ids, vision_df, face_df):
    """
    returns a list of dialogue scenes, by searching groups of alternating shots to see if they actually contain faces
    used to confirm alternating shots are actually dialogue scenes
    """
    alternating_scene_frame_pairs = []
    alternating_scene_anchor_pairs = []

    for pair, anchors in zip(substantial_pair_shot_ids, substantial_anchor_shot_clusters):
        first_frame = vision_df[vision_df['shot_id'].isin([pair[0], pair[1]])][:1].index[0]
        last_frame = vision_df[vision_df['shot_id'].isin([pair[0], pair[1]])][-1:].index[0]
        alternation_face_df = face_df.copy()[first_frame - 1:last_frame]
        left_right_percentage = len(
            alternation_face_df[alternation_face_df['p_center_alignment'].isin(['left', 'right'])]) / len(
            alternation_face_df) * 100
        prim_face_percentage = len(alternation_face_df[alternation_face_df['prim_char_flag'] == 1]) / len(
            alternation_face_df) * 100
        left_anchor_face_cluster, matching_left_clusters = left_face_clusters(alternation_face_df)
        right_anchor_face_cluster, matching_right_clusters = right_face_clusters(alternation_face_df)
        if left_anchor_face_cluster and right_anchor_face_cluster:
            if prim_face_percentage >= .8:
                alternating_scene_frame_pairs.append([first_frame, last_frame])
                alternating_scene_anchor_pairs.append(anchors)
        else:
            pass

    return alternating_scene_frame_pairs, alternating_scene_anchor_pairs


def expand_scene(alternating_scene_frame_pair, vision_df, anchor_search_threshold=6):
    """
    returns expanded scenes, by defining their start and end frames
    expands scenes by identifying cutaway shots
    scenes are expanded by checking for cutaways before or after the original scene boundary (defined by the anchors)
    """
    anchor_shot_cluster_pair = list(vision_df[alternating_scene_frame_pair[0] - 1:alternating_scene_frame_pair[1]].shot_cluster.unique())
    anchor_shot_id_pair = [vision_df[alternating_scene_frame_pair[0] - 1:alternating_scene_frame_pair[1]].shot_id.min(),
                           vision_df[alternating_scene_frame_pair[0] - 1:alternating_scene_frame_pair[1]].shot_id.max()]
    first_anchor_frame = vision_df[(vision_df['shot_id'] > anchor_shot_id_pair[0] - anchor_search_threshold) & (vision_df['shot_id'] < anchor_shot_id_pair[1] + anchor_search_threshold) & (vision_df['shot_cluster'].isin(anchor_shot_cluster_pair))].index.min()
    last_anchor_frame = vision_df[
        (vision_df['shot_id'] > anchor_shot_id_pair[0] - anchor_search_threshold) & (vision_df['shot_id'] < anchor_shot_id_pair[1] + anchor_search_threshold) & (
            vision_df['shot_cluster'].isin(anchor_shot_cluster_pair))].index.max()
    cutaways = vision_df[first_anchor_frame - 1:last_anchor_frame].shot_cluster.unique()
    cutaways = cutaways[cutaways != anchor_shot_cluster_pair[0]] # remove the Speaker A and Speaker B clusters from this list
    cutaways = cutaways[cutaways != anchor_shot_cluster_pair[1]]

    scene_start_frame = first_anchor_frame
    min_flag = 0

    while min_flag == 0:
        try:
            if vision_df.loc[scene_start_frame - 1].shot_cluster in cutaways:
                scene_start_frame -= 1
            else:
                min_flag = 1
        except TypeError:  # error if hitting the beginning of the frame list
            min_flag = 1

    scene_end_frame = last_anchor_frame
    max_flag = 0
    while max_flag == 0:
        try:
            if vision_df.loc[scene_start_frame - 1].shot_cluster in cutaways:
                scene_end_frame += 1
            else:
                max_flag = 1
        except TypeError:  # error if hitting the end of the frame list
            max_flag = 1

    expanded_scene_frame_pair = [scene_start_frame, scene_end_frame]

    return expanded_scene_frame_pair


def generate_scenes(vision_df, face_df, substantial_minimum=6, anchor_search=6):
    """
    returns a dictionary of scenes
    calculates and populates scene information, such as duration, characters, and cutaway shots
    """
    alternating_pairs, pair_shot_ids = find_alternating_clusters(vision_df)
    substantial_anchor_shot_clusters, substantial_pair_shot_ids  = filter_substantial_shot_pairs(alternating_pairs, pair_shot_ids, threshold=substantial_minimum)
    alternating_scene_frame_pairs, alternating_scene_anchor_pairs = find_alternating_scenes(substantial_anchor_shot_clusters, substantial_pair_shot_ids, vision_df, face_df)

    expanded_scene_frame_pairs = []
    for alternating_frame_pair in alternating_scene_frame_pairs:
        expanded_scene_frame_pairs.append(expand_scene(alternating_frame_pair, vision_df, anchor_search_threshold=anchor_search))

    x = 1
    scene_dictionary_list = []
    for expanded_frame_pair, scene_anchor_pair in zip(expanded_scene_frame_pairs, alternating_scene_anchor_pairs):
        first_frame = expanded_frame_pair[0]
        last_frame = expanded_frame_pair[1]
        scene_duration = last_frame + 1 - first_frame
        expanded_face_df = face_df.copy()[first_frame - 1:last_frame]
        expanded_vision_df = vision_df.copy()[first_frame - 1:last_frame]
        left_anchor_shot_cluster = expanded_vision_df[(expanded_face_df['p_center_alignment'] == 'left') & (expanded_vision_df.shot_cluster.isin(scene_anchor_pair))].shot_cluster.value_counts().index[0]
        left_anchor_face_cluster, matching_left_face_clusters = left_face_clusters(expanded_face_df)
        right_anchor_face_cluster, matching_right_face_clusters = right_face_clusters(expanded_face_df)
        right_anchor_shot_cluster = expanded_vision_df[(expanded_face_df['p_center_alignment'] == 'right') & (expanded_vision_df.shot_cluster.isin(scene_anchor_pair))].shot_cluster.value_counts().index[0]
        cutaway_shot_clusters = vision_df[first_frame - 1:last_frame].shot_cluster.unique()
        cutaway_shot_clusters = cutaway_shot_clusters[cutaway_shot_clusters != left_anchor_shot_cluster]
        cutaway_shot_clusters = list(cutaway_shot_clusters[cutaway_shot_clusters != right_anchor_shot_cluster])
        if left_anchor_face_cluster and right_anchor_face_cluster:
            scene_dict = {'scene_id': x,
                          'first_frame': first_frame,
                          'last_frame': last_frame,
                          'scene_duration': scene_duration,
                          'left_anchor_shot_cluster': left_anchor_shot_cluster,
                          'left_anchor_face_cluster': left_anchor_face_cluster,
                          'matching_left_face_clusters': matching_left_face_clusters,
                          'right_anchor_shot_cluster': right_anchor_shot_cluster,
                          'right_anchor_face_cluster': right_anchor_face_cluster,
                          'matching_right_face_clusters': matching_right_face_clusters,
                          'cutaway_shot_clusters': cutaway_shot_clusters}
            scene_dictionary_list.append(scene_dict)
            x += 1

        scene_dictionaries = {}
        x = 1
        for scene_dict in scene_dictionary_list:
            scene_dictionaries[x] = scene_dict
            x += 1

    return scene_dictionaries
