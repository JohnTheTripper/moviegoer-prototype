import face_recognition
import numpy as np
import pandas as pd


def find_alternating_clusters(vision_df):
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


def filter_substantial_shot_pairs(pair_shot_ids, threshold=6):
    substantial_pairs = []
    for pair in pair_shot_ids:
        if pair[1] - pair[0] > threshold:
            substantial_pairs.append(pair)

    return substantial_pairs


def left_face_clusters(alternation_face_df):
    matching_left_clusters = []

    left_value_counts = alternation_face_df[
        alternation_face_df['p_center_alignment'] == 'left'].p_face_cluster.value_counts(normalize=True)

    if len(left_value_counts) != 0:
        if left_value_counts.values[0] >= .5:
            left_anchor_face_cluster = left_value_counts.index.values[0]
            left_anchor_face_encoding = np.average(alternation_face_df.loc[
                                                       (alternation_face_df['p_center_alignment'] == 'left') & (
                                                                   alternation_face_df[
                                                                       'p_face_cluster'] == left_anchor_face_cluster)].face_encodings.tolist(),
                                                   axis=0)[0]
            for candidate in left_value_counts.index.values[1:]:
                left_cluster_candidate = np.average(alternation_face_df.loc[
                                                        (alternation_face_df['p_center_alignment'] == 'left') & (
                                                                    alternation_face_df[
                                                                        'p_face_cluster'] == candidate)].face_encodings.tolist(),
                                                    axis=0)[0]
                if face_recognition.compare_faces([left_anchor_face_encoding], left_cluster_candidate)[0] == True:
                    matching_left_clusters.append(candidate)
            return left_anchor_face_cluster, matching_left_clusters
        else:
            return None, None
    else:
        return None, None


def right_face_clusters(alternation_face_df):
    matching_right_clusters = []

    right_value_counts = alternation_face_df[
        alternation_face_df['p_center_alignment'] == 'right'].p_face_cluster.value_counts(normalize=True)

    if len(right_value_counts) != 0:
        if right_value_counts.values[0] >= .5:
            right_anchor_face_cluster = right_value_counts.index.values[0]
            right_anchor_face_encoding = np.average(alternation_face_df.loc[
                                                        (alternation_face_df['p_center_alignment'] == 'right') & (
                                                                    alternation_face_df[
                                                                        'p_face_cluster'] == right_anchor_face_cluster)].face_encodings.tolist(),
                                                    axis=0)[0]
            for candidate in right_value_counts.index.values[1:]:
                right_cluster_candidate = np.average(alternation_face_df.loc[
                                                         (alternation_face_df['p_center_alignment'] == 'right') & (
                                                                     alternation_face_df[
                                                                         'p_face_cluster'] == candidate)].face_encodings.tolist(),
                                                     axis=0)[0]
                if face_recognition.compare_faces([right_anchor_face_encoding], right_cluster_candidate)[0] == True:
                    matching_right_clusters.append(candidate)
            return right_anchor_face_cluster, matching_right_clusters
        else:
            return None, None
    else:
        return None, None


def find_alternating_scenes(substantial_pairs, vision_df, face_df):
    alternating_scene_frames = []

    for pair in substantial_pairs:
        first_frame = vision_df[vision_df['shot_id'].isin([pair[0], pair[1]])][:1].index[0]
        last_frame = vision_df[vision_df['shot_id'].isin([pair[0], pair[1]])][-1:].index[0]
        alternation_face_df = face_df.copy()[first_frame - 1:last_frame]
        left_right_percentage = len(
            alternation_face_df[alternation_face_df['p_center_alignment'].isin(['left', 'right'])]) / len(
            alternation_face_df)
        prim_face_percentage = len(alternation_face_df[alternation_face_df['prim_char_flag'] == 1]) / len(
            alternation_face_df)
        left_anchor_face_cluster, matching_left_clusters = left_face_clusters(alternation_face_df)
        right_anchor_face_cluster, matching_right_clusters = right_face_clusters(alternation_face_df)
        print(pair)
        print(first_frame, last_frame)
        print('left or right frames make up', round(left_right_percentage, 2), 'percent of frames')
        print(round(prim_face_percentage, 2), 'percent of frames have a primary face')
        if left_anchor_face_cluster and right_anchor_face_cluster:
            print('left participant is', left_anchor_face_cluster, 'and also maybe', matching_left_clusters)
            print('right participant is', right_anchor_face_cluster, 'and also maybe', matching_right_clusters)
            if prim_face_percentage >= .8:
                print('!!!two character alternating scene candidate found!!!')
                alternating_scene_frames.append([first_frame, last_frame])
        else:
            print('no participants found')
        print()

    return alternating_scene_frames
