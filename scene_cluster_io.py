from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def label_clusters(dialogue_folder, frame_choice, film, threshold):

    model = VGG16(weights='imagenet', include_top=False)

    vgg16_feature_list = []

    for x in frame_choice:
        img_path = dialogue_folder + '/' + film + '_frame' + str(x) + '.jpg'
        img = image.load_img(img_path, target_size=(256, 256))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        vgg16_feature = model.predict(img_data)
        vgg16_feature_np = np.array(vgg16_feature)
        vgg16_feature_list.append(vgg16_feature_np.flatten())

        x += 1

    vgg16_feature_list_np = np.array(vgg16_feature_list)
    vgg16_feature_list_np.shape

    hac = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold).fit(vgg16_feature_list_np)
    hac_labels = hac.labels_
    print('Number of clusters:', hac.n_clusters_)

    return hac_labels


def predict_mcu(dialogue_folder, model, frame_choice, film):
    image_list = []
    for x in frame_choice:
        image_list.append(img_to_array(
            load_img(dialogue_folder + '/' + film + '_frame' + str(x) + '.jpg', target_size=(128, 128),
                     color_mode='grayscale')))

    image_array = np.array(image_list)
    y_pred = model.predict_classes(image_array)

    # the model's predict_classes method creates a NumPy array of arrays; this converts it to a list of 0/1 integers
    y_pred_values = []
    for prediction in y_pred:
        y_pred_values.append(prediction[0])

    return y_pred_values


def get_shot_ids(frame_choice, hac_labels):
    shot_id = 0
    shot_id_list = []
    prev_frame = 1000

    for frame_file, cluster in zip(frame_choice, hac_labels):
        if cluster != prev_frame and prev_frame != 1000:
            shot_id += 1
        shot_id_list.append(shot_id)
        prev_frame = cluster

    return shot_id_list


def get_alternating_pairs(frame_choice, hac_labels, y_pred_values, shot_id_list):
    # to check for an A/B/A/B pattern, we must store the previous three clusters in memory
    prev_clust_1 = 1001
    prev_clust_2 = 1002
    prev_clust_3 = 1003
    prev_shot_id = -1
    alternate_a_list = []
    alternate_b_list = []

    # zip our various lists into a usable data structure
    for frame_file, cluster, mcu_flag, shot_id in zip(frame_choice, hac_labels, y_pred_values, shot_id_list):
        # when iterating through each frame, look for an A/B/A/B pattern, and save the clusters of any patterns
        if cluster == prev_clust_2 and prev_clust_1 == prev_clust_3:
            alternate_a_list.append(
                min(cluster, prev_clust_1))  # min and max are used to avoid duplicates of (1, 2), (2, 1)
            alternate_b_list.append(max(cluster, prev_clust_1))

        # we use prev_shot_id to identify when there's a new shot (when the cluster value changes)
        # every time there's a new shot, we update the cluster memory
        if shot_id != prev_shot_id:
            prev_shot_id = shot_id
            prev_clust_3 = prev_clust_2
            prev_clust_2 = prev_clust_1
            prev_clust_1 = cluster

    # save unique alternating pairs
    alternating_pairs = []
    for a, b, in zip(alternate_a_list, alternate_b_list):
        if [int(a), int(b)] not in alternating_pairs:
            alternating_pairs.append([int(a), int(b)])

    return alternating_pairs


def mcu_check(alternating_pairs, scene_df):
    speaker_pairs = []
    print('cluster\t', 'count\t', 'mcu probability')

    for pair in alternating_pairs:
        # calculate the mean of each cluster's MCU column
        mean_a = scene_df.loc[scene_df['cluster'] == pair[0]]['mcu'].mean()
        mean_b = scene_df.loc[scene_df['cluster'] == pair[1]]['mcu'].mean()
        print(pair[0], '\t', scene_df.loc[scene_df['cluster'] == pair[0]]['mcu'].count(), '\t',
              '{0:.2f}%'.format(mean_a * 100))
        print(pair[1], '\t', scene_df.loc[scene_df['cluster'] == pair[1]]['mcu'].count(), '\t',
              '{0:.2f}%'.format(mean_b * 100))

        # an alternating pair will pass the MCU check if BOTH clusters have a MCU mean greater than .5
        if mean_a > .5 and mean_b > .5:
            print('Passes MCU check')
            speaker_pairs.append(pair)
        else:
            print('Fails MCU check')
        print()

    return speaker_pairs


def anchor_scenes(speaker_pairs, scene_df):
    anchors = []

    for pair in speaker_pairs:
        # designate the first and last frames with either Speaker A or Speaker B clusters as Anchors
        anchor_start = scene_df.loc[
            (scene_df['cluster'] == pair[0]) | (scene_df['cluster'] == pair[1])].frame_file.min()
        anchor_end = scene_df.loc[(scene_df['cluster'] == pair[0]) | (scene_df['cluster'] == pair[1])].frame_file.max()

        print('Speaker A and B Clusters:', pair)
        print('Anchor Start/End Frames:', anchor_start, anchor_end)
        print()
        anchors.append((anchor_start, anchor_end))

    return anchors


def expand_scenes(speaker_pairs, scene_df):
    expanded_scenes = []

    for pair in speaker_pairs:
        # designate the first and last frames with either Speaker A or Speaker B clusters as Anchors
        anchor_start = scene_df.loc[
            (scene_df['cluster'] == pair[0]) | (scene_df['cluster'] == pair[1])].frame_file.min()
        anchor_end = scene_df.loc[(scene_df['cluster'] == pair[0]) | (scene_df['cluster'] == pair[1])].frame_file.max()
        # find all unique clusters between the anchor_start and anchor_end frames
        cutaways = scene_df.loc[
            (scene_df['frame_file'] > anchor_start) & (scene_df['frame_file'] < anchor_end)].cluster.unique()
        cutaways = cutaways[cutaways != pair[0]]  # remove the Speaker A and Speaker B clusters from this list
        cutaways = cutaways[cutaways != pair[1]]
        print('Speaker A and B Clusters:', pair)
        print('Anchor Start/End Frames:', anchor_start, anchor_end)
        print('Cutaway Clusters:', cutaways)

        scene_start = anchor_start
        min_flag = 0

        # expand
        while min_flag == 0:
            try:
                if int(scene_df.loc[scene_df['frame_file'] == (scene_start - 1)].cluster) in cutaways:
                    scene_start -= 1
                else:
                    min_flag = 1
            except TypeError:  # error if hitting the beginning of the frame list
                min_flag = 1

        scene_end = anchor_end
        max_flag = 0
        while max_flag == 0:
            try:
                if int(scene_df.loc[scene_df['frame_file'] == (scene_end + 1)].cluster) in cutaways:
                    scene_end += 1
                else:
                    max_flag = 1
            except TypeError:  # error if hitting the end of the frame list
                max_flag = 1

        print('Expanded Start/End Frames:', scene_start, scene_end)
        print()
        expanded_scenes.append((scene_start, scene_end))

    return expanded_scenes


