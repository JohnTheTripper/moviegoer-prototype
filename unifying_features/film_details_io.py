import sys
sys.path.append('../data_serialization')
from serialization_preprocessing_io import *


def get_upset_percentage(face_nocreds_df):
    try:
        sad_percentage = face_nocreds_df.p_emotion.value_counts(normalize=True)['sad']
    except KeyError:
        sad_percentage = 0
    try:
        angry_percentage = face_nocreds_df.p_emotion.value_counts(normalize=True)['angry']
    except KeyError:
        angry_percentage = 0

    upset_emotion_percentage = sad_percentage + angry_percentage

    return upset_emotion_percentage


def get_word_count(sentence_nocreds_df):
    space_count = 0
    sentence_list = sentence_nocreds_df.sentence.tolist()
    for sentence in sentence_list:
        for character in sentence:
            if character.isspace():
                space_count += 1
    word_count = space_count + len(sentence_list)

    return word_count


def get_words_per_sentence(sentence_nocreds_df):
    sentence_list = sentence_nocreds_df.sentence.tolist()

    word_count = get_word_count(sentence_nocreds_df)
    words_per_sentence = word_count / len(sentence_list)

    return words_per_sentence


def get_profanity_per_word(sentence_nocreds_df):
    word_count = get_word_count(sentence_nocreds_df)
    profanity_count = sentence_nocreds_df.profanity.sum()

    profanity_per_word = profanity_count/word_count

    return profanity_per_word


def display_film_baseline(film):
    srt_df, subtitle_df, sentence_df, vision_df, face_df = read_pickle(film)

    frame_before_credits = face_df[face_df['faces_found'] > 0].tail(1).index[0]  # final frame before credits
    vision_nocreds_df = vision_df[0:frame_before_credits].copy()
    face_nocreds_df = face_df[0:frame_before_credits].copy()
    subtitle_nocreds_df = subtitle_df[0:frame_before_credits].copy()
    sentence_nocreds_df = sentence_df[0:frame_before_credits].copy()

    upset_emotion_percentage = get_upset_percentage(face_nocreds_df)
    words_per_sentence = get_words_per_sentence(sentence_nocreds_df)
    profanity_per_word = get_profanity_per_word(sentence_nocreds_df)

    print('---------')
    print('Technical')
    print('---------')
    print('Aspect Ratio:', vision_nocreds_df.aspect_ratio.value_counts().index[0])
    print('Average shot duration:', round(len(vision_nocreds_df) / vision_nocreds_df.shot_id.max(), 2))
    print('Average frame brightness:', round(vision_nocreds_df.brightness.mean()))
    print('Average frame contrast:', round(vision_nocreds_df.contrast.mean()))
    print()
    print('--------')
    print('Dialogue')
    print('--------')
    print('Spoken sentences per minute:', round(len(sentence_nocreds_df) / (len(vision_nocreds_df) / 60)))
    print('Words per sentence:', round(words_per_sentence, 2))
    print()
    print('-------')
    print('Emotion')
    print('-------')
    print('Percentage of Upset facial expressions: ' + '{:.0%}'.format(upset_emotion_percentage))
    print('Instances of laughter, per minute:',
          round(subtitle_nocreds_df.laugh.sum() / (len(vision_nocreds_df) / 60), 2))
    if profanity_per_word == 0:
        print('The film contains no profanity.')
    else:
        print('One in', round(1 / profanity_per_word), 'words is a profanity.')


def get_long_takes(vision_nocreds_df, duration_threshold=20):
    long_takes = []
    for index, value in zip(vision_nocreds_df[vision_nocreds_df['blank'].isnull()].shot_id.value_counts().index,
                            vision_nocreds_df[vision_nocreds_df['blank'].isnull()].shot_id.value_counts().values):
        if value >= duration_threshold:
            long_takes.append(index)

    long_take_frame_pairs = []

    for shot_id in long_takes:
        if shot_id + 1 in long_takes:
            shot_brightness = vision_nocreds_df[vision_nocreds_df['shot_id'] == shot_id].brightness.mean()
            candidate_brightness = vision_nocreds_df[vision_nocreds_df['shot_id'] == shot_id + 1].brightness.mean()
            if (.95 * candidate_brightness) < shot_brightness < (1.05 * candidate_brightness):
                first_frame = vision_nocreds_df[vision_nocreds_df['shot_id'] == shot_id].head(1).index[0]
                last_frame = vision_nocreds_df[vision_nocreds_df['shot_id'] == shot_id + 1].tail(1).index[0]
            else:
                first_frame = vision_nocreds_df[vision_nocreds_df['shot_id'] == shot_id].head(1).index[0]
                last_frame = vision_nocreds_df[vision_nocreds_df['shot_id'] == shot_id].tail(1).index[0]
        elif shot_id - 1 in long_takes:
            shot_brightness = vision_nocreds_df[vision_nocreds_df['shot_id'] == shot_id].brightness.mean()
            candidate_brightness = vision_nocreds_df[vision_nocreds_df['shot_id'] == shot_id - 1].brightness.mean()
            if (.95 * candidate_brightness) < shot_brightness < (1.05 * candidate_brightness):
                first_frame = vision_nocreds_df[vision_nocreds_df['shot_id'] == shot_id - 1].head(1).index[0]
                last_frame = vision_nocreds_df[vision_nocreds_df['shot_id'] == shot_id].tail(1).index[0]
            else:
                first_frame = vision_nocreds_df[vision_nocreds_df['shot_id'] == shot_id].head(1).index[0]
                last_frame = vision_nocreds_df[vision_nocreds_df['shot_id'] == shot_id].tail(1).index[0]
        else:
            first_frame = vision_nocreds_df[vision_nocreds_df['shot_id'] == shot_id].head(1).index[0]
            last_frame = vision_nocreds_df[vision_nocreds_df['shot_id'] == shot_id].tail(1).index[0]

        long_take_frame_pairs.append((first_frame, last_frame))

    long_take_frame_pairs = list(set(long_take_frame_pairs))

    return long_take_frame_pairs


def get_color_shots(vision_df, primary_threshold=2, secondary_threshold=.3, brightness_minimum=35):
    color_shots_nested = []

    color_shots_nested.append(vision_df[(vision_df['red'] > vision_df['brightness'] * primary_threshold) & (
                vision_df['brightness'] >= brightness_minimum)].shot_id.tolist())
    color_shots_nested.append(vision_df[(vision_df['blue'] > vision_df['brightness'] * primary_threshold) & (
                vision_df['brightness'] >= brightness_minimum)].shot_id.tolist())
    color_shots_nested.append(vision_df[(vision_df['green'] > vision_df['brightness'] * primary_threshold) & (
                vision_df['brightness'] >= brightness_minimum)].shot_id.tolist())
    color_shots_nested.append(vision_df[(vision_df['red'] < vision_df['brightness'] * secondary_threshold) & (
                vision_df['brightness'] >= brightness_minimum)].shot_id.tolist())
    color_shots_nested.append(vision_df[(vision_df['blue'] < vision_df['brightness'] * secondary_threshold) & (
                vision_df['brightness'] >= brightness_minimum)].shot_id.tolist())
    color_shots_nested.append(vision_df[(vision_df['green'] < vision_df['brightness'] * secondary_threshold) & (
                vision_df['brightness'] >= brightness_minimum)].shot_id.tolist())

    color_shots = []
    for shot_id_list in color_shots_nested:
        for shot_id in shot_id_list:
            color_shots.append(shot_id)
    color_shots = list(set(color_shots))

    return color_shots


def get_nonconform_aspect_ratio_shots(vision_nocreds_df):
    aspect_ratio = vision_nocreds_df.aspect_ratio.value_counts().index[0]

    nonconform_aspect_ratio_shots = vision_nocreds_df[(vision_nocreds_df['aspect_ratio'] != aspect_ratio) & (vision_nocreds_df['blank'].isnull())].shot_id.tolist()

    nonconform_aspect_ratio_shots = list(set(nonconform_aspect_ratio_shots))

    return nonconform_aspect_ratio_shots
