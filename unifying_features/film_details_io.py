import sys
sys.path.append('../data_serialization')
from serialization_preprocessing_io import *


def generate_nocreds_dfs(vision_df, face_df):
    """
    returns vision dataframes without the credits
    currently looks for the final frame with a face, and uses that as the last frame before credits
    """
    frame_before_credits = face_df[face_df['faces_found'] > 0].tail(1).index[0]  # final frame before credits
    vision_nocreds_df = vision_df[0:frame_before_credits].copy()
    face_nocreds_df = face_df[0:frame_before_credits].copy()

    return vision_nocreds_df, face_nocreds_df


def get_upset_percentage(face_nocreds_df):
    """
    returns percentage of frames that are sad or angry
    """
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


def get_word_count(sentence_df):
    """
    returns number of words in a list of sentences
    for each sentence, counts number of spaces, then adds one
    """
    space_count = 0
    sentence_list = sentence_df.sentence.tolist()
    for sentence in sentence_list:
        for character in sentence:
            if character.isspace():
                space_count += 1
    word_count = space_count + len(sentence_list)

    return word_count


def get_words_per_sentence(sentence_df):
    """
    returns average number of words per sentence from a list of sentences
    """
    sentence_list = sentence_df.sentence.tolist()

    word_count = get_word_count(sentence_df)
    words_per_sentence = word_count / len(sentence_list)

    return words_per_sentence


def get_profanity_per_word(sentence_df):
    """
    returns a percentage of profanity per word
    more easily presentable in the inverse form of *word_per_profanity*, e.g. "one in 140 words is a profanity"
    """
    word_count = get_word_count(sentence_df)

    profanity_count = sentence_df.profanity.sum()
    profanity_per_word = profanity_count / word_count

    if profanity_per_word == 0:
        print('The film contains no profanity.')
    else:
        print('One in', round(1 / profanity_per_word), 'words is a profanity.')

    return profanity_per_word


def display_film_baseline(film):
    """
    calculates and prints various information about scene-level dialogue
    attempts to remove credits from calculation by looking for the last face of the film
    doesn't return any values - this code can form the basis of other functions
    useful for comparing with display_scene_dialogue_context(), though some *per minute* stats don't cleanly compare
    """
    srt_df, subtitle_df, sentence_df, vision_df, face_df = read_pickle(film)

    vision_nocreds_df, face_nocreds_df = generate_nocreds_dfs(vision_df, face_df)

    upset_emotion_percentage = get_upset_percentage(face_nocreds_df)
    words_per_sentence = get_words_per_sentence(sentence_df)

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
    print('Spoken sentences per minute:', round(len(sentence_df) / (len(vision_nocreds_df) / 60)))
    print('Words per sentence:', round(words_per_sentence, 2))
    print()
    print('-------')
    print('Emotion')
    print('-------')
    print('Percentage of Upset facial expressions: ' + '{:.0%}'.format(upset_emotion_percentage))
    print('Instances of laughter, per minute:',
          round(subtitle_df.laugh.sum() / (len(vision_nocreds_df) / 60), 2))
    get_profanity_per_word(sentence_df)
