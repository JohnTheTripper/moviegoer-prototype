import sys
import pandas as pd
from subtitle_cleaning_io import *
from subtitle_auxiliary_io import *
from phrases_io import *
sys.path.append('../unifying_features')
from time_reference_io import *


def generate_srt_df(subs):
    """
    returns srt_df, a basic dataframes of subtitles and their start/end times
    """
    indices = []
    original_texts = []
    start_times = []
    end_times = []

#    subs.insert(0, subs[0])                # fix this to align df and srt
    for sub in subs:
        indices.append(sub.index)
        original_texts.append(sub.text)
        start_times.append(sub.start.to_time())
        end_times.append(sub.end.to_time())

    srt_df = pd.DataFrame(list(zip(indices, original_texts, start_times, end_times)), columns=['srt_index', 'original_text', 'start_time', 'end_time'])
#    srt_df.drop(index=0, inplace=True)
    srt_df = srt_df.set_index('srt_index')

    return srt_df


def generate_base_subtitle_df(subs):
    """
    returns subtitle_df, a dataframe with original/cleaned subtitle lines and their start/end times
    """
    indices = []
    original_texts = []
    start_times = []
    end_times = []
    concat_sep_texts = []
    separated_flag = []

    for sub in subs:
        indices.append(sub.index)
        original_texts.append(sub.text)
        start_times.append(sub.start.to_time())
        end_times.append(sub.end.to_time())
        top_line, bottom_line = concat_sep_lines(sub.text)
        if bottom_line != 0:
            concat_sep_texts.append(top_line)
            separated_flag.append(1)
            separated_flag.append(1)

            indices.append(sub.index)
            original_texts.append(sub.text)
            start_times.append(sub.start.to_time())
            end_times.append(sub.end.to_time())
            concat_sep_texts.append(bottom_line)
        else:
            concat_sep_texts.append(top_line)
            separated_flag.append(0)

    subtitle_df = pd.DataFrame(list(zip(indices, original_texts, start_times, end_times, concat_sep_texts, separated_flag)), columns=['srt_index', 'original_text', 'start_time', 'end_time', 'concat_sep_text', 'separated_flag'])

    return subtitle_df


def generate_subtitle_features(subtitle_df):
    """
    returns subtitle_df with various new feature columns added
    """
    subtitle_df['laugh'] = subtitle_df['concat_sep_text'].map(find_laugh)
    subtitle_df['hesitation'] = subtitle_df['concat_sep_text'].map(find_midsentence_interjection)
    subtitle_df['speaker'] = subtitle_df['concat_sep_text'].map(find_speaker)
    subtitle_df['music'] = subtitle_df['concat_sep_text'].map(find_music)
    subtitle_df['parenthetical'] = subtitle_df['concat_sep_text'].map(find_parenthetical)
    subtitle_df['el_parenthetical'] = subtitle_df['concat_sep_text'].map(find_el_parenthetical)
    subtitle_df['el_italic'] = subtitle_df['concat_sep_text'].map(find_el_italic)

    return subtitle_df


def generate_sentence_features(sentence_df, nlp):
    """
    returns subtitle_df with various new feature columns added
    """
    sentence_df['profanity'] = sentence_df['sentence'].apply(profanity, args=[nlp])
    sentence_df['self_intro'] = sentence_df['sentence'].apply(self_intro, args=[nlp])
    sentence_df['other_intro'] = sentence_df['sentence'].apply(other_intro, args=[nlp])
    sentence_df['direct_address'] = sentence_df['sentence'].apply(direct_address, args=[nlp])
    sentence_df['conv_boundary'] = sentence_df['sentence'].map(conversation_boundary)

    return sentence_df


def tie_sentence_subtitle_indices(sentences, subtitle_df):
    """
    for each row of sentence_df, returns a list of indices corresponding to subtitle_df, to tie them together
    """
    row_count = 0
    character_count = 0
    subtitle_indices = []

    for sent in sentences:
        row_indices = []
        first_char = 1
        for character in sent:
            row_found = 0
            while row_found == 0:
                if len(subtitle_df['cleaned_text'][row_count]) != 0 and character_count < len(subtitle_df['cleaned_text'][row_count]):
                    if first_char == 1 and subtitle_df['cleaned_text'][row_count][character_count] == ' ':  # single subtitle object split into two halves
                        character_count += 1
                    elif character == subtitle_df['cleaned_text'][row_count][character_count]:
                        row_indices.append(row_count)

                        character_count += 1
                        row_found = 1
                        first_char = 0
                else:
                    if character_count < len(subtitle_df['cleaned_text'][row_count]):
                        character_count += 1
                    elif character == ' ':  # single sentence created from multiple subtitles
                        row_found = 1
                        row_count += 1
                        character_count = 0
                    else:
                        row_count += 1
                        character_count = 0

        subtitle_indices.append(list(set(row_indices)))

    return subtitle_indices


def get_implied_speaker(sentence_index, sentence_df, subtitle_df, face_df):
    """
    returns the implied speaker of a sentence
    calculates the frames onscreen during the entire duration of the subtitle, and checks for faces with open mouths
    somewhat experimental
    """
    sub_indices = sentence_df.iloc[sentence_index].subtitle_indices
    start_frame = time_to_frame(subtitle_df.iloc[sub_indices[0]].start_time)
    end_frame = time_to_frame(subtitle_df.iloc[sub_indices[-1]].end_time)
    open_mouth_df = face_df[face_df['p_open_mouth'] == 1].loc[start_frame:end_frame]
    if len(open_mouth_df) == 0:
        return None
    elif len(open_mouth_df.p_face_cluster.value_counts()) == 1:
        return open_mouth_df.p_face_cluster.value_counts().index[0]
    elif len(open_mouth_df.p_face_cluster.value_counts()) >= 2: # must check for tie between first two faces
        if open_mouth_df.p_face_cluster.value_counts().values[0] == open_mouth_df.p_face_cluster.value_counts().values[1]:
            return None
        else:
            return open_mouth_df.p_face_cluster.value_counts().index[0]


def generate_speakers(sentence_df, subtitle_df, face_df):
    """
    returns sentence_df with two new columns for speaker names
    finds (best-guess) implied speakers based on onscreen faces with open mouths
    also finds (ground truth) offscreen speaker names
    """
    offscreen_speakers = []

    for sentence_index in range(0, len(sentence_df)):
        sub_start_index = sentence_df.iloc[sentence_index].subtitle_indices[0]
        if subtitle_df.iloc[sub_start_index].speaker:
            offscreen_speakers.append(subtitle_df.iloc[sub_start_index].speaker)
        else:
            offscreen_speakers.append(None)

    implied_speakers = []

    for sentence_index in range(0, len(sentence_df)):
        if not offscreen_speakers[sentence_index]:
            implied_speakers.append(get_implied_speaker(sentence_index, sentence_df, subtitle_df, face_df))
        else:
            implied_speakers.append(None)

    sentence_df['offscreen_speaker'] = offscreen_speakers
    sentence_df['implied_speaker'] = implied_speakers

    return sentence_df


def add_paren_offscreen_speaker(subtitle_df, sentence_df, nlp):
    """
    returns subtitle_df with updated speaker column
    originally, speaker column only identified speakers in this form `ADAM: No way` but not this form: `[Adam] No way`
    generates a character list using character_subtitle_mentions() and compares parentheticals against this list
    """
    speaker_amended = []
    sentences = sentence_df.sentence.tolist()
    mentioned_characters = character_subtitle_mentions(sentences, nlp)
    generic_characters = ['man', 'woman', 'boy', 'girl', 'both', 'all']

    paren_list = subtitle_df.parenthetical.tolist()
    speaker_list = subtitle_df.speaker.tolist()

    for paren, speaker in zip(paren_list, speaker_list):
        if speaker:
            speaker_amended.append(speaker)
        elif paren:
            if paren in mentioned_characters or paren in generic_characters:
                speaker_amended.append(paren)
            else:
                speaker_amended.append(None)
        else:
            speaker_amended.append(None)

    subtitle_df['speaker'] = speaker_amended

    return subtitle_df


def generate_scene_sub_sent_df(scene_dict, subtitle_df, sentence_df):
    """
    returns subtitle_df and sentence_df at the scene level, according to a scene dictionary's start and end frames
    """
    scene_start_time = frame_to_time(scene_dict['first_frame'])
    scene_end_time = frame_to_time(
        scene_dict['last_frame'] + 1)  # add 1 second; scene ends one second after this frame is onscreen
    scene_subtitle_df = subtitle_df[
        (subtitle_df['end_time'] > scene_start_time) & (subtitle_df['start_time'] < scene_end_time)].copy()

    scene_sentence_indices = []
    x = 0
    for sub_index_list in sentence_df.subtitle_indices.values:
        for sub_index in sub_index_list:
            if sub_index in scene_subtitle_df.index.values:
                scene_sentence_indices.append(x)
        x += 1
    scene_sentence_df = sentence_df[scene_sentence_indices[0]: scene_sentence_indices[-1] + 1]

    return scene_subtitle_df, scene_sentence_df
