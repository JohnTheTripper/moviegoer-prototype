import pandas as pd
import datetime
from subtitle_cleaning_io import *
from phrases_io import *


def generate_srt_df(subs):
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
    subtitle_df['laugh'] = subtitle_df['concat_sep_text'].map(find_laugh)
    subtitle_df['hesitation'] = subtitle_df['concat_sep_text'].map(find_midsentence_interjection)
    subtitle_df['speaker'] = subtitle_df['concat_sep_text'].map(find_speaker)
    subtitle_df['music'] = subtitle_df['concat_sep_text'].map(find_music)
    subtitle_df['parenthetical'] = subtitle_df['concat_sep_text'].map(find_parenthetical)
    subtitle_df['el_parenthetical'] = subtitle_df['concat_sep_text'].map(find_el_parenthetical)
    subtitle_df['el_italic'] = subtitle_df['concat_sep_text'].map(find_el_italic)

    return subtitle_df


def generate_sentence_features(sentence_df, nlp):
    sentence_df['self_intro'] = sentence_df['sentence'].apply(self_intro, args=[nlp])
    sentence_df['other_intro'] = sentence_df['sentence'].apply(other_intro, args=[nlp])
    sentence_df['direct_address'] = sentence_df['sentence'].apply(direct_address, args=[nlp])
    sentence_df['conv_boundary'] = sentence_df['sentence'].map(conversation_boundary)

    return sentence_df


def tie_sentence_subtitle_indices(sentences, subtitle_df):
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
