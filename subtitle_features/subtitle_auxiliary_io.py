from subtitle_dataframes_io import *
from collections import Counter
from datetime import datetime, date, timedelta


def character_subtitle_mentions(sentences, nlp):
    people_blacklist = ['Jesus', 'Jesus Christ', 'Whoo', 'God', 'Mm', 'Dude', 'Mm-hmm', 'Huh']

    doc = nlp(' '.join(sentences))
    people = []

    for ent in doc.ents:
        if ent.label_ == 'PERSON' and ent.text not in people_blacklist:
            people.append(ent.text)
    count = Counter(people)

    return count.most_common(10)


def character_offscreen_speakers(subtitle_df):
    speaker_blacklist = ['MAN', 'WOMAN', 'BOY', 'GIRL', 'BOTH', 'ALL']

    speaker_counts = subtitle_df.speaker.value_counts()
    speakers = []

    x = 0
    while x < len(speaker_counts):
        if speaker_counts.index[x] not in speaker_blacklist:
            speakers.append((speaker_counts.index[x], speaker_counts[x]))
        x += 1

    return speakers[0:10]


def dialogue_breaks(subtitle_df, threshold=10):
    x = 1
    delay_threshold = timedelta(seconds=threshold)
    breaks = []

    while x < len(subtitle_df):
        if subtitle_df.iloc[x].cleaned_text or subtitle_df.iloc[x].laugh == 1:
            y = 1

            while not subtitle_df.iloc[x - y].cleaned_text and subtitle_df.iloc[x - y].laugh == 0:
                y += 1
            delay = datetime.combine(date.today(), subtitle_df.iloc[x].start_time) - datetime.combine(date.today(),
                                                                                                      subtitle_df.iloc[
                                                                                                          x - y].end_time)

            if delay > delay_threshold:
                breaks.append(subtitle_df.iloc[x].srt_index)
        x += 1

    return breaks
