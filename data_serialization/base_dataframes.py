import sys
import os
sys.path.append('../subtitle_features')
from subtitle_dataframes_io import *
from subtitle_auxiliary_io import *
sys.path.append('../vision_features')
from vision_dataframes_io import *
# sys.path.append('../audio_features')
# from audio_dataframes_io import *
sys.path.append('../unifying_features')
from time_reference_io import *
nlp = spacy.load('en')

film = 'plus_one_2019'
frame_choice = range(1, 5913)   # (1, number of frame files plus one)

serialized_object_directory = '../serialized_objects/'
film_directory = os.path.join(serialized_object_directory, film)
os.mkdir(film_directory)

# vision dataframes
face_df = generate_face_df(film, frame_choice)
face_df.to_pickle(os.path.join(film_directory, 'face_df_pre_cluster.pkl'))

vision_df = generate_vision_df(film, frame_choice)
vision_df.to_pickle(os.path.join(film_directory, 'vision_df_pre_cluster.pkl'))


# subtitle dataframes
subs_directory = '../subtitles/'
subs = pysrt.open(os.path.join(subs_directory, film + '.srt'))
srt_df = generate_srt_df(subs)
srt_df.head()
subtitle_df = generate_base_subtitle_df(subs)
subtitle_df = generate_subtitle_features(subtitle_df)
subtitle_df['cleaned_text'] = subtitle_df['concat_sep_text'].map(clean_line)
sentences = partition_sentences(remove_blanks(subtitle_df['cleaned_text'].tolist()), nlp)
subtitle_indices = tie_sentence_subtitle_indices(sentences, subtitle_df)
sentence_df = pd.DataFrame(list(zip(sentences, subtitle_indices)), columns=['sentence', 'subtitle_indices'])
sentence_df = generate_sentence_features(sentence_df, nlp)


srt_df.to_pickle(os.path.join(film_directory, 'srt_df.pkl'))
subtitle_df.to_pickle(os.path.join(film_directory, 'subtitle_df.pkl'))
sentence_df.to_pickle(os.path.join(film_directory, 'sentence_df_pre_cluster.pkl'))
