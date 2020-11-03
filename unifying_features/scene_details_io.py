from sklearn.feature_extraction.text import TfidfVectorizer
from time_reference_io import *
from collections import Counter
import pandas as pd


def display_scene_dialogue_context(scene_dict, subtitle_df, sentence_df, nlp):
    scene_start_time = frame_to_time(scene_dict['first_frame'])
    scene_end_time = frame_to_time(scene_dict['last_frame'] + 1) # add 1 second; scene ends one second after this frame is onscreen
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

    # tf-idf
    film_doc = sentence_df.sentence.tolist()
    scene_doc = scene_sentence_df.sentence.tolist()
    del film_doc[scene_sentence_indices[0]: scene_sentence_indices[-1] + 1]
    scene_doc_joined = (' '.join(scene_doc))
    film_doc_joined = (' '.join(film_doc))
    film_scene_doc = [scene_doc_joined, film_doc_joined]

    # tf-idf
    vectorizer = TfidfVectorizer(use_idf=True, stop_words='english', ngram_range=(1, 3))
    idf_transformed = vectorizer.fit_transform(film_scene_doc)
    tf_idf_df = pd.DataFrame(idf_transformed[0].T.todense(), index=vectorizer.get_feature_names(), columns=["TF-IDF"])
    tf_idf_df = tf_idf_df.sort_values('TF-IDF', ascending=False)
    tf_idf_terms = list(tf_idf_df.head(5).index)

    # nlp prep
    scene_sentence_doc = nlp((' '.join(scene_sentence_df.sentence.tolist())))
    sent_nlp_list = list(scene_sentence_doc.sents)

    i_indices = []
    x = 0
    for sent in sent_nlp_list:
        for token in sent:
            if token.dep_ == 'nsubj' and token.text == 'I':
                if x not in i_indices:
                    i_indices.append(x)
        x += 1

    you_indices = []
    x = 0
    for sent in sent_nlp_list:
        for token in sent:
            if token.dep_ == 'nsubj' and token.text == 'you':
                if x not in you_indices:
                    you_indices.append(x)
        x += 1

    direct_question_indices = []
    x = 0
    for sent in sent_nlp_list:
        if sent[-1].text == '?':
            for token in sent:
                if token.dep_ == 'nsubj' and token.text == 'you':
                    direct_question_indices.append(x)
        x += 1
    direct_question_indices = list(set(direct_question_indices))

    noun_groups = []
    for group in scene_sentence_doc.noun_chunks:
        if group.root.pos_ != 'PRON':
            noun_groups.append(str(group))
    ng_count = Counter(noun_groups)
    ng_terms = []
    for ng in ng_count.most_common(5):
        ng_terms.append(ng[0])

    print('Possible important terms, tf-idf')
    print(tf_idf_terms)
    print()
    print('Possible important terms, noun groups')
    print(ng_terms)
    print()
    print('First-person declarations')
    for index in i_indices:
        print(sent_nlp_list[index])
    print()
    print('Second-person addresses')
    for index in you_indices:
        print(sent_nlp_list[index])
    print()
    print('Direct questions and responses')
    for index in direct_question_indices:
        print(sent_nlp_list[index])
        print(sent_nlp_list[index + 1])
        print()
    print()
    print('There are', scene_subtitle_df.laugh.sum(), 'instances of laughter.')
    print('There are', scene_subtitle_df.hesitation.sum(), 'midsentence hesitation interjections.')


def display_scene_emotions(scene_dict, face_df):
    scene_start_frame = scene_dict['first_frame']
    scene_last_frame = scene_dict['last_frame']
    scene_face_df = face_df.copy()[scene_start_frame - 1:scene_last_frame]
    left_face_clusters = [scene_dict['left_anchor_face_cluster']]
    for matching_cluster in scene_dict['matching_left_face_clusters']:
        left_face_clusters.append(matching_cluster)
    right_face_clusters = [scene_dict['right_anchor_face_cluster']]
    for matching_cluster in scene_dict['matching_right_face_clusters']:
        right_face_clusters.append(matching_cluster)
    left_emotion_index = scene_face_df[scene_face_df.p_face_cluster.isin(left_face_clusters)].emotion.value_counts(normalize=True).index[0]
    left_emotion_percentage = scene_face_df[scene_face_df.p_face_cluster.isin(left_face_clusters)].emotion.value_counts(normalize=True).values[0]
    print('Left character, with face clusters', left_face_clusters, 'has the primary emotion:', left_emotion_index, 'in', round(left_emotion_percentage, 2), 'percent of frames')
    right_emotion_index = scene_face_df[scene_face_df.p_face_cluster.isin(right_face_clusters)].emotion.value_counts(normalize=True).index[0]
    right_emotion_percentage = scene_face_df[scene_face_df.p_face_cluster.isin(right_face_clusters)].emotion.value_counts(normalize=True).values[0]
    print('Right character, with face clusters', right_face_clusters, 'has the primary emotion:', right_emotion_index,
          'in', round(right_emotion_percentage, 2), 'percent of frames')


def generate_scene_participants(scene_dict, subtitle_df, sentence_df):
    scene_start_time = frame_to_time(scene_dict['first_frame'])
    scene_end_time = frame_to_time(scene_dict['last_frame'] + 1) # add 1 second; scene ends one second after this frame is onscreen
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

    scene_participants = []
    for name in scene_sentence_df.direct_address.value_counts().index:
        if name[0].isupper():
            scene_participants.append(name.lower())

    for name in scene_subtitle_df.speaker.value_counts().index:
        scene_participants.append(name.lower())

    scene_participants = list(set(scene_participants))

    return scene_participants
