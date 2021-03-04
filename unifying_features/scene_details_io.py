from sklearn.feature_extraction.text import TfidfVectorizer
from time_reference_io import *
from film_details_io import *
from collections import Counter
import pandas as pd
sys.path.append('../subtitle_features')
from subtitle_dataframes_io import *


def generate_scene_level_dfs(scene_dict, vision_df, face_df, subtitle_df, sentence_df):
    """
    creates scene-level dataframes from an individual entry from the scene dictionary
    """
    vision_scene_df = vision_df[scene_dict['first_frame'] - 1:scene_dict['last_frame']]
    face_scene_df = face_df[scene_dict['first_frame'] - 1:scene_dict['last_frame']]
    scene_start_time = frame_to_time(scene_dict['first_frame'])
    scene_end_time = frame_to_time(
        scene_dict['last_frame'] + 1)  # add 1 second; scene ends one second after this frame is onscreen
    subtitle_scene_df = subtitle_df[
        (subtitle_df['end_time'] > scene_start_time) & (subtitle_df['start_time'] < scene_end_time)].copy()

    scene_sentence_indices = []
    x = 0
    for sub_index_list in sentence_df.subtitle_indices.values:
        for sub_index in sub_index_list:
            if sub_index in subtitle_scene_df.index.values:
                scene_sentence_indices.append(x)
        x += 1
    sentence_scene_df = sentence_df[scene_sentence_indices[0]: scene_sentence_indices[-1] + 1]

    return vision_scene_df, face_scene_df, subtitle_scene_df, sentence_scene_df


def get_scene_cadence(vision_scene_df, subtitle_scene_df):
    """
    returns the number of words per minute in a scene
    """
    scene_duration = len(vision_scene_df)
    cadence = len(subtitle_scene_df) / (scene_duration / 60)

    if round(cadence) >= 35:
        print('This scene has a fast cadence, with a conversation speed of', round(cadence), 'sentences per minute.')
    elif round(cadence) < 20:
        print('This scene has a slow cadence, with a conversation speed of', round(cadence), 'sentences per minute.')
    else:
        print('This scene has a medium cadence, with a conversation speed of', round(cadence), 'sentences per minute.')

    return cadence


def get_scene_ppw(sentence_scene_df):
    """
    returns the proportion of profanity in a scene
    useful as a measure of drama, especially against the film baseline
    """
    space_count = 0
    sentence_list = sentence_scene_df.sentence.tolist()
    for sentence in sentence_list:
        for character in sentence:
            if character.isspace():
                space_count += 1

    word_count = space_count + len(sentence_list)
    profanity_count = sentence_scene_df.profanity.sum()

    profanity_per_word = profanity_count / word_count

    if profanity_per_word == 0:
        print('The scene contains no profanity.')
    else:
        print('One in', round(1 / profanity_per_word), 'words is a profanity.')

    return profanity_per_word


def display_scene_important_phrases(sentence_df, sentence_scene_df, nlp):
    """
    prints potentially important phrases of a scene:
    noun chunks
    tf-idf phrases vs. the entire movie
    """
    # tf_idf data preparation
    film_doc_df = sentence_df.copy()
    film_doc_df.drop(range(sentence_scene_df.index.start, sentence_scene_df.index.stop), inplace=True)
    film_doc = film_doc_df.sentence.tolist()
    scene_doc = sentence_scene_df.sentence.tolist()
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
    scene_sentence_doc = nlp((' '.join(sentence_scene_df.sentence.tolist())))

    noun_groups = []
    for group in scene_sentence_doc.noun_chunks:
        if group.root.pos_ != 'PRON':
            noun_groups.append(str(group))
    ng_count = Counter(noun_groups)
    ng_terms = []
    for ng in ng_count.most_common(5):
        ng_terms.append(ng[0])

    print('--------------------------------')
    print('Possible Important Terms, TF-IDF')
    print('--------------------------------')
    print(tf_idf_terms)
    print()
    print()
    print('-------------------------------------')
    print('Possible Important Terms, Noun Groups')
    print('-------------------------------------')
    print(ng_terms)


def display_scene_important_sentences(sentence_scene_df, nlp):
    """
    prints potentially important sentences of a scene
    directed questions and responses - questions directly addressing "you", and the next sentence
    first-person declarations - sentences with "I" as the subject
    second-person addresses - sentences addressing "you"
    """
    # nlp prep
    scene_sentence_doc = nlp((' '.join(sentence_scene_df.sentence.tolist())))
    sent_nlp_list = list(scene_sentence_doc.sents)

    i_indices = []
    x = 0
    for sent in sent_nlp_list:
        for token in sent:
            if token.dep_ == 'nsubj' and token.text == 'I' and sent[-1].text != '?':
                if x not in i_indices:
                    i_indices.append(x)
        x += 1

    you_indices = []
    x = 0
    for sent in sent_nlp_list:
        if sent[-1].text != '?':
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

    print('--------------------------------')
    print('Directed Questions and Responses')       # second-person questions directed at "you"
    print('--------------------------------')
    for index in direct_question_indices:
        print(sent_nlp_list[index])
        print(sent_nlp_list[index + 1])
        print()
    print('-------------------------')
    print('First-Person Declarations')
    print('-------------------------')
    for index in i_indices:
        print(sent_nlp_list[index])
    print()
    print('-----------------------')
    print('Second-Person Addresses')
    print('-----------------------')
    for index in you_indices:
        print(sent_nlp_list[index])


def display_scene_start_end(sentence_scene_df):
    """
    prints the first and last three lines of a scene
    """
    scene_sentences = sentence_scene_df.sentence.tolist()
    print('-------------------------------')
    print('Icebreaker (Conversation Start)')            # first three sentences of scene
    print('-------------------------------')
    print(scene_sentences[0])
    print(scene_sentences[1])
    print(scene_sentences[2])
    print()
    print()
    print('-------------------------')
    print('Kicker (Conversation End)')                  # final three sentences of scene
    print('-------------------------')
    print(scene_sentences[-3])
    print(scene_sentences[-2])
    print(scene_sentences[-1])


def display_scene_emotions(scene_dict, face_scene_df):
    """
    prints the primary emotion for each of the two anchor characters
    primary emotion is each character's most common emotion in each frame in which they appear
    doesn't return any values - this code can form the basis of other functions
    """
    left_face_clusters = [scene_dict['left_anchor_face_cluster']]
    for matching_cluster in scene_dict['matching_left_face_clusters']:
        left_face_clusters.append(matching_cluster)
    right_face_clusters = [scene_dict['right_anchor_face_cluster']]
    for matching_cluster in scene_dict['matching_right_face_clusters']:
        right_face_clusters.append(matching_cluster)
    left_emotion_index = face_scene_df[face_scene_df.p_face_cluster.isin(left_face_clusters)].p_emotion.value_counts(normalize=True).index[0]
    left_emotion_percentage = face_scene_df[face_scene_df.p_face_cluster.isin(left_face_clusters)].p_emotion.value_counts(normalize=True).values[0]
    print('Left character, with face clusters', left_face_clusters, 'has the primary emotion:', left_emotion_index + ', in ' + '{:.0%}'.format(left_emotion_percentage) + ' of frames')
    right_emotion_index = face_scene_df[face_scene_df.p_face_cluster.isin(right_face_clusters)].p_emotion.value_counts(normalize=True).index[0]
    right_emotion_percentage = face_scene_df[face_scene_df.p_face_cluster.isin(right_face_clusters)].p_emotion.value_counts(normalize=True).values[0]
    print('Right character, with face clusters', right_face_clusters, 'has the primary emotion:', right_emotion_index +
          ', in ' + '{:.0%}'.format(right_emotion_percentage) + ' of frames')


def generate_scene_participants(subtitle_scene_df, sentence_scene_df):
    """
    returns a best-guess at names of possible scene participants
    looks for sentences with a direct-address ('Many thanks, Adam.') or offscreen speaker label
    """
    scene_participants = []
    for name in sentence_scene_df.direct_address.value_counts().index:
        if name[0].isupper():
            scene_participants.append(name.lower())

    for name in subtitle_scene_df.speaker.value_counts().index:
        scene_participants.append(name.lower())

    scene_participants = list(set(scene_participants))

    return scene_participants
