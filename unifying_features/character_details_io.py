from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from time_reference_io import *
import matplotlib.pyplot as plt
import pandas as pd


def display_character_dialogue_context(character_face_clusters, sentence_df, nlp):
    """
    calculates and prints various information about dialogue attributed to a specific character
    will be more useful as dialogue attribution improves
    """
    character_sentences = sentence_df[
        sentence_df['implied_speaker'].isin(character_face_clusters)].sentence.tolist()
    character_doc = nlp(' '.join(character_sentences))
    sent_nlp_list = list(character_doc.sents)

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
    for group in character_doc.noun_chunks:
        if group.root.pos_ != 'PRON':
            noun_groups.append(str(group))
    ng_count = Counter(noun_groups)
    ng_terms = []
    for ng in ng_count.most_common(5):
        ng_terms.append(ng[0])

    print('------------------')
    print('Directed Questions')         # second-person questions directed at "you"
    print('------------------')
    for index in direct_question_indices:
        print(sent_nlp_list[index])
    print()
    print('-------------------------------------')
    print('Possible Important Terms, Noun Groups')
    print('-------------------------------------')
    print(ng_terms)
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
    print()


def plot_character_filmwide_emotion(character_face_clusters, face_df):
    """
    plots a character's "upset" emotions (sad and angry) throughout eight portions ("sequences") of the film
    for now, attempts to create sequences by removing the credits and dividing the film into eight equal portions
    """
    frame_before_credits = face_df[face_df['faces_found'] > 0].tail(1).index[0]  # final frame before credits
    face_nocreds_df = face_df[0:frame_before_credits].copy()

    face_nocreds_df['sequence'] = pd.qcut(face_nocreds_df.index, q=8, labels=['1', '2', '3', '4', '5', '6', '7', '8'])
    face_nocreds_df['sequence'] = face_nocreds_df['sequence'].astype(int)

    sequences = range(1, 9)
    sad_angry_percentage = []
    for seq in sequences:
        try:
            sad_percentage = face_nocreds_df[
                (face_nocreds_df['p_face_cluster'].isin(character_face_clusters)) & (
                            face_nocreds_df['sequence'] == seq)].p_emotion.value_counts(normalize=True)['sad']
        except KeyError:
            sad_percentage = 0
        try:
            angry_percentage = face_nocreds_df[
                (face_nocreds_df['p_face_cluster'].isin(character_face_clusters)) & (
                            face_nocreds_df['sequence'] == seq)].p_emotion.value_counts(normalize=True)['angry']
        except KeyError:
            angry_percentage = 0
        sad_angry_percentage.append(100 * (sad_percentage + angry_percentage))
    plt.bar(x=sequences, height=(sad_angry_percentage), color='salmon')
    plt.xlabel('Film Sequence')
    plt.ylabel('Upset Percentage')
    plt.title('Upset Facial Expressions Throughout the Film')
    plt.show()
