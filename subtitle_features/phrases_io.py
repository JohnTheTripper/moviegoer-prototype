import pysrt
import spacy


# self-introductions

def self_intro_i_am(sent_doc, start_token=0):
    pnoun_components = []
    pnoun_flag = 0

    if sent_doc[start_token].text == 'I' and (
            sent_doc[start_token + 1].text == "'m" or sent_doc[start_token + 1].text == "am") and sent_doc[
        start_token + 2].pos_ == 'PROPN':
        if (start_token + 3) < len(sent_doc):               # this clause stops the line "I'm Wilmington's horse"
            if sent_doc[start_token + 3].tag_ == 'POS':
                return None
        while pnoun_flag == 0 and start_token + 2 < len(sent_doc):
            if sent_doc[start_token + 2].pos_ == 'PROPN':
                pnoun_components.append(sent_doc[start_token + 2].text)
                start_token += 1
            else:
                pnoun_flag = 1

    string_value = ' '.join(pnoun_components)

    return string_value


def self_intro_my_name(sent_doc, start_token=0):
    pnoun_components = []
    pnoun_flag = 0

    if sent_doc[start_token].text in ['My', 'my'] and sent_doc[start_token + 1].text == "name" and sent_doc[start_token + 2].text == "is" and sent_doc[start_token + 3].pos_ == 'PROPN':
        while pnoun_flag == 0 and start_token + 3 < len(sent_doc):
            if sent_doc[start_token + 3].pos_ == 'PROPN':
                pnoun_components.append(sent_doc[start_token + 3].text)
                start_token += 1
            else:
                pnoun_flag = 1

    string_value = ' '.join(pnoun_components)

    return string_value


def self_intro_calls_me(sent_doc, start_token=0):
    pnoun_components = []
    pnoun_flag = 0

    if sent_doc[start_token].text == 'calls' and sent_doc[start_token + 1].text == "me" and sent_doc[start_token + 2].pos_ == 'PROPN':
        while pnoun_flag == 0 and start_token + 2 < len(sent_doc):
            if sent_doc[start_token + 2].pos_ == 'PROPN':
                pnoun_components.append(sent_doc[start_token + 2].text)
                start_token += 1
            else:
                pnoun_flag = 1

    string_value = ' '.join(pnoun_components)

    return string_value


def self_intro(sentence, nlp):
    sent_doc = nlp(sentence)

    start_token = 0
    characters = []

    try:
        while start_token < len(sent_doc):
            name = self_intro_i_am(sent_doc, start_token)
            if name:
                characters.append(name)
            name = self_intro_my_name(sent_doc, start_token)
            if name:
                characters.append(name)
            name = self_intro_calls_me(sent_doc, start_token)
            if name:
                characters.append(name)
            start_token += 1
    except IndexError:
        if characters:
            return characters
        else:
            return None

    if characters:
        return characters
    else:
        return None
