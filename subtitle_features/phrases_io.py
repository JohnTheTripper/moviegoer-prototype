import pysrt
import spacy

def self_intro_i_am(sent_doc, start_token):
    pnoun_components = []
    pnoun_flag = 0

    if sent_doc[start_token].text == 'I' and (
            sent_doc[start_token + 1].text == "'m" or sent_doc[start_token + 1].text == "am") and sent_doc[
        start_token + 2].pos_ == 'PROPN':
        while pnoun_flag == 0 and start_token + 2 < len(sent_doc):
            if sent_doc[start_token + 2].pos_ == 'PROPN':
                pnoun_components.append(sent_doc[start_token + 2].text)
                start_token += 1
            else:
                pnoun_flag = 1

    string_value = ' '.join(pnoun_components)

    return string_value


def self_intro_my_name(sent_doc, start_token):
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

