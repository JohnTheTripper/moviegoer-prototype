import pysrt
import spacy


# self-introduction
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


# other-introduction
def other_intro_this_is(sent_doc, start_token=0):
    pnoun_components = []
    pnoun_flag = 0

    if sent_doc[start_token].text in ['This', 'this'] and sent_doc[start_token + 1].text == "is" and sent_doc[
        start_token + 2].pos_ == 'PROPN':
        while pnoun_flag == 0 and start_token + 2 < len(sent_doc):
            if sent_doc[start_token + 2].pos_ == 'PROPN':
                pnoun_components.append(sent_doc[start_token + 2].text)
                start_token += 1
            else:
                pnoun_flag = 1

    string_value = ' '.join(pnoun_components)

    return string_value


def other_intro(sentence, nlp):
    sent_doc = nlp(sentence)

    start_token = 0
    characters = []

    try:
        while start_token < len(sent_doc):
            name = other_intro_this_is(sent_doc, start_token)
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


# direct-address
def direct_address_start(sent_doc):
    if len(sent_doc) > 1:
        if sent_doc[0].pos_ == 'PROPN' and sent_doc[1].text == ",":
            return sent_doc[0].text


def direct_address_end(sent_doc):
    if len(sent_doc) > 2:
        if sent_doc[-3].text == ',' and sent_doc[-2].pos_ == 'PROPN' and sent_doc[-1].pos_ == 'PUNCT':
            return sent_doc[-2].text


def direct_address_mid(sent_doc, start_token=0):
    if sent_doc[start_token].text == ',' and sent_doc[start_token + 1].pos_ == 'PROPN' and sent_doc[start_token + 2].text == ',':
        return sent_doc[start_token + 1].text


def direct_address(sentence, nlp):
    sent_doc = nlp(sentence)

    start_token = 0
    characters = []

    name = direct_address_start(sent_doc)
    if name:
        characters.append(name)
    name = direct_address_end(sent_doc)
    if name:
        characters.append(name)

    try:
        while start_token < len(sent_doc):
            name = direct_address_mid(sent_doc, start_token)
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
