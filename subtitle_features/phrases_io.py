import pysrt
import spacy


# self-introduction
def self_intro_i_am(sent_doc, start_token=0):
    """
    returns proper nouns as a string, from an NLP input sentence, if the proper noun introduces themselves
    e.g. in a sentence "Nice to meet you, I'm Shaun", returns Shaun
    may also return other words that are part of a proper noun phrase
    e.g. in a sentence "Nice to meet you, I'm Detective Yellen.", returns Detective Yellen
    """
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
    """
    returns proper nouns as a string, from an NLP input sentence, if the proper noun introduces themselves
    e.g. in a sentence "Hello, my name is Jamie.", returns Jamie
    may also return other words that are part of a proper noun phrase
    e.g. in a sentence "Hello, my name is Inspector Brauer.", returns Inspector Brauer
    """
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
    """
    returns proper nouns as a string, from an NLP input sentence, if the proper noun introduces themselves
    e.g. in a sentence "But everybody calls me John.", returns John
    """
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
    """
    returns a potential name by checking various self-introduction phrases
    only returns a single name, for now
    """
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
            return characters[0]
        else:
            return None

    if characters:
        return characters[0]
    else:
        return None


# other-introduction
def other_intro_this_is(sent_doc, start_token=0):
    """
    returns proper nouns as a string, from an NLP input sentence, if someone else introduces the proper noun
    e.g. in a sentence "I'm Jill, and this is Jane.", returns Jane
    """
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
    """
    returns a potential name by checking one other-introduction phrase (may add more later)
    only returns a single name, for now
    """
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
            return characters[0]
        else:
            return None

    if characters:
        return characters[0]
    else:
        return None


# direct-address
def direct_address_start(sent_doc):
    """
    returns proper nouns as a string, from an NLP input sentence, if the sentence begins addressing the proper noun
    e.g. in a sentence "Alex, that's so dumb.", returns Alex
    """
    if len(sent_doc) > 1:
        if sent_doc[0].pos_ == 'PROPN' and sent_doc[1].text == ",":
            return sent_doc[0].text


def direct_address_end(sent_doc):
    """
    returns proper nouns as a string, from an NLP input sentence, if the sentence ends addressing the proper noun
    e.g. in a sentence "That so dumb, Alex.", returns Alex
    """
    if len(sent_doc) > 2:
        if sent_doc[-3].text == ',' and sent_doc[-2].pos_ == 'PROPN' and sent_doc[-1].pos_ == 'PUNCT':
            return sent_doc[-2].text


def direct_address_mid(sent_doc, start_token=0):
    """
    returns proper nouns as a string, from an NLP input sentence, if the proper noun is addressed mid-sentence
    e.g. in a sentence "I can't believe, Alex, that you can't see how dumb that is.", returns Alex
    """
    if sent_doc[start_token].text == ',' and sent_doc[start_token + 1].pos_ == 'PROPN' and sent_doc[start_token + 2].text == ',':
        return sent_doc[start_token + 1].text


def direct_address(sentence, nlp):
    """
    returns a potential name by checking various direct-address phrases
    only returns a single name, for now
    """
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
            return characters[-1]
        else:
            return None

    if characters:
        return characters[-1]
    else:
        return None


# conversation boundaries
def conversation_boundary(sentence):
    """
    searches for various conversation starters or enders and returns a phrase depending on what was found
    """
    conversation_starters = ['how are you?', 'hi.', 'hi!', 'what can I do for you?']
    conversation_enders = ['goodbye.', 'goodbye,', 'bye,', 'bye.', 'see you.', 'see you,', 'see you later', 'see ya']
    for starter in conversation_starters:
        if starter in sentence.lower():
            return 'starter'
    for ender in conversation_enders:
        if ender in sentence.lower():
            return 'ender'

    return None


# profanity
def profanity(sentence, nlp):
    """
    returns a count of profanity
    useful as a measurement of emotion
    """
    sent_doc = nlp(sentence)
    profanity_count = 0

    profanities = ['fuck', 'fucking', 'fuckin', 'fucked', 'shit', 'shitty', 'bullshit', 'ass', 'dumbass', 'bitch', 'asshole', 'tit', 'cunt', 'goddamn', 'damn', 'dammit', 'cock', 'cocksucker', 'dick']
    for word in sent_doc:
        if word.lemma_.lower() in profanities:
            profanity_count += 1

    return profanity_count
