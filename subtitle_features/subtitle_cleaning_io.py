import spacy


def concat_sep_lines(text):
    newline = text.find('\n')
    if newline == -1:                                       # one-liner
        return text, 0
    elif text[0] == '-' and text[newline + 1] == '-':       # two-liner spoken by two characters
        top_line = text[1:newline]
        bottom_line = text[newline + 2:]
        return top_line.lstrip(), bottom_line.lstrip()
    else:                                                   # two-liner from one character
        concat_line = text[:newline] + ' ' + text[newline + 1:]
        return concat_line, 0


def generate_single_lines(subs):
    single_lines = []
    for sub_object in subs:
        text = sub_object.text
        line_a, line_b = concat_sep_lines(text)
        single_lines.append(line_a)
        if line_b != 0:
            single_lines.append(line_b)
    return single_lines


def music_clean(line):
    entire_line_music = 0
    if line[:1] == '♪' and line[-1:] == '♪':
        entire_line_music = 1
        line = ''
    return entire_line_music, line


def parenthetical_clean(line):
    entire_line_parenthetical = 0
    if line[:1] == '(' and line[-1:] == ')':
        # entire_line_parenthetical = line[1:-1]
        entire_line_parenthetical = line
        line = ''
    return entire_line_parenthetical, line


def italic_clean(line):
    entire_line_italic = 0
    if line[:3] == '<i>' and line[-4:] == '</i>':
        entire_line_italic = 1
    line = line.replace('<i>', '')
    line = line.replace('</i>', '')
    return entire_line_italic, line


def speaker_clean(line):                    # replicated below without clean, will be depreciated
    colon_find = line.find(':')
    speaker = 'none'
    if colon_find != -1 and line[0:colon_find].isupper():
        speaker = line[0:colon_find]
        line = line[colon_find + 2:]
    return speaker, line


def find_speaker(line):
    colon_find = line.find(':')
    # speaker = 'none'              # let it return NULL
    if colon_find != -1 and line[0:colon_find].isupper():
        speaker = line[0:colon_find]
    else:
        speaker = None
    return speaker


def laugh_clean(line):                       # replicated below without clean, will be depreciated
    laugh_found = 0
    laugh_strings = ['(laughing)', '(laughs)', '(laughter)', '(chuckles)', '(chuckling)']
    for laugh in laugh_strings:
        if laugh in line:
            laugh_found = 1
            line = line.replace(laugh, '')
    return laugh_found, line


def find_laugh(line):
    laugh_found = 0
    laugh_strings = ['(laughing)', '(laughs)', '(laughter)', '(chuckles)', '(chuckling)']
    for laugh in laugh_strings:
        if laugh in line:
            laugh_found = 1
    return laugh_found


def clean_subs(subs):                   # replicated below, may be depreciated

    cleaned_lines = []

    single_lines = generate_single_lines(subs)
    for line in single_lines:
        entire_line_italic, line = italic_clean(line)

        entire_line_music, line = music_clean(line)

        laugh_found, line = laugh_clean(line)

        speaker, line = speaker_clean(line)

        entire_line_parenthetical, line = parenthetical_clean(line)

        cleaned_lines.append(line)

    return cleaned_lines


def clean_and_flag_subs(subs):
    italic_flags = []
    music_flags = []
    laugh_flags = []
    speakers = []
    entire_line_parentheticals = []

    cleaned_lines = []

    single_lines = generate_single_lines(subs)
    for line in single_lines:
        entire_line_italic, line = italic_clean(line)
        italic_flags.append(entire_line_italic)

        entire_line_music, line = music_clean(line)
        music_flags.append(entire_line_music)

        laugh_found, line = laugh_clean(line)
        laugh_flags.append(laugh_found)

        speaker, line = speaker_clean(line)
        speakers.append(speaker)

        entire_line_parenthetical, line = parenthetical_clean(line)
        entire_line_parentheticals.append(entire_line_parenthetical)

        cleaned_lines.append(line)

    return cleaned_lines, italic_flags, music_flags, laugh_flags, speakers, entire_line_parentheticals


def clean_line(line):                   # replicated below, may be depreciated

    entire_line_italic, line = italic_clean(line)

    entire_line_music, line = music_clean(line)

    laugh_found, line = laugh_clean(line)

    speaker, line = speaker_clean(line)

    entire_line_parenthetical, line = parenthetical_clean(line)

    line = midsentence_interjection(line)

    return line


def remove_blanks(cleaned_lines):
    blanks_removed = []

    for line in cleaned_lines:
        if line:
            blanks_removed.append(line)

    return blanks_removed


def partition_sentences(input_lines, nlp):
    doc = nlp(' '.join(input_lines))
    sentences = []
    for sent in doc.sents:
        sentences.append(sent.text)

    return sentences


def midsentence_interjection(line):
    interjection_strings = [', uh,', ', um']
    for interjection in interjection_strings:
        found_interjection = line.find(interjection)
        if found_interjection != -1:
            line = line[:found_interjection] + line[(len(line) - found_interjection - len(interjection)) * -1:]
    return line
