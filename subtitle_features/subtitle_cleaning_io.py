import spacy


def concat_sep_lines(text):
    """
    returns one or two lines of subtitles
    if one line of subtitle, return the original text
    if two lines of subtitles spoken by one character, remove the line break and return the subtitle
    if two lines of subtitles spoken by two characters, break them into separate subtitles
    """
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
    """
    returns a list of subtitle lines after concatenation and separation
    """
    single_lines = []
    for sub_object in subs:
        text = sub_object.text
        line_a, line_b = concat_sep_lines(text)
        single_lines.append(line_a)
        if line_b != 0:
            single_lines.append(line_b)
    return single_lines


def find_music(line):
    """
    returns a flag denoting if music is present
    """
    if '♪' in line:
        return 1
    else:
        return 0


def clean_music(line):
    """
    cleans line by removing all content if music is present
    """
    if '♪' in line:
        return ''
    else:
        return line


def find_parenthetical(line):
    """
    returns parenthetical content, in both forms of brackets: () and []
    """
    parenthetical_content = None

    paren_open_find = line.find('(')
    if paren_open_find != -1:
        paren_close_find = line.find(')')
        parenthetical_content = line[paren_open_find + 1:paren_close_find].strip()

    paren_open_find = line.find('[')
    if paren_open_find != -1:
        paren_close_find = line.find(']')
        parenthetical_content = line[paren_open_find + 1:paren_close_find].strip()

    return parenthetical_content


def find_el_parenthetical(line):
    """
    returns flag denoting if the entire line is parenthetical content, in both forms of brackets: () and []
    """
    entire_line_parenthetical = 0
    if line[:1] == '(' and line[-1:] == ')':
        entire_line_parenthetical = 1
    elif line[:1] == '[' and line[-1:] == ']':
        entire_line_parenthetical = 1
    return entire_line_parenthetical


def clean_parenthetical(line):
    """
    cleans line content by removing parenthetical content, in both forms of brackets: () and []
    """
    paren_open_find = line.find('(')
    if paren_open_find != -1:
        paren_close_find = line.find(')')
        line = (line[:paren_open_find] + line[paren_close_find + 1:]).strip()
    paren_open_find = line.find('[')
    if paren_open_find != -1:
        paren_close_find = line.find(']')
        line = (line[:paren_open_find] + line[paren_close_find + 1:]).strip()

    return line


def italic_clean(line):
    """
    cleans line content by removing italic content, and also returns a flag denoting if the entire line is italics
    """
    entire_line_italic = 0
    if line[:3] == '<i>' and line[-4:] == '</i>':
        entire_line_italic = 1
    line = line.replace('<i>', '')
    line = line.replace('</i>', '')
    return entire_line_italic, line


def find_italic(line):
    """
    returns content in italics if the entire line is italics
    will be refactored to differentiate between entire line, and portions, as well as multiple sets of italics
    """
    entire_line_italic = None
    if line[:3] == '<i>' and line[-4:] == '</i>':
        entire_line_italic = line[3:-4]
    return entire_line_italic


def find_el_italic(line):
    """
    returns flag denoting if entire line is in italics
    """
    if line[:3] == '<i>' and line[-4:] == '</i>':
        return 1
    else:
        return 0


def clean_italic(line):
    """
    cleans line by removing italic tags
    """
    line = line.replace('<i>', '').replace('</i>', '')
    return line


def find_speaker(line):
    """
    returns offscreen speaker name, if labeled
    """
    colon_find = line.find(':')
    if colon_find != -1 and line[0:colon_find].isupper():
        speaker = line[0:colon_find]
    else:
        speaker = None
    return speaker


def clean_speaker(line):
    """
    cleans offscreen speaker name, if labeled
    """
    colon_find = line.find(':')
    if colon_find != -1 and line[0:colon_find].isupper():
        line = line[colon_find + 2:]
    return line


def find_laugh(line):
    """
    returns flag denoting if laughter found
    """
    laugh_found = 0
    laugh_strings = ['laughing', 'laughs', 'laughter', 'chuckles', 'chuckling', 'giggles', 'giggling']
    for laugh in laugh_strings:
        if laugh in line.lower():
            laugh_found = 1
    return laugh_found


def clean_line(line):
    """
    returns cleaned line
    """
    line = clean_parenthetical(line)
    line = clean_italic(line)
    line = clean_music(line)
    line = clean_speaker(line)
    line = clean_midsentence_interjection(line)

    return line


def remove_blanks(cleaned_lines):
    """
    removes blank lines from list of cleaned lines
    """
    blanks_removed = []

    for line in cleaned_lines:
        if line:
            blanks_removed.append(line.strip())

    return blanks_removed


def partition_sentences(input_lines, nlp):
    """
    returns a list of sentences based on input subtitle lines
    """
    doc = nlp(' '.join(input_lines))
    sentences = []
    for sent in doc.sents:
        sentences.append(sent.text)

    return sentences


def clean_midsentence_interjection(line):
    """
    cleans midsentence interjections
    """
    interjection_strings = [', uh,', ', um', ', you know,']
    for interjection in interjection_strings:
        found_interjection = line.find(interjection)
        if found_interjection != -1:
            line = line[:found_interjection] + line[(len(line) - found_interjection - len(interjection)) * -1:]
    return line


def find_midsentence_interjection(line):
    """
    returns a flag denoting if a midsentence interjection was found
    """
    interjection_strings = [', uh,', ', um', ', you know,']
    for interjection in interjection_strings:
        found_interjection = line.find(interjection)
        if found_interjection != -1:
            return 1

    return 0
