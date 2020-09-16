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
        entire_line_parenthetical = 1
        line = ''
    return entire_line_parenthetical, line


def italic_clean(line):
    entire_line_italic = 0
    if line[:3] == '<i>' and line[-4:] == '</i>':
        entire_line_italic = 1
    line = line.replace('<i>', '')
    line = line.replace('</i>', '')
    return entire_line_italic, line


def speaker_clean(line):
    colon_find = line.find(':')
    speaker = 'none'
    if line[0:colon_find].isupper():
        speaker = line[0:colon_find]
        line = line[colon_find + 2:]
    return speaker, line


def laugh_clean(line):
    laugh_found = 0
    laugh_strings = ['(laughing)', '(laughs)', '(chuckles)', '(laughter)']
    for laugh in laugh_strings:
        if laugh in line:
            laugh_found = 1
            line = line.replace(laugh, '')
    return laugh_found, line


def clean_subs(subs):

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
    parenthetical_flags = []

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
        parenthetical_flags.append(entire_line_parenthetical)

        cleaned_lines.append(line)

    return cleaned_lines, italic_flags, music_flags, laugh_flags, speakers, parenthetical_flags


def remove_blanks(cleaned_lines):
    blanks_removed = []

    for line in cleaned_lines:
        if line:
            blanks_removed.append(line)

    return blanks_removed
