import math
import face_recognition
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import datetime
import pysrt
import pyAudioAnalysis.audioSegmentation


# visual functions
# this function returns three lists, but to avoid running a costly analysis 3x, they're combined into one function
def frame_attribution_analysis(dialogue_folder, film, frame_choice):
    """
    returns three lists for each frame: primary character (A or B), count of faces found, and a mouth_open flag
    """
    encodings_list = []
    faces_found = []
    mouth_open_list = []

    for x in frame_choice:
        img_path = dialogue_folder + '/' + film + '_frame' + str(x) + '.jpg'
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1)
        face_landmarks_list = face_recognition.face_landmarks(image, face_locations)

        # print('Found ' + str(len(face_locations)) + ' face(s) in frame ' + str(x))
        frame_encodings_list = face_recognition.face_encodings(image, face_locations)

        if frame_encodings_list:
            encoding = frame_encodings_list[0]
            encodings_list.append(encoding)
            faces_found.append(len(face_locations))
            face_landmarks = face_landmarks_list[0]
            mouth_open_list.append(mouth_open_check(face_landmarks))
        else:
            faces_found.append(0)
            mouth_open_list.append(0)

    encodings_list_np = np.array(encodings_list)
    hac = AgglomerativeClustering(n_clusters=None, distance_threshold=1).fit(encodings_list_np)
    hac_labels = hac.labels_

    primary_character_list = []
    y = 0
    for x in faces_found:
        if x != 0:
            primary_character_list.append(chr(hac_labels[y] + 65))  # converts numbers to Unicode chars (A, B, etc.)
            y += 1
        else:
            primary_character_list.append(0)

    return primary_character_list, faces_found, mouth_open_list


def count_faces(dialogue_folder, film, frame_choice):
    """
    returns a list of counts of faces found in each frame
    """
    faces_found = []

    for x in frame_choice:
        img_path = dialogue_folder + '/' + film + '_frame' + str(x) + '.jpg'
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1)

        # print('Found ' + str(len(face_locations)) + ' face(s) in frame ' + str(x))
        # frame_encodings_list = face_recognition.face_encodings(image, face_locations)

        if face_locations:
            faces_found.append(len(face_locations))
        else:
            faces_found.append(0)

    return faces_found


def get_primary_character(dialogue_folder, film, frame_choice):
    """
    returns a list of the primary character found in each frame
    """
    encodings_list = []
    faces_found = []

    for x in frame_choice:
        img_path = dialogue_folder + '/' + film + '_frame' + str(x) + '.jpg'
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1)

        # print('Found ' + str(len(face_locations)) + ' face(s) in frame ' + str(x))
        frame_encodings_list = face_recognition.face_encodings(image, face_locations)

        if frame_encodings_list:
            encoding = frame_encodings_list[0]
            encodings_list.append(encoding)
            faces_found.append(len(face_locations))
        else:
            faces_found.append(0)

    encodings_list_np = np.array(encodings_list)
    hac = AgglomerativeClustering(n_clusters=None, distance_threshold=1).fit(encodings_list_np)
    hac_labels = hac.labels_

    primary_character_list = []
    y = 0
    for x in faces_found:
        if x != 0:
            primary_character_list.append(chr(hac_labels[y] + 65))  # converts numbers to Unicode chars (A, B, etc.)
            y += 1
        else:
            primary_character_list.append(0)

    return primary_character_list


def analyze_mouth_open(dialogue_folder, film, frame_choice):
    """
    returns a list of flags, for if a character has their mouth open, in each frame
    returns a 0 if no face detected in frame
    """
    mouth_open_list = []

    for x in frame_choice:
        img_path = dialogue_folder + '/' + film + '_frame' + str(x) + '.jpg'
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1)
        face_landmarks_list = face_recognition.face_landmarks(image, face_locations)

        # print('Found ' + str(len(face_locations)) + ' face(s) in frame ' + str(x))
        frame_encodings_list = face_recognition.face_encodings(image, face_locations)

        if frame_encodings_list:
            face_landmarks = face_landmarks_list[0]
            mouth_open_list.append(mouth_open_check(face_landmarks))
        else:
            mouth_open_list.append(0)

    return mouth_open_list


def get_lip_height(lip):
    for i in [2, 3, 4]:
        sum = 0
        distance = math.sqrt((lip[i][0] - lip[12-i][0])**2 + (lip[i][1] - lip[12-i][1])**2)
        sum += distance
    return sum / 3


def get_mouth_height(top_lip, bottom_lip):
    for i in [8, 9, 10]:
        sum = 0
        distance = math.sqrt((top_lip[i][0] - bottom_lip[18-i][0])**2 + (top_lip[i][1] - bottom_lip[18-i][1])**2)
        sum += distance
    return sum / 3


def mouth_open_check(face_landmarks, open_ratio=.8):
    top_lip = face_landmarks['top_lip']
    bottom_lip = face_landmarks['bottom_lip']

    top_lip_height = get_lip_height(top_lip)
    bottom_lip_height = get_lip_height(bottom_lip)
    mouth_height = get_mouth_height(top_lip, bottom_lip)

    if mouth_height > min(top_lip_height, bottom_lip_height) * open_ratio:
        return 1
    else:
        return 0


# subtitle functions
def load_subtitles(subs_file):
    subs = pysrt.open(subs_file)
    subs.insert(0, subs[0])     # dummy at 0, because .srt files are explicitly numbered, and start at 1

    return subs


def frame_to_time(frame_number):
    seconds = frame_number % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    timestamp = datetime.time(hours, minutes, seconds)

    return timestamp


def add_time_offset(time_object, offset):
    datetime_object = datetime.datetime(year=2000, month=1, day=1, hour=time_object.hour, minute=time_object.minute,
                                        second=time_object.second, microsecond=time_object.microsecond)
    datetime_offset = datetime.timedelta(milliseconds=offset)
    datetime_set_off = datetime_object + datetime_offset
    time_added = datetime.time(hour=datetime_set_off.hour, minute=datetime_set_off.minute,
                               second=datetime_set_off.second, microsecond=datetime_set_off.microsecond)
    return time_added


def subtract_time_offset(time_object, offset):
    datetime_object = datetime.datetime(year=2000, month=1, day=1, hour=time_object.hour, minute=time_object.minute,
                                        second=time_object.second, microsecond=time_object.microsecond)
    datetime_offset = datetime.timedelta(milliseconds=offset)
    datetime_set_off = datetime_object - datetime_offset
    time_subtracted = datetime.time(hour=datetime_set_off.hour, minute=datetime_set_off.minute,
                                    second=datetime_set_off.second, microsecond=datetime_set_off.microsecond)
    return time_subtracted


def analyze_onscreen_subtitles(subs, frame_choice):
    subtitle_onscreen = []
    for frame in frame_choice:
        time = frame_to_time(frame)

        # check for presence of subtitle, then if the subtitle is a parenthetical subtitle, which should be excluded
        if subs.at(time) and (subs.at(time).text[0] != '(' or subs.at(time).text[-1] != ')'):
            subtitle_onscreen.append(1)
        elif subs.at(time.replace(microsecond=999000)) and (subs.at(time.replace(microsecond=999000)).text[0] != '(' or
                                                            subs.at(time.replace(microsecond=999000)).text[-1] != ')'):
            subtitle_onscreen.append(1)
        else:
            subtitle_onscreen.append(0)

    return subtitle_onscreen


# audio functions
def cluster_voices(audio_file, plot=True):
    clusters = pyAudioAnalysis.audioSegmentation.speaker_diarization(audio_file, n_speakers=2, mid_window=0.8,
                                                                     mid_step=0.1, short_window=0.02, lda_dim=0,
                                                                     plot_res=plot)

    speaker_list = []
    x = 0
    while x < len(clusters):
        if sum(clusters[x:x + 10]) == 5:  # ignore tie
            speaker_list.append(0)
        else:
            speaker_list.append(chr(int(round(np.mean(clusters[x:x + 10]))) + 77))  # convert 0 and 1 to M and N
        x += 10

    return speaker_list


def analyze_audible_sound(audio_file, plot=False):
    sampling_rate, signal = pyAudioAnalysis.audioBasicIO.read_audio_file(audio_file)
    segments_with_sound = pyAudioAnalysis.audioSegmentation.silence_removal(signal, sampling_rate, st_win=0.05,
                                                                            st_step=0.025, smooth_window=0.5,
                                                                            weight=0.3, plot=plot)
    audible_sound = []

    for frame in range(0, 58):
        sound_found = 0

        for segment in segments_with_sound:
            sound_start = segment[0]
            sound_end = segment[1]
            if sound_start <= frame <= sound_end:
                sound_found = 1
            if sound_start <= frame + .25 <= sound_end:
                sound_found = 1
            if sound_start <= frame + .5 <= sound_end:
                sound_found = 1
            if sound_start <= frame + .75 <= sound_end:
                sound_found = 1
            if sound_start <= frame + .999 <= sound_end:
                sound_found = 1

        if sound_found:
            audible_sound.append(1)
        else:
            audible_sound.append(0)

    return audible_sound


