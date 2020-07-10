import math
import face_recognition
from sklearn.cluster import AgglomerativeClustering
import numpy as np


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
