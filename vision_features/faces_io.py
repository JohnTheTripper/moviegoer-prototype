from vision_features_io import *
import face_recognition
from deepface import DeepFace
import math
import os


def primary_character_flag(locations):
    if len(locations) == 1:
        return 1
    else:
        local_locations = locations.copy()
        first_face = local_locations.pop(0)
        first_face_size = first_face[2] - first_face[0]
        primary_char_threshold = .75  # 75% of the first face
        for face in local_locations:
            if face[2] - face[0] > (first_face_size * primary_char_threshold):
                return 0
        else:
            return 1


def third_points_alignment(character, frame):
    thirds_point_a, thirds_point_b, thirds_point_c, thirds_point_d = thirds_points(frame)

    if character[3] <= thirds_point_a[0] <= character[1] and character[0] <= thirds_point_a[1] <= character[2]:
        return 'left'
    elif character[3] <= thirds_point_b[0] <= character[1] and character[0] <= thirds_point_b[1] <= character[2]:
        return 'right'
    else:
        return None


def horizontal_center_alignment(character, frame):
    vertical_center = center_point(frame)[0]

    if character[1] > vertical_center and character[3] > vertical_center:
        return 'right'
    elif character[1] < vertical_center and character[3] < vertical_center:
        return 'left'
    else:
        return None


def horizontal_distance_from_center(character, frame):
    vertical_center = center_point(frame)[0]

    if character[1] > vertical_center and character[3] > vertical_center:
        return character[1] - vertical_center
    elif character[1] < vertical_center and character[3] < vertical_center:
        return vertical_center - character[3]
    else:
        return None


def get_face_size(locations, frame):
    face_size = pow((locations[1] - locations[3]), 2)
    image_size = frame.shape[0] * frame.shape[1]
    face_size *= 100

    return round(face_size / image_size, 2)


def analyze_mouth_open(frame_folder, film, frame_choice):
    """
    returns a list of flags, for if a character has their mouth open, in each frame
    returns a 0 if no face detected in frame
    """
    mouth_open_list = []

    for x in frame_choice:
        img_path = frame_folder + '/' + film + '_frame' + str(x) + '.jpg'
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


def get_primary_char_emotion(film, frame_number):
    frame_folder = os.path.join('../frame_per_second', film)
    img_path = frame_folder + '/' + film + '_frame_' + str(frame_number) + '.jpg'

    obj = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=False)
    return obj["dominant_emotion"]
