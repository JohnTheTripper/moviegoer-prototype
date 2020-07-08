import math


def get_lip_height(lip):
    for i in [2 ,3 ,4]:
        sum = 0
        distance = math.sqrt((lip[i][0] - lip[12-i][0])**2 + (lip[i][1] - lip[12-i][1])**2)
        sum += distance
    return sum / 3


def get_mouth_height(top_lip, bottom_lip):
    for i in [8 ,9 ,10]:
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
