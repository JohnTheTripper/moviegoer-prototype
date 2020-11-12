import os
from deepface import DeepFace

"""This exists in a separate .py file because it may not run in a Jupyter notebook"""

film = 'plus_one_2019'
frame_number = 593
frame_folder = os.path.join('../frame_per_second', film)
img_path = frame_folder + '/' + film + '_frame_' + str(frame_number) + '.jpg'

obj = DeepFace.analyze(img_path, actions=['age', 'gender', 'race', 'emotion'])
print(obj["age"], " years old ", obj["dominant_race"], " ", obj["dominant_emotion"], " ", obj["gender"])
print(obj)
