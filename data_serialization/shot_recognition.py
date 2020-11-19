import os
import sys
sys.path.append('../venv/lib/python3.6/site-packages')
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

"""
uses VGG16 image recognition model to vectorize image files and exports to a single numpy file
does not work on animated movies
to be run in Docker container to take advantage of GPU, using "tensorflow:latest-gpu" Docker image
to use, change film and frame_choice
"""

film = 'black_and_blue_2019'
frame_choice = range(1, 6478)   # (1, number of frame files plus one)

'''change above parameters'''

serialized_object_directory = '../serialized_objects/'
film_directory = os.path.join(serialized_object_directory, film)

image_directory = '../frame_per_second/'
print(os.path.join(image_directory, film))

model = VGG16(weights='imagenet', include_top=False)

vgg16_feature_list = []

for x in frame_choice:
    img_path = os.path.join(image_directory, film) + '/' + film + '_frame_' + str(x) + '.jpg'
    img = image.load_img(img_path, target_size=(256, 256))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    vgg16_feature = model.predict(img_data)
    vgg16_feature_np = np.array(vgg16_feature)
    vgg16_feature_list.append(vgg16_feature_np.flatten())

    x += 1

vgg16_feature_list_np = np.array(vgg16_feature_list)

print(vgg16_feature_list_np.shape)

np.save(os.path.join(film_directory, 'vgg16_features.npy'), vgg16_feature_list_np)
