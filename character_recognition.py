import dlib
import face_recognition
import os
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

# fixing issue of using GPU
print(dlib.DLIB_USE_CUDA)

# choose film and frames
film = 'extremely_wicked'
frame_choice = list(range(954, 998))
dialogue_folder = os.path.join('dialogue_frames', film)

face_feature_list = []
encodings_list_np = []

# loop through each frame, detect each face, and encode as an array
for x in frame_choice:
    img_path = dialogue_folder + '/' + film + '_frame' + str(x) + '.jpg'
    image = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(image)

    print('Found ' + str(len(face_locations)) + ' face(s) in frame ' + str(x))
    encodings = face_recognition.face_encodings(image, face_locations)

    for encoding in encodings:
        encodings_list_np.append(np.array(encoding).flatten())

print('Total faces found: ' + str(len(encodings_list_np)))
encodings_list_np = np.array(encodings_list_np)
print(encodings_list_np.shape)

# HAC will find more than two characters
hac = AgglomerativeClustering(n_clusters=None, distance_threshold=1, linkage='ward').fit(encodings_list_np)
hac_labels = hac.labels_
print('Number of clusters:', hac.n_clusters_)
print(hac_labels)

# KMeans can only find two characters
kmeans = KMeans(n_clusters=2).fit(encodings_list_np)
kmeans_labels = kmeans.labels_
print(kmeans_labels)
