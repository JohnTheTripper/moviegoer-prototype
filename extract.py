import cv2
import math

# create frames from video file
title = 'preggoland'
cap = cv2.VideoCapture('sample_videos/preggoland.mkv')   # capturing the video from the given path

if not cap.isOpened():
    print("Error opening video stream or file")

frameRate = cap.get(5)  # framerate

count = 0
x = 1
while cap.isOpened():
    frameId = cap.get(1)                # current frame number
    ret, frame = cap.read()
    if not ret:
        break
    if frameId % math.floor(frameRate * 8) == 0: # extract a frame once every 8 seconds
        filename = '/media/collect/Toshiba/movie_frames/' + title + '_frame%d.jpg' % count
        count += 1
        cv2.imwrite(filename, frame)
cap.release()
print('Extracted', count, 'frames')
