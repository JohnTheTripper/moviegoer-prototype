import cv2
import os

# create frames from video file
title = 'extremely_wicked'
cap = cv2.VideoCapture('input_videos/' + title + '.mkv')   # capturing the video from the given path

output_directory = os.path.join('dialogue_frames/', title)
os.mkdir(output_directory)

if not cap.isOpened():
    print("Error opening video stream or file")

frameRate = cap.get(5)  # framerate
print(frameRate)

count = 0
x = 1
while cap.isOpened():
    frameId = cap.get(1)                # current frame number
    ret, frame = cap.read()

    if not ret:
        break
    if cap.get(1) % 24 == 0:     # frame every 1 second, on a 24fps video
        filename = output_directory + '/' + title + '_frame%d.jpg' % count
        count += 1

        # new, resize
        # percent by which the image is resized
        scale_percent = 100
        # calculate the 33 percent of original dimensions
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        # dsize
        dsize = (width, height)
        # resize image
        frame = cv2.resize(frame, dsize)

        print(cap.get(0), cap.get(1), count)
        cv2.imwrite(filename, frame)

cap.release()
print('Extracted', count, 'frames')
