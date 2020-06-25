import cv2
import os

# create frames from video file
title = 'hobbs_shaw'
scale_percent = 100     # optional image scaling

cap = cv2.VideoCapture('input_videos/' + title + '.mkv')
output_directory = os.path.join('dialogue_frames/', title)
os.mkdir(output_directory)

if not cap.isOpened():
    print("Error opening video stream or file")

frame_rate = cap.get(5)              # frame rate
print('Video frame rate:', frame_rate)  # most movie files will have a frame_rate of 23.976 frames per second


# because of the 23.976 fps frame_rate, we can't just capture every 24 frames
count = 1   # starts at 1, because first image isn't saved until 25th frame (timestamp of 1.001 seconds)
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    if cap.get(0) / 1000 >= count:     # first frame after 1 second, 2 seconds, etc.
        filename = output_directory + '/' + title + '_frame%d.jpg' % count

        # optional image scaling
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dimensions = (width, height)
        frame = cv2.resize(frame, dimensions)

        print(cap.get(0), cap.get(1), count)
        cv2.imwrite(filename, frame)
        count += 1

cap.release()
print('Extracted', count, 'frames')
